import json
import os

import typer
import yaml
from dotenv import load_dotenv, set_key
from typing_extensions import Annotated

from genie_toolkit.genie_service import (
    get_genie_service,
)
from genie_toolkit.optimiser import run_optimisation
from genie_toolkit.schemas import (
    GenieBenchmarks,
    GenieConfig,
    GenieDataSources,
    GenieInstructions,
    GenieLoadOptions,
    GenieSchemaSettings,
)
from genie_toolkit.templates import GENIE_CONFIG_TEMPLATE

app = typer.Typer()
load_dotenv()


def update_env(key: str, value: str):
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, "a").close()
    set_key(env_path, key, value)


@app.command()
def init(
    warehouse_id: Annotated[
        str | None,
        typer.Option(
            help="The ID of the SQL warehouse to use.",
            envvar="WAREHOUSE_ID",
        ),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            help="The Databricks CLI profile to use.",
            envvar="DATABRICKS_PROFILE",
        ),
    ] = None,
    config: Annotated[
        str,
        typer.Option(
            help="Path to the YAML configuration file.",
        ),
    ] = "genie.yml",
):
    """
    Initialize a new genie project.
    """
    if not warehouse_id:
        warehouse_id = typer.prompt("Enter your Databricks SQL Warehouse ID")
    if not profile:
        profile = typer.prompt(
            "Enter your Databricks CLI Profile name", default="DEFAULT"
        )

    assert isinstance(warehouse_id, str)
    assert isinstance(profile, str)

    update_env("WAREHOUSE_ID", warehouse_id)
    update_env("DATABRICKS_PROFILE", profile)

    config_content = GENIE_CONFIG_TEMPLATE
    if os.path.exists(config):
        overwrite = typer.confirm(
            f"'{config}' already exists. Do you want to overwrite it?"
        )
        if not overwrite:
            print("Aborted!")
            raise typer.Abort()

    with open(config, "w") as f:
        f.write(config_content)

    typer.echo(f"✨ Genie project initialized. '{config}' created.")


@app.command()
def create(  # TODO: Consolidate into load_config function shared between create and push
    warehouse_id: Annotated[
        str,
        typer.Option(
            help="The ID of the SQL warehouse to use.",
            envvar="WAREHOUSE_ID",
        ),
    ],
    profile: Annotated[
        str | None,
        typer.Option(
            help="The Databricks CLI profile to use. Optional in Databricks notebook environments.",
            envvar="DATABRICKS_PROFILE",
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(
            help="Title of the Genie Space.",
        ),
    ] = None,
    config: Annotated[
        typer.FileText,
        typer.Option(
            help="Path to the YAML configuration file.",
        ),
    ] = "genie.yml",
    parent_path: Annotated[
        str | None,
        typer.Option(
            help="Parent path for the Genie Space.",
        ),
    ] = None,
):
    """
    Create a new Genie Space.
    """
    if not warehouse_id:
        raise typer.BadParameter(
            "SQL warehouse ID not provided. Please set WAREHOUSE_ID env var or use --warehouse_id argument."
        )

    if not title:
        title = typer.prompt("Enter the name of your genie space.")

    genie_service = get_genie_service(profile)

    yaml_config = yaml.safe_load(config)
    genie_config = yaml_config.get("genie", {})

    data_tables = genie_config["data_sources"]["tables"]

    if len(data_tables) > 0:
        data_sources = GenieDataSources.from_uc(
            genie_service.wc, data_tables, GenieLoadOptions()
        )
    else:
        data_sources = None

    sample_questions = GenieConfig.from_dict(genie_config.get("sample_questions", []))

    instructions = GenieInstructions.from_dict(genie_config.get("instructions", {}))

    genie_schema_settings = GenieSchemaSettings(
        version=1,
        config=sample_questions,
        data_sources=data_sources,
        instructions=instructions,
    )

    genie_space = genie_service.create(
        warehouse_id,
        genie_schema_settings,
        title=title,
        parent_path=parent_path,
    )

    update_env("GENIE_SPACE_ID", genie_space.space_id)
    typer.echo(f"Genie Space '{genie_space.title}' created successfully.")
    typer.echo("Space ID has been stored in your local .env file.")


@app.command()
def pull(
    space_id: Annotated[
        str,
        typer.Option(
            help="The ID of the genie space to track.",
            envvar="GENIE_SPACE_ID",
        ),
    ],
    profile: Annotated[
        str | None,
        typer.Option(
            help="The Databricks CLI profile to use. Optional in Databricks notebook environments.",
            envvar="DATABRICKS_PROFILE",
        ),
    ] = None,
):
    """
    Pull the latest configuration from a Genie Space.
    """
    typer.echo(f"Pulling configuration from Genie Space: {space_id}")
    genie_service = get_genie_service(profile)
    genie_space = genie_service.wc.genie.get_space(
        space_id=space_id, include_serialized_space=True
    )
    genie_schema_settings = GenieSchemaSettings(
        **json.loads(genie_space.serialized_space)
    )
    genie_schema_settings.to_yaml()

    update_env("GENIE_SPACE_ID", space_id)
    typer.echo(f"Current active genie space id: {space_id}")
    typer.echo("Successfully pulled config from workspace.")


@app.command()
def push(
    space_id: Annotated[
        str,
        typer.Option(
            help="The ID of the genie space to update.",
            envvar="GENIE_SPACE_ID",
        ),
    ],
    profile: Annotated[
        str | None,
        typer.Option(
            help="The Databricks CLI profile to use. Optional in Databricks notebook environments.",
            envvar="DATABRICKS_PROFILE",
        ),
    ] = None,
    config: Annotated[
        typer.FileText,
        typer.Option(
            help="Path to the YAML configuration file.",
        ),
    ] = "genie.yml",
    title: Annotated[
        str | None,
        typer.Option(
            help="Title of the Genie Space.",
        ),
    ] = None,
):
    """
    Push local configuration changes to a Genie Space.
    """
    typer.echo(f"Pushing configuration to Genie Space: {space_id}")

    if not space_id:
        raise typer.BadParameter(
            "Space ID not provided. Please set GENIE_SPACE_ID env var or use --space_id argument."
        )

    genie_service = get_genie_service(profile)

    yaml_config = yaml.safe_load(config)
    genie_config = yaml_config.get("genie", {})

    genie_schema_settings = GenieSchemaSettings(
        version=1,
        config=genie_config.get("sample_questions"),
        data_sources=genie_config.get("data_sources", {"tables": None}),
        instructions=genie_config.get("instructions"),
    )

    genie_space = genie_service.update(
        space_id,
        genie_schema_settings,
        title=title,
    )

    typer.echo(f"Genie Space '{genie_space.title}' updated successfully.")


@app.command()
def optimise(
    space_id: Annotated[
        str,
        typer.Option(
            help="The ID of the Genie Space to optimise.",
            envvar="GENIE_SPACE_ID",
        ),
    ],
    profile: Annotated[
        str | None,
        typer.Option(
            help="The Databricks CLI profile to use. Optional in Databricks notebook environments.",
            envvar="DATABRICKS_PROFILE",
        ),
    ] = None,
    config: Annotated[
        typer.FileText,
        typer.Option(
            help="Path to the YAML configuration file.",
        ),
    ] = "genie.yml",
    model_endpoint: Annotated[
        str,
        typer.Option(
            help="Databricks Foundation Model serving endpoint for LLM judging.",
        ),
    ] = "databricks-meta-llama-3-3-70b-instruct",
    max_evals: Annotated[
        int,
        typer.Option(
            help="Maximum number of evaluations for the optimisation loop.",
        ),
    ] = 100,
    train_ratio: Annotated[
        float,
        typer.Option(
            help="Fraction of benchmarks to use for training.",
        ),
    ] = 0.7,
    val_ratio: Annotated[
        float,
        typer.Option(
            help="Fraction of benchmarks to use for validation.",
        ),
    ] = 0.15,
):
    """
    Optimise a Genie Space configuration using benchmark questions.

    Uses GEPA (evolutionary text optimisation) to iteratively improve the
    genie config's text components against benchmark performance.
    """
    yaml_config = yaml.safe_load(config)
    genie_config = yaml_config.get("genie", {})

    benchmarks_data = genie_config.get("benchmarks")
    if not benchmarks_data or not benchmarks_data.get("questions"):
        typer.echo(
            "No benchmarks found in config. Add benchmarks with questions and expected answers to genie.yml."
        )
        raise typer.Exit(code=1)

    benchmarks = GenieBenchmarks.from_dict(benchmarks_data)

    genie_schema_settings = GenieSchemaSettings(
        version=1,
        config=genie_config.get("sample_questions"),
        data_sources=genie_config.get("data_sources", {"tables": None}),
        instructions=genie_config.get("instructions"),
        benchmarks=benchmarks,
    )

    genie_service = get_genie_service(profile)

    typer.echo(f"Optimising Genie Space: {space_id}")
    typer.echo(f"Benchmarks: {len(benchmarks.questions)} questions")
    typer.echo(f"Max evaluations: {max_evals}")

    results = run_optimisation(
        genie_service=genie_service,
        space_id=space_id,
        settings=genie_schema_settings,
        benchmarks_questions=benchmarks.questions,
        model_endpoint=model_endpoint,
        max_evals=max_evals,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Write optimised config back
    results["best_settings"].to_yaml()

    typer.echo(f"Train score: {results['train_score']:.2%}")
    typer.echo(f"Validation score: {results['val_score']:.2%}")
    typer.echo(f"Test questions held out: {results['test_count']}")
    typer.echo("Optimised configuration written to genie.yml")


if __name__ == "__main__":
    app()
