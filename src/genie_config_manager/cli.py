import json
import os

import typer
import yaml
from dotenv import load_dotenv, set_key
from typing_extensions import Annotated

from genie_config_manager.genie_service import (
    get_genie_service,
)
from genie_config_manager.schemas import (
    GenieConfig,
    GenieDataSources,
    GenieInstructions,
    GenieLoadOptions,
    GenieSchemaSettings,
)
from genie_config_manager.templates import GENIE_CONFIG_TEMPLATE

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
        str,
        typer.Option(
            help="The Databricks CLI profile to use.",
            envvar="DATABRICKS_PROFILE",
        ),
    ],
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
    if not profile:
        raise typer.BadParameter(
            "Databricks CLI profile not provided. Please set DATABRICKS_PROFILE env var or use --profile argument."
        )

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
        str,
        typer.Option(
            help="The Databricks CLI profile to use.",
            envvar="DATABRICKS_PROFILE",
        ),
    ],
):
    """
    Pull the latest configuration from a Genie Space.
    """
    typer.echo(f"Pulling configuration from Genie Space: {space_id}")
    typer.echo(f"Using profile: {profile}")
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
        str,
        typer.Option(
            help="The Databricks CLI profile to use.",
            envvar="DATABRICKS_PROFILE",
        ),
    ],
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
    typer.echo(f"Using profile: {profile}")

    if not profile:
        raise typer.BadParameter(
            "Databricks CLI profile not provided. Please set DATABRICKS_PROFILE env var or use --profile argument."
        )

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


if __name__ == "__main__":
    app()
