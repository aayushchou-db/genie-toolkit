import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml
from databricks.sdk import WorkspaceClient
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from genie_config_manager.templates import GENIE_CONFIG_TEMPLATE

TTables = TypeVar("TTables", bound="GenieTableConfig")
TInstructions = TypeVar("TInstructions", bound="GenieInstructions")
TSamples = TypeVar("TSamples", bound="GenieConfig")
TSources = TypeVar("TSources", bound="GenieDataSources")


@dataclass
class GenieLoadOptions:
    include_column_configs: bool = True
    include_example_values: bool = True
    include_value_dictionary: bool = True


class GenieSampleQuestion(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: list[str] | None = None


class GenieConfig(BaseModel):
    sample_questions: list[GenieSampleQuestion]

    @classmethod
    def from_dict(cls: Type[TSamples], data: List[Dict[str, Any]]) -> TSamples:
        """Factory method to create settings from a dictionary."""
        sample_questions = [GenieSampleQuestion(**item) for item in data]
        return cls(sample_questions=sample_questions)


class GenieColumnConfig(BaseModel):
    column_name: str
    get_example_values: bool = False
    build_value_dictionary: bool = False


class GenieTableConfig(BaseModel):
    identifier: str = Field(default_factory=lambda: uuid.uuid4().hex)
    description: Optional[List[str | None]] = None
    column_configs: Optional[List[GenieColumnConfig]] = None

    @classmethod
    def from_dict(cls: Type[TTables], data: Dict[str, Any]) -> TTables:
        """Factory method to create a table config from a dictionary."""
        return cls(**data)

    @classmethod
    def from_unity_catalog(
        cls: Type[TTables],
        wc: WorkspaceClient,
        full_table_name: str,
        load_options: GenieLoadOptions,
    ) -> TTables:
        """Build a GenieTableConfig from a Unity Catalog table metadata."""
        try:
            table_info = wc.tables.get(full_name=full_table_name)
        except Exception as e:
            raise ValueError(
                f"Failed to load Unity Catalog table '{full_table_name}': {e}"
            )

        desc_lines: Optional[List[str | None]] = None
        if getattr(table_info, "comment", None):
            desc_lines = [table_info.comment]

        column_cfgs: Optional[List[GenieColumnConfig]] = None
        if load_options.include_column_configs and hasattr(table_info, "columns"):
            column_cfgs = [
                GenieColumnConfig(
                    column_name=col.name,
                    get_example_values=load_options.include_example_values,
                    build_value_dictionary=load_options.include_value_dictionary,
                )
                for col in table_info.columns
                if hasattr(col, "name") and not str(col.name).startswith("__")
            ]
            column_cfgs = sorted(column_cfgs, key=lambda x: x.column_name)

        return cls(
            identifier=full_table_name,
            description=desc_lines,
            column_configs=column_cfgs,
        )


class GenieDataSources(BaseModel):
    tables: list[GenieTableConfig] | None = None

    @classmethod
    def from_dict(cls: Type[TSources], data: Dict[str, Any]) -> TSources:
        """Factory method to create data sources from a dictionary."""
        return cls(**data)

    @classmethod
    def from_list(cls: Type[TSources], table_identifiers: List[str]) -> TSources:
        """Factory method to create data sources from a list of table names."""
        return cls(
            tables=[GenieTableConfig(identifier=name) for name in table_identifiers]
        )

    @classmethod
    def from_uc(
        cls, wc: WorkspaceClient, table_names: list[str], opts: GenieLoadOptions
    ):
        return cls(
            tables=[
                GenieTableConfig.from_unity_catalog(wc, name, opts)
                for name in table_names
            ]
        )


class GenieExampleSQL(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: list[str]
    sql: list[str]


class GenieInstruction(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: list[str] | None = None


class GenieTableJoinSpec(BaseModel):
    identifier: str = Field(default_factory=lambda: uuid.uuid4().hex)
    alias: str


class GenieJoinSpecs(BaseModel):
    id: str | None = Field(default_factory=lambda: uuid.uuid4().hex)
    left: GenieTableJoinSpec
    right: GenieTableJoinSpec
    sql: list[str]


class GenieSQLSnippet(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    alias: str | None
    sql: list[str]
    display_name: str
    synonyms: list[str]


class GenieSQLSnippets(BaseModel):
    filters: list[GenieSQLSnippet] | None = None
    expressions: list[GenieSQLSnippet] | None = None
    measures: list[GenieSQLSnippet] | None = None


class GenieInstructions(BaseModel):
    text_instructions: list[GenieInstruction] | None = None
    example_question_sqls: list[GenieExampleSQL] | None = None
    join_specs: list[GenieJoinSpecs] | None = None
    sql_snippets: GenieSQLSnippets | None = None

    @classmethod
    def from_dict(cls: Type[TInstructions], data: Dict[str, Any]) -> TInstructions:
        """Factory method to create instructions from a dictionary."""

        return cls(
            text_instructions=data.get("text_instructions"),
            example_question_sqls=data.get("example_question_sqls"),
            join_specs=data.get("join_specs"),
            sql_snippets=data.get("sql_snippets"),
        )


class GenieSchemaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")
    version: int
    config: Optional[GenieConfig] = None
    data_sources: Optional[GenieDataSources] = None
    instructions: Optional[GenieInstructions] = None

    def to_yaml(self, file_path: str = "genie.yml") -> None:
        """
        Serializes the model to the specific YAML format and writes it
        to a file (default: genie.yml), preserving the template header.
        """
        header = GENIE_CONFIG_TEMPLATE.split("genie:")[0].strip()
        data = self.model_dump(exclude_none=True)

        genie_payload = {
            "genie": {
                "data_sources": data.get("data_sources") or {},
                "config": data.get("config") or [],
                "instructions": data.get("instructions")
                or {"text_instructions": {"content": []}},
                "example_question_sqls": [],
                "join_specs": [],
                "sql_snippets": {},
            }
        }

        yaml_content = yaml.dump(
            genie_payload, sort_keys=False, default_flow_style=False
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header + "\n" + yaml_content)
