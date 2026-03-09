from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from genie_toolkit.schemas import (
    GenieBenchmarks,
    GenieConfig,
    GenieDataSources,
    GenieInstructions,
    GenieLoadOptions,
    GenieSchemaSettings,
    GenieTableConfig,
)

TABLE_NAME = "catalog.schema.table"


def _mock_wc(columns=None, comment=None, raises=None):
    """Build a mock WorkspaceClient with configurable table metadata."""
    wc = MagicMock()
    if raises:
        wc.tables.get.side_effect = raises
    else:
        cols = columns or []
        table_info = SimpleNamespace(
            comment=comment,
            columns=[SimpleNamespace(name=c) for c in cols],
        )
        wc.tables.get.return_value = table_info
    return wc


def test_table_config_from_unity_catalog():
    wc = _mock_wc(columns=["beta", "alpha", "gamma"], comment="A test table")
    opts = GenieLoadOptions(
        include_column_configs=True,
        include_example_values=False,
        include_value_dictionary=False,
    )

    table_config = GenieTableConfig.from_unity_catalog(
        wc=wc, full_table_name=TABLE_NAME, load_options=opts
    )

    assert table_config.identifier == TABLE_NAME
    assert table_config.description == ["A test table"]
    assert table_config.column_configs is not None
    assert len(table_config.column_configs) == 3
    assert table_config.column_configs[0].column_name == "alpha"
    assert table_config.column_configs[1].column_name == "beta"


def test_table_config_from_unity_catalog_no_comment():
    wc = _mock_wc(columns=["col_a"], comment=None)
    opts = GenieLoadOptions(include_column_configs=True)

    table_config = GenieTableConfig.from_unity_catalog(
        wc=wc, full_table_name=TABLE_NAME, load_options=opts
    )

    assert table_config.description is None


def test_table_config_from_unity_catalog_no_columns():
    wc = _mock_wc(columns=["col_a"])
    opts = GenieLoadOptions(include_column_configs=False)

    table_config = GenieTableConfig.from_unity_catalog(
        wc=wc, full_table_name=TABLE_NAME, load_options=opts
    )

    assert table_config.column_configs is None


def test_table_config_from_unity_catalog_filters_internal_columns():
    wc = _mock_wc(columns=["visible", "__internal"])
    opts = GenieLoadOptions(include_column_configs=True)

    table_config = GenieTableConfig.from_unity_catalog(
        wc=wc, full_table_name=TABLE_NAME, load_options=opts
    )

    assert len(table_config.column_configs) == 1
    assert table_config.column_configs[0].column_name == "visible"


def test_from_unity_catalog_invalid_table():
    wc = _mock_wc(raises=Exception("Not found"))
    opts = GenieLoadOptions()

    with pytest.raises(ValueError) as exc:
        GenieTableConfig.from_unity_catalog(
            wc=wc, full_table_name="bad.table.name", load_options=opts
        )
    assert "Failed to load Unity Catalog table" in str(exc.value)


def test_table_config_from_dict():
    data = {"identifier": "my_table", "description": ["desc"]}
    table = GenieTableConfig.from_dict(data)
    assert table.identifier == "my_table"


def test_data_sources_from_uc():
    wc = _mock_wc(columns=["col_a"])
    opts = GenieLoadOptions(include_column_configs=False)

    ds = GenieDataSources.from_uc(wc=wc, table_names=[TABLE_NAME], opts=opts)

    assert len(ds.tables) == 1
    assert ds.tables[0].identifier == TABLE_NAME
    assert ds.tables[0].column_configs is None


def test_data_sources_from_dict():
    ds = GenieDataSources.from_dict(
        {"tables": [{"identifier": "t1"}, {"identifier": "t2"}]}
    )
    assert len(ds.tables) == 2


def test_genie_config_from_dict():
    data = [{"question": ["What is the revenue?"]}]
    config = GenieConfig.from_dict(data)

    assert len(config.sample_questions) == 1
    assert config.sample_questions[0].question == ["What is the revenue?"]
    assert config.sample_questions[0].id is not None


def test_instructions_from_dict():
    data = {
        "text_instructions": [{"content": ["Do this"]}],
        "example_question_sqls": [{"question": ["q"], "sql": ["SELECT 1"]}],
    }
    instructions = GenieInstructions.from_dict(data)
    assert instructions.text_instructions is not None
    assert instructions.example_question_sqls is not None


def test_benchmarks_from_dict():
    data = {
        "questions": [
            {
                "question": ["What is average order value?"],
                "answer": [{"format": "SQL", "content": ["SELECT AVG(amount)"]}],
            }
        ]
    }
    benchmarks = GenieBenchmarks.from_dict(data)
    assert len(benchmarks.questions) == 1
    assert benchmarks.questions[0].answer[0].format == "SQL"


def test_schema_serialization_to_yaml(tmp_path):
    settings = GenieSchemaSettings(
        version=1,
        config=GenieConfig(sample_questions=[]),
        data_sources=GenieDataSources.from_list(["table_a", "table_b"]),
    )

    output_file = tmp_path / "genie_test.yml"
    settings.to_yaml(file_path=str(output_file))

    assert output_file.exists()

    with open(output_file, "r") as f:
        content = yaml.safe_load(f)

    assert "genie" in content
    assert content["genie"]["data_sources"]["tables"][0]["identifier"] == "table_a"
    assert "instructions" in content["genie"]


def test_schema_settings_benchmarks_optional():
    settings = GenieSchemaSettings(version=1)
    assert settings.benchmarks is None


def test_load_options_dataclass():
    opts = GenieLoadOptions()
    assert opts.include_column_configs is True
    assert opts.include_example_values is True
