from unittest.mock import MagicMock, patch

import pytest
import yaml

from genie_toolkit.schemas import (
    GenieConfig,
    GenieDataSources,
    GenieLoadOptions,
    GenieSampleQuestion,
    GenieSchemaSettings,
    GenieTableConfig,
)


class TestGenieConfigModels:
    def test_genie_sample_question_id_generation(self):
        q = GenieSampleQuestion(question=["How many users?"])
        assert q.id is not None
        assert len(q.id) == 32

    def test_genie_config_from_dict(self):
        data = [
            {"question": ["What is revenue?"]},
            {"id": "123", "question": ["Growth?"]},
        ]
        config = GenieConfig.from_dict(data)
        assert len(config.sample_questions) == 2
        assert config.sample_questions[1].id == "123"


class TestUnityCatalogIntegration:
    @pytest.fixture
    def mock_wc(self):
        """Fixture to provide a mocked Databricks WorkspaceClient."""
        return MagicMock()

    def test_table_config_from_unity_catalog_success(self, mock_wc):
        """Test successful metadata extraction from UC."""
        mock_table = MagicMock()
        mock_table.comment = "Test table comment"

        col1 = MagicMock()
        col1.name = "user_id"
        col2 = MagicMock()
        col2.name = "__hidden_col"

        mock_table.columns = [col1, col2]
        mock_wc.tables.get.return_value = mock_table

        opts = GenieLoadOptions(include_column_configs=True)
        config = GenieTableConfig.from_unity_catalog(
            mock_wc, "catalog.schema.table", opts
        )

        assert config.identifier == "catalog.schema.table"
        assert config.description == ["Test table comment"]

        assert config.column_configs
        assert len(config.column_configs) == 1
        assert config.column_configs[0].column_name == "user_id"

    def test_table_config_from_unity_catalog_error(self, mock_wc):
        """Test behavior when UC table is not found."""
        mock_wc.tables.get.side_effect = Exception("Table not found")

        with pytest.raises(ValueError, match="Failed to load Unity Catalog table"):
            GenieTableConfig.from_unity_catalog(
                mock_wc, "bad.table", GenieLoadOptions()
            )


class TestGenieDataSources:
    def test_data_sources_from_list(self):
        """Test creating sources from a simple list of strings."""
        tables = ["main.default.users", "main.default.orders"]
        sources = GenieDataSources.from_list(tables)

        assert sources.tables
        assert len(sources.tables) == 2
        assert sources.tables[0].identifier == "main.default.users"


class TestSerialization:
    def test_to_yaml_output(self, tmp_path):
        """Test that YAML is written correctly with headers."""

        questions = GenieConfig(
            sample_questions=[GenieSampleQuestion(question=["Test?"])]
        )
        settings = GenieSchemaSettings(version=1, config=questions)

        file_path = tmp_path / "genie.yml"

        with patch(
            "genie_toolkit.templates.GENIE_CONFIG_TEMPLATE",
            "HEADER_TEXT\ngenie: dummy",
        ):
            settings.to_yaml(str(file_path))

        assert file_path.exists()
        content = file_path.read_text()

        parsed = yaml.safe_load(content)
        assert "genie" in parsed
        assert parsed["genie"]["config"]["sample_questions"][0]["question"] == ["Test?"]
