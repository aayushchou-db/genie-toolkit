from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from genie_toolkit.cli import app, update_env
from genie_toolkit.schemas import GenieDataSources

runner = CliRunner()


class TestUpdateEnv:
    def test_creates_env_file_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_path = tmp_path / ".env"
        assert not env_path.exists()

        with patch("genie_toolkit.cli.set_key") as mock_set:
            update_env("FOO", "bar")
            mock_set.assert_called_once_with(".env", "FOO", "bar")

    def test_uses_existing_env_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("EXISTING=val\n")

        with patch("genie_toolkit.cli.set_key") as mock_set:
            update_env("FOO", "bar")
            mock_set.assert_called_once_with(".env", "FOO", "bar")


class TestInitCommand:
    def test_init_creates_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["init", "--warehouse-id", "wh123", "--profile", "dev"],
        )
        assert result.exit_code == 0
        assert "initialized" in result.stdout.lower()
        assert (tmp_path / "genie.yml").exists()

    def test_init_prompts_when_no_args(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["init"],
            input="wh123\ndev\n",
        )
        assert result.exit_code == 0
        assert (tmp_path / "genie.yml").exists()

    def test_init_aborts_on_existing_no_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "genie.yml").write_text("existing")
        result = runner.invoke(
            app,
            ["init", "--warehouse-id", "wh123", "--profile", "dev"],
            input="n\n",
        )
        assert result.exit_code != 0 or "Aborted" in result.stdout

    def test_init_overwrites_on_confirm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "genie.yml").write_text("old")
        result = runner.invoke(
            app,
            ["init", "--warehouse-id", "wh123", "--profile", "dev"],
            input="y\n",
        )
        assert result.exit_code == 0
        assert (tmp_path / "genie.yml").read_text() != "old"

    def test_init_custom_config_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["init", "--warehouse-id", "wh123", "--profile", "dev", "--config", "custom.yml"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "custom.yml").exists()


class TestCreateCommand:
    @patch("genie_toolkit.cli.get_genie_service")
    @patch("genie_toolkit.cli.update_env")
    def test_create_space(self, mock_update_env, mock_get_svc, tmp_path):
        mock_space = MagicMock()
        mock_space.space_id = "sp123"
        mock_space.title = "Test Space"
        mock_get_svc.return_value.wc = MagicMock()
        mock_get_svc.return_value.create.return_value = mock_space

        config_file = tmp_path / "genie.yml"
        config_file.write_text(
            "genie:\n  data_sources:\n    tables: []\n  sample_questions: []\n  instructions: {}\n"
        )

        result = runner.invoke(
            app,
            [
                "create",
                "--warehouse-id", "wh123",
                "--profile", "dev",
                "--title", "Test Space",
                "--config", str(config_file),
            ],
        )
        assert result.exit_code == 0
        assert "created successfully" in result.stdout

    @patch("genie_toolkit.cli.GenieDataSources.from_uc")
    @patch("genie_toolkit.cli.get_genie_service")
    @patch("genie_toolkit.cli.update_env")
    def test_create_with_tables(self, mock_update_env, mock_get_svc, mock_from_uc, tmp_path):
        mock_space = MagicMock()
        mock_space.space_id = "sp123"
        mock_space.title = "Test"
        mock_get_svc.return_value.create.return_value = mock_space
        mock_from_uc.return_value = GenieDataSources.from_list(["catalog.schema.tbl"])

        config_file = tmp_path / "genie.yml"
        config_file.write_text(
            "genie:\n  data_sources:\n    tables:\n      - catalog.schema.tbl\n  sample_questions: []\n  instructions: {}\n"
        )

        result = runner.invoke(
            app,
            [
                "create",
                "--warehouse-id", "wh123",
                "--profile", "dev",
                "--title", "Test",
                "--config", str(config_file),
            ],
        )
        assert result.exit_code == 0
        mock_from_uc.assert_called_once()


class TestPushCommand:
    @patch("genie_toolkit.cli.get_genie_service")
    def test_push(self, mock_get_svc, tmp_path):
        mock_space = MagicMock()
        mock_space.title = "Updated"
        mock_get_svc.return_value.update.return_value = mock_space

        config_file = tmp_path / "genie.yml"
        config_file.write_text(
            "genie:\n  data_sources:\n    tables: []\n  instructions: {}\n"
        )

        result = runner.invoke(
            app,
            [
                "push",
                "--space-id", "sp123",
                "--profile", "dev",
                "--config", str(config_file),
            ],
        )
        assert result.exit_code == 0
        assert "updated successfully" in result.stdout


class TestPullCommand:
    @patch("genie_toolkit.cli.update_env")
    @patch("genie_toolkit.cli.get_genie_service")
    def test_pull(self, mock_get_svc, mock_update_env, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_space = MagicMock()
        mock_space.serialized_space = '{"version": 1}'
        mock_get_svc.return_value.wc.genie.get_space.return_value = mock_space

        result = runner.invoke(
            app,
            ["pull", "--space-id", "sp123", "--profile", "dev"],
        )
        assert result.exit_code == 0
        assert "Successfully pulled" in result.stdout
