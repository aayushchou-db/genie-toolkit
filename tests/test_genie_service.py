from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieSpace

from genie_toolkit.genie_service import (
    GenieService,
    get_genie_service,
)
from genie_toolkit.schemas import GenieSchemaSettings


@pytest.fixture
def mock_workspace_client():
    """Provides a mocked WorkspaceClient."""
    return MagicMock(spec=WorkspaceClient)


@pytest.fixture
def genie_service(mock_workspace_client):
    """Provides an instance of GenieService with a mocked client."""
    return GenieService(mock_workspace_client)


@pytest.fixture
def mock_settings():
    """Provides a mocked GenieSchemaSettings model."""
    settings = MagicMock(spec=GenieSchemaSettings)
    settings.model_dump_json.return_value = '{"fake": "json"}'
    return settings


### Test GenieService.create
def test_create_space_calls_correct_sdk_method(
    genie_service, mock_workspace_client, mock_settings
):
    # Arrange
    mock_workspace_client.genie.create_space.return_value = MagicMock(spec=GenieSpace)

    # Act
    genie_service.create(
        warehouse_id="123",
        genie_schema_settings=mock_settings,
        title="Test Title",
        parent_path="/Shared",
    )

    # Assert
    mock_settings.model_dump_json.assert_called_once_with(exclude_none=True)
    mock_workspace_client.genie.create_space.assert_called_once_with(
        warehouse_id="123",
        serialized_space='{"fake": "json"}',
        title="Test Title",
        parent_path="/Shared",
    )


### Test GenieService.update
def test_update_space_calls_correct_sdk_method(
    genie_service, mock_workspace_client, mock_settings
):
    # Arrange
    mock_workspace_client.genie.update_space.return_value = MagicMock(spec=GenieSpace)

    # Act
    genie_service.update(
        space_id="abc-789",
        genie_schema_settings=mock_settings,
        description="New Description",
    )

    # Assert
    mock_settings.model_dump_json.assert_called_once()
    mock_workspace_client.genie.update_space.assert_called_once_with(
        space_id="abc-789",
        serialized_space='{"fake": "json"}',
        title=None,
        description="New Description",
    )


def test_get_genie_service_success():
    with patch("genie_toolkit.genie_service.WorkspaceClient") as mock_wc_init:
        service = get_genie_service(profile="dev")

        mock_wc_init.assert_called_once_with(profile="dev")
        assert isinstance(service, GenieService)


def test_get_genie_service_failure():
    with patch(
        "genie_toolkit.genie_service.WorkspaceClient",
        side_effect=Exception("Connection Error"),
    ):
        with pytest.raises(ValueError):
            get_genie_service(profile="invalid")
