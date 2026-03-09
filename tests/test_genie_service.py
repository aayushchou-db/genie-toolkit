from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.dashboards import GenieSpace

from genie_toolkit.genie_service import GenieService, get_genie_service
from genie_toolkit.schemas import GenieSchemaSettings


@pytest.fixture
def mock_wc():
    return MagicMock()


@pytest.fixture
def genie_service(mock_wc):
    return GenieService(mock_wc)


@pytest.fixture
def settings():
    return GenieSchemaSettings(version=1)


def test_create_space(genie_service, settings, mock_wc):
    mock_wc.genie.create_space.return_value = GenieSpace(
        space_id="sp123", title="Test Space"
    )

    space = genie_service.create(
        warehouse_id="wh123",
        genie_schema_settings=settings,
        title="Test Space",
        parent_path="/Shared",
    )

    assert space.space_id == "sp123"
    assert space.title == "Test Space"
    mock_wc.genie.create_space.assert_called_once()


def test_update_space(genie_service, settings, mock_wc):
    mock_wc.genie.update_space.return_value = GenieSpace(
        space_id="sp123", title="Updated", description="New desc"
    )

    space = genie_service.update(
        space_id="sp123",
        genie_schema_settings=settings,
        description="New desc",
    )

    assert space.space_id == "sp123"
    assert space.description == "New desc"
    mock_wc.genie.update_space.assert_called_once()


def test_get_genie_service_success():
    with patch("genie_toolkit.genie_service.WorkspaceClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        service = get_genie_service(profile=None)
        assert isinstance(service, GenieService)


def test_get_genie_service_failure():
    with patch(
        "genie_toolkit.genie_service.WorkspaceClient",
        side_effect=Exception("Config Error"),
    ):
        with pytest.raises(ValueError) as exc_info:
            get_genie_service(profile="invalid_profile")

        assert "Failed to connect to Databricks" in str(exc_info.value)
