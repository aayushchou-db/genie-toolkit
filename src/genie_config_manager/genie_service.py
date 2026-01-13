import logging

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieSpace

from genie_config_manager.schemas import GenieSchemaSettings


class GenieService:
    def __init__(self, wc: WorkspaceClient):
        self.wc = wc

    def create(
        self,
        warehouse_id: str,
        genie_schema_settings: GenieSchemaSettings,
        title: str | None = None,
        parent_path: str | None = None,
    ) -> GenieSpace:
        serialized_space = genie_schema_settings.model_dump_json(exclude_none=True)

        genie_space = self.wc.genie.create_space(
            warehouse_id=warehouse_id,
            serialized_space=serialized_space,
            title=title,
            parent_path=parent_path,
        )
        return genie_space

    def update(
        self,
        space_id: str,
        genie_schema_settings: GenieSchemaSettings,
        title: str | None = None,
        description: str | None = None,
    ) -> GenieSpace:
        serialized_space = genie_schema_settings.model_dump_json()

        genie_space = self.wc.genie.update_space(
            space_id=space_id,
            serialized_space=serialized_space,
            title=title,
            description=description,
        )
        return genie_space


def get_genie_service(profile: str | None) -> GenieService:
    try:
        wc = WorkspaceClient(profile=profile)
        return GenieService(wc)
    except Exception as e:
        logging.error(f"❌ Failed to connect to Databricks: {e}")
        raise ValueError
