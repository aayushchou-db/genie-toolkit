from genie_config_manager.schemas import GenieSchemaSettings
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieSpace

def create_genie_space(
    wc: WorkspaceClient,
    warehouse_id: str,
    genie_schema_settings: GenieSchemaSettings,
    title: str | None = None,
    parent_path: str | None = None,
) -> GenieSpace:
    serialized_space = genie_schema_settings.model_dump_json(exclude_none=True)

    genie_space = wc.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=serialized_space,
        title=title,
        parent_path=parent_path,
    )
    return genie_space


def update_genie_space(
    wc: WorkspaceClient,
    space_id: str,
    genie_schema_settings: GenieSchemaSettings,
    title: str | None = None,
    description: str | None = None,
) -> GenieSpace:
    serialized_space = genie_schema_settings.model_dump_json()

    genie_space = wc.genie.update_space(
        space_id=space_id,
        serialized_space=serialized_space,
        title=title,
        description=description,
    )
    return genie_space
