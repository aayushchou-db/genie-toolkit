import logging
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage, GenieSpace, MessageStatus

from genie_toolkit.schemas import GenieSchemaSettings


class GenieService:
    def __init__(self, wc: WorkspaceClient):
        self.wc = wc

    def ask_question(
        self, space_id: str, question: str, timeout: int = 300
    ) -> GenieMessage:
        """Send a question to a Genie Space and return the completed message.

        Uses start_conversation_and_wait for automatic polling, with a fallback
        manual poll loop if the message isn't completed yet.
        """
        from datetime import timedelta

        message = self.wc.genie.start_conversation_and_wait(
            space_id=space_id,
            content=question,
            timeout=timedelta(seconds=timeout),
        )

        # If the wait helper returns a non-completed status, poll manually
        deadline = time.time() + timeout
        while message.status not in (
            MessageStatus.COMPLETED,
            MessageStatus.FAILED,
            MessageStatus.CANCELLED,
        ):
            if time.time() > deadline:
                raise TimeoutError(
                    f"Genie did not respond within {timeout}s for: {question!r}"
                )
            time.sleep(2)
            message = self.wc.genie.get_message(
                space_id=space_id,
                conversation_id=message.conversation_id,
                message_id=message.message_id,
            )

        if message.status == MessageStatus.FAILED:
            error_msg = ""
            if message.error:
                error_msg = str(message.error)
            raise RuntimeError(f"Genie query failed for {question!r}: {error_msg}")

        return message

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


def get_genie_service(profile: str | None = None) -> GenieService:
    try:
        if profile:
            wc = WorkspaceClient(profile=profile)
            return GenieService(wc)
        return GenieService(WorkspaceClient())
    except Exception as e:
        logging.error(f"❌ Failed to connect to Databricks: {e}")
        raise ValueError(f"Failed to connect to Databricks: {e}")
