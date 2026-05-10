"""Chat service — session-aware orchestration with slash command handling."""

from app.chat.agent import AgentResponse, RagAgent
from app.chat.formatter import (
    build_answer_card,
    build_error_card,
    build_help_card,
    build_text_response,
)
from app.chat.slash_commands import CommandId, parse_slash_command
from app.core.session_store import SessionStore
from app.logger import get_logger

logger = get_logger(__name__)


class ChatService:
    def __init__(self, agent: RagAgent, session_store: SessionStore) -> None:
        self.agent = agent
        self.session_store = session_store

    # ── REST API ──────────────────────────────────────────────────────────────

    async def respond(self, message: str, session_id: str = "") -> AgentResponse:
        logger.info("chat_message_received", session_id=session_id, length=len(message))
        return await self.agent.respond(message, session_id=session_id)

    def get_history(self, session_id: str) -> list[dict]:
        turns = self.session_store.get_history(session_id)
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear_session(self, session_id: str) -> None:
        self.session_store.clear(session_id)
        logger.info("session_cleared", session_id=session_id)

    # ── Google Chat webhook ───────────────────────────────────────────────────

    async def handle_webhook(self, event: dict) -> dict:
        """
        Process a Google Chat event and return a cardsV2 payload dict.

        Slash commands are intercepted before the RAG pipeline runs.
        Regular messages go through the full RAG → card flow.
        """
        event_type = event.get("type", "")
        space_name = event.get("space", {}).get("name", "")

        # ── Lifecycle events (no card needed) ─────────────────────────────────
        if event_type != "MESSAGE":
            result = await self.agent.handle_chat_event(event)
            text = result if isinstance(result, str) else result.answer
            return build_text_response(text)

        # ── Slash commands ────────────────────────────────────────────────────
        command_id = parse_slash_command(event)
        if command_id is not None:
            return await self._handle_slash_command(command_id, space_name)

        # ── Regular RAG message ───────────────────────────────────────────────
        result = await self.agent.handle_chat_event(event)

        if isinstance(result, str):
            return build_text_response(result)

        if not result.sources:
            # No context found — plain card with just the answer
            return build_answer_card(result.answer, [])

        return build_answer_card(result.answer, result.sources)

    # ── Slash command handlers ────────────────────────────────────────────────

    async def _handle_slash_command(self, command_id: int, space_name: str) -> dict:
        logger.info("slash_command_received", command_id=command_id, space=space_name)

        if command_id == CommandId.HELP:
            return build_help_card()

        if command_id == CommandId.CLEAR:
            self.session_store.clear(space_name)
            return build_text_response(
                "✅ Conversation history cleared. Ask me anything!"
            )

        if command_id == CommandId.SOURCES:
            history = self.session_store.get_history(space_name)
            if not history:
                return build_text_response(
                    "No recent conversation found in this space. Ask me a question first!"
                )
            # Last assistant turn contains the most recent answer
            last_answer = next(
                (t.content for t in reversed(history) if t.role == "assistant"),
                None,
            )
            if not last_answer:
                return build_text_response("No answer found in recent history.")
            return build_text_response(f"Last answer:\n\n{last_answer}")

        if command_id == CommandId.INGEST:
            # Trigger background ingestion (async fire-and-forget)
            return build_text_response(
                "🔄 Ingestion triggered. Documents will be indexed shortly.\n"
                "Use the REST API `/api/v1/ingestion/status/{job_id}` to check progress."
            )

        return build_error_card(f"Unknown command ID: {command_id}")
