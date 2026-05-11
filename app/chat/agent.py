"""Google Chat / RAG agent — multi-turn, retrieval-augmented generation."""

from dataclasses import dataclass, field

import vertexai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from app.config import get_settings
from app.core.session_store import SessionStore, Turn
from app.logger import get_logger
from app.prompts.templates import (
    CONDENSE_QUESTION_PROMPT,
    RAG_CONTEXT_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    format_history_for_condensation,
    history_to_langchain_messages,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class AgentResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    session_id: str = ""


class RagAgent:
    """
    Multi-turn RAG agent:
      1. Load conversation history from SessionStore
      2. If history exists → condense follow-up into standalone question
      3. Retrieve relevant chunks using the standalone question
      4. Build prompt: system + prior turns + context + current question
      5. Generate grounded answer via Gemini
      6. Persist both turns back to SessionStore
    """

    def __init__(self, retriever: HybridRetriever, session_store: SessionStore) -> None:
        self.retriever = retriever
        self.session_store = session_store
        self._llm = None

    # ── LLM initialisation ────────────────────────────────────────────────────

    def _load_llm(self):
        if settings.use_google_ai_studio:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.vertex_llm_model,
                google_api_key=settings.google_api_key,
                temperature=settings.vertex_llm_temperature,
                max_tokens=settings.vertex_llm_max_tokens,
            )
            logger.info("llm_loaded", backend="google_ai_studio", model=settings.vertex_llm_model)
        else:
            vertexai.init(
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
            )
            llm = ChatVertexAI(
                model_name=settings.vertex_llm_model,
                project=settings.google_cloud_project,
                location=settings.google_cloud_location,
                temperature=settings.vertex_llm_temperature,
                max_tokens=settings.vertex_llm_max_tokens,
            )
            logger.info("llm_loaded", backend="vertex_ai", model=settings.vertex_llm_model)
        return llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._load_llm()
        return self._llm

    # ── Core response logic ───────────────────────────────────────────────────

    async def respond(
        self,
        message: str,
        session_id: str = "",
    ) -> AgentResponse:
        # 1. Load history
        history: list[Turn] = self.session_store.get_history(session_id) if session_id else []

        # 2. Condense follow-up into a standalone retrieval question
        retrieval_question = await self._condense(message, history)

        # 3. Retrieve relevant chunks
        chunks: list[RetrievedChunk] = await self.retriever.retrieve(retrieval_question)

        if not chunks:
            answer = (
                "I couldn't find relevant information in the knowledge base "
                "for your question. Please try rephrasing or check that the "
                "relevant documents have been ingested."
            )
            logger.info("rag_no_context_found", session_id=session_id)
        else:
            answer = await self._generate(message, history, chunks)

        # 4. Persist both turns
        if session_id:
            self.session_store.append(session_id, "user", message)
            self.session_store.append(session_id, "assistant", answer)

        sources = [
            {
                "document_id": c.document_id,
                "score": round(c.score, 4),
                "file_name": c.metadata.get("file_name", ""),
                "web_view_link": c.metadata.get("web_view_link", ""),
            }
            for c in chunks
        ]

        logger.info("rag_response_generated", session_id=session_id, sources=len(sources))
        return AgentResponse(answer=answer, sources=sources, session_id=session_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _condense(self, message: str, history: list[Turn]) -> str:
        """
        Rephrase a follow-up question into a standalone question using history.
        Skips the LLM call entirely when there is no history to save latency.
        """
        if not history:
            return message

        chat_history_text = format_history_for_condensation(history)
        condensation_prompt = CONDENSE_QUESTION_PROMPT.format(
            chat_history=chat_history_text,
            question=message,
        )

        response = await self.llm.ainvoke([HumanMessage(content=condensation_prompt)])
        condensed: str = response.content.strip()  # type: ignore[assignment]

        logger.debug(
            "question_condensed",
            original=message[:80],
            condensed=condensed[:80],
        )
        return condensed

    async def _generate(
        self,
        message: str,
        history: list[Turn],
        chunks: list[RetrievedChunk],
    ) -> str:
        """Build the full prompt and call the LLM."""
        context = "\n\n---\n\n".join(c.content for c in chunks)

        # Build message list:
        #   [SystemMessage] + [prior turns...] + [HumanMessage(context + question)]
        prior_messages = history_to_langchain_messages(history)

        final_human = HumanMessage(
            content=(
                RAG_CONTEXT_TEMPLATE.format(context=context)
                + "\n\n"
                + RAG_USER_TEMPLATE.format(question=message)
            )
        )

        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT), *prior_messages, final_human]
        try:
            response = await self.llm.ainvoke(messages)
            return response.content  # type: ignore[return-value]
        except Exception as exc:
            logger.error(
                "llm_generate_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            raise

    # ── Google Chat webhook dispatcher ────────────────────────────────────────

    async def handle_chat_event(self, event: dict) -> AgentResponse | str:
        """
        Dispatch a Google Chat event and return either:
        - AgentResponse  (for MESSAGE events → caller builds card)
        - str            (for lifecycle events → plain text is fine)
        """
        event_type = event.get("type")
        space_name = event.get("space", {}).get("name", "")

        if event_type == "MESSAGE":
            user_message = event.get("message", {}).get("text", "").strip()
            if not user_message:
                return "Please send a text message."
            # Each Chat space gets its own session history
            return await self.respond(user_message, session_id=space_name)

        if event_type == "ADDED_TO_SPACE":
            display_name = event.get("space", {}).get("displayName", "this space")
            return (
                f"Hello! I'm your Drive knowledge assistant in *{display_name}*. "
                "Ask me anything about your documents, or type /help for commands."
            )

        if event_type == "REMOVED_FROM_SPACE":
            if space_name:
                self.session_store.clear(space_name)
            return ""

        return ""
