import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.chat.webhook_verifier import WebhookVerificationError, verify_chat_request
from app.dependencies import ChatServiceDep
from app.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── REST API models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(
        None,
        description="Omit to start a new session. Reuse the returned session_id to continue.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.post("/message", response_model=ChatResponse, summary="Send a chat message")
async def chat_message(
    payload: ChatRequest,
    service: ChatServiceDep,
) -> ChatResponse:
    """
    Send a message and receive a grounded answer from Gemini.

    - First call: omit ``session_id`` — a new one is created and returned.
    - Follow-up calls: pass the ``session_id`` to continue the conversation.
      History is managed server-side automatically.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    agent_response = await service.respond(message=payload.message, session_id=session_id)
    return ChatResponse(
        answer=agent_response.answer,
        sources=agent_response.sources,
        session_id=session_id,
    )


@router.get("/history/{session_id}", response_model=HistoryResponse, summary="Get session history")
async def get_history(session_id: str, service: ChatServiceDep) -> HistoryResponse:
    """Return the full conversation history for a session."""
    return HistoryResponse(
        session_id=session_id,
        turns=service.get_history(session_id),
    )


@router.delete("/history/{session_id}", summary="Clear session history")
async def clear_history(session_id: str, service: ChatServiceDep) -> dict:
    """Delete all stored turns for a session."""
    service.clear_session(session_id)
    return {"session_id": session_id, "status": "cleared"}


# ── Google Chat webhook ───────────────────────────────────────────────────────

@router.post("/webhook", summary="Google Chat webhook receiver")
async def google_chat_webhook(
    request: Request,
    service: ChatServiceDep,
    authorization: str | None = Header(default=None),
) -> dict:
    """
    Receives events from Google Chat API.

    Authentication:
    - Verifies the Bearer JWT sent by Google (skipped in dev when
      GOOGLE_CHAT_SERVICE_ACCOUNT is not configured).

    Supported event types:
    - MESSAGE          → RAG answer as a cardsV2 response
    - ADDED_TO_SPACE   → Welcome message
    - REMOVED_FROM_SPACE → Session cleared
    - CARD_CLICKED     → (future: interactive card actions)

    Slash commands (register in GCP Console):
    - /help    (ID 1) — Show available commands
    - /clear   (ID 2) — Clear conversation history
    - /sources (ID 3) — Show last answer sources
    - /ingest  (ID 4) — Trigger Drive re-ingestion
    """
    # 1. Verify request origin
    try:
        verify_chat_request(authorization or "")
    except WebhookVerificationError as exc:
        logger.warning("webhook_verification_failed", error=str(exc))
        raise HTTPException(status_code=401, detail=str(exc))

    # 2. Parse body
    event: dict = await request.json()
    event_type = event.get("type", "UNKNOWN")
    space = event.get("space", {}).get("name", "unknown")
    logger.info("chat_event_received", event_type=event_type, space=space)

    # 3. Dispatch and return card payload
    return await service.handle_webhook(event)
