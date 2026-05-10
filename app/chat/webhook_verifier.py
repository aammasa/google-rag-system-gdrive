"""
Google Chat webhook request verifier.

Google Chat signs every incoming request with a Bearer JWT token.
The token is a Google-issued OIDC ID token where:
  - iss  == "chat@system.gserviceaccount.com"
  - aud  == your Chat app's service account email

In development (GOOGLE_CHAT_SERVICE_ACCOUNT is empty), verification is
skipped with a warning so you can test locally via curl / Postman.
"""

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_GOOGLE_CHAT_ISSUER = "chat@system.gserviceaccount.com"


class WebhookVerificationError(Exception):
    """Raised when a Chat request cannot be authenticated."""


def verify_chat_request(authorization_header: str) -> None:
    """
    Verify the Bearer JWT sent by Google Chat.

    Raises WebhookVerificationError if the token is missing, invalid,
    has the wrong issuer, or does not match the configured service account.

    Skips verification entirely (with a warning) when
    GOOGLE_CHAT_SERVICE_ACCOUNT is not configured — useful for local dev.
    """
    service_account = settings.google_chat_service_account

    if not service_account:
        logger.warning(
            "chat_webhook_verification_skipped",
            reason="GOOGLE_CHAT_SERVICE_ACCOUNT not set",
        )
        return

    if not authorization_header or not authorization_header.startswith("Bearer "):
        raise WebhookVerificationError("Missing or malformed Authorization header.")

    token = authorization_header.removeprefix("Bearer ").strip()

    try:
        id_info = id_token.verify_oauth2_token(
            token,
            GoogleRequest(),
            audience=service_account,
        )
    except Exception as exc:
        raise WebhookVerificationError(f"Token verification failed: {exc}") from exc

    issuer = id_info.get("iss", "")
    if issuer != _GOOGLE_CHAT_ISSUER:
        raise WebhookVerificationError(
            f"Unexpected token issuer: {issuer!r}. Expected {_GOOGLE_CHAT_ISSUER!r}."
        )

    logger.debug("chat_webhook_verified", email=id_info.get("email"))
