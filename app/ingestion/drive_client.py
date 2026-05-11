"""Google Drive API client — authentication and file listing/download."""

import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Google Workspace native types must be exported; everything else is downloaded directly.
_EXPORT_MIME_MAP: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Binary / text types we can ingest directly.
_DOWNLOADABLE_MIME_TYPES: set[str] = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_ALL_SUPPORTED = set(_EXPORT_MIME_MAP) | _DOWNLOADABLE_MIME_TYPES

_FILE_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, size"


class DriveClient:
    """
    Thin wrapper around the Google Drive API v3.

    Auth strategy (tried in order):
      1. Saved OAuth token file (local dev, user's own Drive)
      2. OAuth flow via browser (first-time local setup)

    For Cloud Run, mount a service account key and set
    GOOGLE_APPLICATION_CREDENTIALS; the ADC (Application Default Credentials)
    will be picked up automatically if token/credential files are absent.
    """

    def __init__(self) -> None:
        self._service: Any = None

    # ── Authentication ────────────────────────────────────────────────────────

    def _build_service(self) -> Any:
        creds: Credentials | None = None
        token_path = Path(settings.gdrive_token_file)
        scopes = settings.gdrive_scopes_list

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("drive_token_refresh")
                creds.refresh(Request())
            else:
                creds_path = self._resolve_credentials_file()
                logger.info("drive_oauth_flow_starting")
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
            logger.info("drive_token_saved", path=str(token_path))

        return build("drive", "v3", credentials=creds)

    def _resolve_credentials_file(self) -> Path:
        """
        Return a path to a valid credentials JSON file.

        Priority:
          1. GDRIVE_CREDENTIALS_JSON env var (base64-encoded) — written to a temp file
          2. GDRIVE_CREDENTIALS_FILE path on disk
        """
        if settings.gdrive_credentials_json:
            logger.info("drive_credentials_from_env")
            raw = base64.b64decode(settings.gdrive_credentials_json)
            # Validate it's proper JSON before writing
            json.loads(raw)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="wb"
            )
            tmp.write(raw)
            tmp.flush()
            return Path(tmp.name)

        creds_path = Path(settings.gdrive_credentials_file)
        if not creds_path.exists():
            raise FileNotFoundError(
                f"OAuth credentials not found at {creds_path}. "
                "Set GDRIVE_CREDENTIALS_JSON env var or place the file at the path above."
            )
        logger.info("drive_credentials_from_file", path=str(creds_path))
        return creds_path

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = self._build_service()
        return self._service

    # ── Public API ────────────────────────────────────────────────────────────

    def list_files(
        self,
        folder_id: str | None = None,
        page_size: int = 100,
    ) -> list[dict]:
        """
        Return a flat list of file metadata for all supported file types.
        Paginates automatically until the full result set is fetched.
        """
        mime_filter = " or ".join(f"mimeType='{mt}'" for mt in sorted(_ALL_SUPPORTED))
        query = f"({mime_filter}) and trashed=false"
        if folder_id:
            query = f"'{folder_id}' in parents and {query}"

        files: list[dict] = []
        page_token: str | None = None

        while True:
            params: dict = {
                "q": query,
                "pageSize": page_size,
                "fields": f"nextPageToken, files({_FILE_FIELDS})",
            }
            if page_token:
                params["pageToken"] = page_token

            response = self.service.files().list(**params).execute()
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")

            if not page_token:
                break

        logger.info("drive_files_listed", count=len(files), folder_id=folder_id)
        return files

    def download_file(self, file_id: str, mime_type: str) -> bytes:
        """
        Download a file's content as bytes.
        Google Workspace docs are exported to their plain-text equivalent;
        all other supported types are downloaded directly.
        """
        if mime_type in _EXPORT_MIME_MAP:
            export_mime = _EXPORT_MIME_MAP[mime_type]
            request = self.service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
            logger.debug("drive_export", file_id=file_id, export_mime=export_mime)
        else:
            request = self.service.files().get_media(fileId=file_id)
            logger.debug("drive_download", file_id=file_id, mime_type=mime_type)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return buffer.getvalue()

    def get_file_metadata(self, file_id: str) -> dict:
        """Fetch metadata for a single file by ID."""
        return (
            self.service.files()
            .get(fileId=file_id, fields=_FILE_FIELDS)
            .execute()
        )
