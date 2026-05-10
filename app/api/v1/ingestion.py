import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.dependencies import IngestionServiceDep
from app.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# In-memory job registry — sufficient for a single-process deployment.
# Replace with Redis or a DB-backed store for multi-instance Cloud Run.
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    folder_id: str | None = Field(None, description="Google Drive folder ID to ingest")
    file_ids: list[str] = Field(default_factory=list, description="Specific file IDs to ingest")


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    message: str


async def _run_ingestion(
    job_id: str,
    service,
    folder_id: str | None,
    file_ids: list[str],
) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        result = await service.run(folder_id=folder_id, file_ids=file_ids or None)
        _jobs[job_id].update({"status": "done", **result})
        logger.info("ingestion_job_done", job_id=job_id, **result)
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        logger.error("ingestion_job_failed", job_id=job_id, error=str(exc))


@router.post("/sync", response_model=IngestResponse, summary="Trigger Drive ingestion")
async def trigger_ingestion(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    service: IngestionServiceDep,
) -> IngestResponse:
    """
    Queues a background job that fetches documents from Google Drive,
    chunks them, embeds them via Vertex AI, and upserts into ChromaDB.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued"}

    background_tasks.add_task(
        _run_ingestion,
        job_id,
        service,
        payload.folder_id,
        payload.file_ids,
    )

    logger.info("ingestion_job_queued", job_id=job_id, folder_id=payload.folder_id)
    return IngestResponse(
        job_id=job_id,
        status="queued",
        message="Ingestion job queued. Poll /status/{job_id} for progress.",
    )


@router.get("/status/{job_id}", summary="Get ingestion job status")
async def ingestion_status(job_id: str) -> dict:
    """Returns the current status of an ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, **job}
