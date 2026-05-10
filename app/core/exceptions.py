from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base application error with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str) -> None:
        super().__init__(f"{resource} not found.", status_code=404)


class ValidationError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=422)


class ExternalServiceError(AppError):
    """Raised when an upstream service (Drive, Vertex, ChromaDB) fails."""

    def __init__(self, service: str, detail: str = "") -> None:
        super().__init__(f"{service} error: {detail}", status_code=502)


# ── FastAPI exception handlers ────────────────────────────────────────────────

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred."},
    )
