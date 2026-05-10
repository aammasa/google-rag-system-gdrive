from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import (
    AppError,
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from app.logger import configure_logging, get_logger
from app.middleware import RequestLoggingMiddleware

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info(
        "starting_up",
        app=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )

    # Eagerly initialise ChromaDB so the first request isn't slow.
    try:
        from app.dependencies import get_vectorstore_client
        store = get_vectorstore_client()
        count = store.count()
        logger.info("chroma_ready", vectors=count)
    except Exception as exc:
        logger.warning("chroma_init_failed", error=str(exc))

    # Verify Vertex AI project config is present (does not make an API call).
    if not settings.google_cloud_project:
        logger.warning(
            "vertex_ai_project_missing",
            hint="Set GOOGLE_CLOUD_PROJECT in .env before ingesting or querying.",
        )

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("shutting_down", app=settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost applied last) ───────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ────────────────────────────────────────────────────
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


app = create_app()
