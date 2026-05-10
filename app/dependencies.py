"""
FastAPI dependency injection providers.

All clients and services are constructed lazily and cached with ``lru_cache``
so a single instance is shared across requests within a process.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.chat.agent import RagAgent
from app.config import Settings, get_settings
from app.core.session_store import SessionStore
from app.embeddings.vertex_embeddings import VertexEmbeddings
from app.ingestion.drive_client import DriveClient
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import Retriever
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService
from app.vectorstore.chroma_store import ChromaStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Infrastructure clients (singletons) ───────────────────────────────────────

@lru_cache
def get_drive_client() -> DriveClient:
    return DriveClient()


@lru_cache
def get_vectorstore_client() -> ChromaStore:
    return ChromaStore()


@lru_cache
def get_embedding_client() -> VertexEmbeddings:
    return VertexEmbeddings()


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(ttl_seconds=3600, max_turns=20)


# ── Domain objects (singletons) ───────────────────────────────────────────────

@lru_cache
def get_retriever() -> Retriever:
    return Retriever(
        embeddings=get_embedding_client(),
        vectorstore=get_vectorstore_client(),
    )


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        embeddings=get_embedding_client(),
        vectorstore=get_vectorstore_client(),
    )


@lru_cache
def get_rag_agent() -> RagAgent:
    return RagAgent(
        retriever=get_hybrid_retriever(),
        session_store=get_session_store(),
    )


# ── Service layer (singletons) ────────────────────────────────────────────────

@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        drive_client=get_drive_client(),
        embeddings=get_embedding_client(),
        vectorstore=get_vectorstore_client(),
    )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(retriever=get_hybrid_retriever())


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        agent=get_rag_agent(),
        session_store=get_session_store(),
    )


# ── FastAPI-injectable type aliases ───────────────────────────────────────────

IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RetrievalServiceDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
