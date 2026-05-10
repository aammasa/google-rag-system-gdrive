"""
Ingestion service — orchestrates the full Drive → Chunk → Embed → Store pipeline.
"""

import asyncio
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings.vertex_embeddings import VertexEmbeddings
from app.ingestion.drive_client import DriveClient
from app.logger import get_logger
from app.utils.helpers import stable_id
from app.utils.keywords import enrich_chunk_for_embedding, extract_keywords
from app.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)
settings = get_settings()

# Vertex AI embedding endpoint accepts at most 250 texts per batch.
_EMBED_BATCH_SIZE = 50


class IngestionService:
    def __init__(
        self,
        drive_client: DriveClient,
        embeddings: VertexEmbeddings,
        vectorstore: ChromaStore,
    ) -> None:
        self.drive = drive_client
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            length_function=len,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        folder_id: str | None = None,
        file_ids: list[str] | None = None,
    ) -> dict:
        """
        Full ingestion pipeline:
          1. Resolve files to process (by folder or explicit IDs)
          2. Download & extract text per file
          3. Split into chunks
          4. Batch-embed via Vertex AI (runs in thread to avoid blocking the loop)
          5. Upsert into ChromaDB
        """
        logger.info("ingestion_started", folder_id=folder_id, file_count=len(file_ids or []))

        # 1. Resolve file list
        files = self._resolve_files(folder_id, file_ids)
        if not files:
            logger.warning("ingestion_no_files_found")
            return {"status": "done", "files_processed": 0, "chunks_stored": 0}

        # 2 & 3. Download + chunk
        all_docs = await self._download_and_chunk(files)
        if not all_docs:
            logger.warning("ingestion_no_content_extracted", files=len(files))
            return {"status": "done", "files_processed": len(files), "chunks_stored": 0}

        # 4. Embed in batches (blocking I/O → thread pool)
        await self._embed_documents(all_docs)

        # 5. Upsert into ChromaDB
        self.vectorstore.add_documents(all_docs)

        logger.info(
            "ingestion_complete",
            files_processed=len(files),
            chunks_stored=len(all_docs),
        )
        return {
            "status": "done",
            "files_processed": len(files),
            "chunks_stored": len(all_docs),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_files(
        self,
        folder_id: str | None,
        file_ids: list[str] | None,
    ) -> list[dict]:
        if file_ids:
            files = []
            for fid in file_ids:
                try:
                    files.append(self.drive.get_file_metadata(fid))
                except Exception as exc:
                    logger.error("drive_metadata_failed", file_id=fid, error=str(exc))
            return files

        return self.drive.list_files(
            folder_id=folder_id or settings.gdrive_root_folder_id or None
        )

    async def _download_and_chunk(self, files: list[dict]) -> list[dict]:
        """Download each file, extract text, and split into chunks."""
        all_docs: list[dict] = []

        for file in files:
            file_id = file["id"]
            mime_type = file.get("mimeType", "")
            file_name = file.get("name", file_id)

            try:
                raw_bytes = await asyncio.to_thread(
                    self.drive.download_file, file_id, mime_type
                )
                text = self._extract_text(raw_bytes, mime_type)

                if not text.strip():
                    logger.warning("file_empty_content", file_id=file_id, name=file_name)
                    continue

                chunks = self.splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    all_docs.append(
                        {
                            "id": stable_id(file_id, str(i)),
                            "content": chunk,
                            "metadata": {
                                "file_id": file_id,
                                "file_name": file_name,
                                "mime_type": mime_type,
                                "web_view_link": file.get("webViewLink", ""),
                                "modified_time": file.get("modifiedTime", ""),
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                            },
                        }
                    )

                logger.info(
                    "file_chunked", file_id=file_id, name=file_name, chunks=len(chunks)
                )

            except Exception as exc:
                logger.error(
                    "file_ingestion_failed",
                    file_id=file_id,
                    name=file_name,
                    error=str(exc),
                )

        return all_docs

    async def _embed_documents(self, documents: list[dict]) -> None:
        """
        Populate the ``embedding`` key on each document dict in-place.

        Uses keyword-enriched text for embedding (better recall) while
        storing only the original chunk text in ChromaDB (cleaner retrieval).
        """
        for i in range(0, len(documents), _EMBED_BATCH_SIZE):
            batch = documents[i : i + _EMBED_BATCH_SIZE]

            # Build enriched texts for embedding only
            enriched_texts: list[str] = []
            for doc in batch:
                keywords = extract_keywords(doc["content"], top_n=10)
                doc["metadata"]["keywords"] = ", ".join(keywords)  # store for reference
                enriched_texts.append(enrich_chunk_for_embedding(doc["content"], keywords))

            embeddings: list[list[float]] = await asyncio.to_thread(
                self.embeddings.embed_documents, enriched_texts
            )

            for doc, emb in zip(batch, embeddings):
                doc["embedding"] = emb

            logger.debug("batch_embedded", batch_start=i, batch_size=len(batch))

    @staticmethod
    def _extract_text(raw_bytes: bytes, mime_type: str) -> str:
        """Convert raw file bytes to a plain-text string."""
        if mime_type == "application/pdf":
            return IngestionService._extract_pdf(raw_bytes)
        # Everything else (text/plain, text/csv, exported Google Docs, etc.)
        return raw_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_pdf(raw_bytes: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
