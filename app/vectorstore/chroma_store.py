"""ChromaDB vector store — persistent (dev) or HTTP (Docker / Cloud Run)."""

import chromadb
from chromadb import Collection

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ChromaStore:
    """
    Manages a single ChromaDB collection used to store document embeddings.

    Mode is selected automatically from env vars:
      - CHROMA_HOST set  → HttpClient (remote, used in docker-compose / Cloud Run)
      - CHROMA_HOST unset → PersistentClient (local filesystem)

    All vectors use cosine distance so scores returned by ``query`` are
    cosine similarity values in [0, 1] (higher = more similar).
    """

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def _build_client(self) -> chromadb.ClientAPI:
        if settings.chroma_use_remote:
            logger.info(
                "chroma_http_client",
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            return chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )

        logger.info("chroma_persistent_client", path=settings.chroma_persist_directory)
        return chromadb.PersistentClient(path=settings.chroma_persist_directory)

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                # cosine distance → distance = 1 − cosine_similarity
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "chroma_collection_ready",
                name=settings.chroma_collection_name,
                count=self._collection.count(),
            )
        return self._collection

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[dict]) -> None:
        """
        Upsert chunked documents with their pre-computed embeddings.

        Each ``dict`` in ``documents`` must have:
          - ``id``        : stable string ID (deduplicated on re-ingest)
          - ``embedding`` : list[float]
          - ``content``   : raw chunk text stored alongside the vector
          - ``metadata``  : dict of scalar values (str / int / float / bool)
        """
        if not documents:
            return

        self.collection.upsert(
            ids=[d["id"] for d in documents],
            embeddings=[d["embedding"] for d in documents],
            documents=[d["content"] for d in documents],
            metadatas=[d["metadata"] for d in documents],
        )
        logger.info("chroma_upserted", count=len(documents))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict]:
        """
        Return the ``top_k`` most similar documents with cosine similarity
        above ``score_threshold``.

        Returns a list of dicts with keys:
          ``document_id``, ``content``, ``score``, ``metadata``
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc_id, content, meta, dist in zip(ids, docs, metas, dists):
            # ChromaDB cosine distance ∈ [0, 2]; normalised to similarity ∈ [-1, 1].
            # In practice with normalised vectors: similarity = 1 - distance.
            score = round(1.0 - dist, 6)
            if score >= score_threshold:
                chunks.append(
                    {
                        "document_id": doc_id,
                        "content": content,
                        "score": score,
                        "metadata": meta or {},
                    }
                )

        logger.debug("chroma_query_results", returned=len(chunks), top_k=top_k)
        return chunks

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Drop the collection — next access via ``collection`` will recreate it."""
        self.client.delete_collection(settings.chroma_collection_name)
        self._collection = None
        logger.warning("chroma_collection_deleted", name=settings.chroma_collection_name)
