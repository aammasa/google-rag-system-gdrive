# Google Drive RAG Agent

A production-ready backend that ingests Google Drive documents, indexes them in
ChromaDB via Vertex AI embeddings, and answers questions through a Gemini-powered
RAG pipeline surfaced as a Google Chat bot.

## Architecture

```
Google Drive
     │  (OAuth / Service Account)
     ▼
Ingestion Service        ← DriveClient fetches & chunks docs
     │
     ▼
Vertex AI Embeddings     ← text-embedding-005
     │
     ▼
ChromaDB                 ← persisted vector store
     │
     ▼
Retriever                ← cosine similarity search
     │
     ▼
RagAgent (Gemini LLM)    ← grounded answer generation
     │
     ├── FastAPI REST API  (/api/v1/chat, /api/v1/retrieval, /api/v1/ingestion)
     └── Google Chat Bot   (/api/v1/chat/webhook)
```

## Folder Structure

```
.
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── config.py            # Pydantic settings (env-driven)
│   ├── logger.py            # Structlog setup (JSON prod / colored dev)
│   ├── middleware.py        # Request-ID logging middleware
│   ├── dependencies.py      # FastAPI DI providers
│   ├── api/
│   │   ├── router.py        # Top-level router assembly
│   │   └── v1/
│   │       ├── health.py    # /health, /ready
│   │       ├── ingestion.py # /api/v1/ingestion/sync
│   │       ├── retrieval.py # /api/v1/retrieval/search
│   │       └── chat.py      # /api/v1/chat/message, /webhook
│   ├── core/
│   │   └── exceptions.py    # Domain exceptions + FastAPI handlers
│   ├── ingestion/
│   │   └── drive_client.py  # Google Drive API wrapper
│   ├── embeddings/
│   │   └── vertex_embeddings.py  # Vertex AI embedding model
│   ├── vectorstore/
│   │   └── chroma_store.py  # ChromaDB client
│   ├── retrieval/
│   │   └── retriever.py     # Query → embed → search pipeline
│   ├── chat/
│   │   └── agent.py         # RagAgent (retrieval + LLM)
│   ├── prompts/
│   │   └── templates.py     # Prompt strings (version-controlled)
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── chat_service.py
│   └── utils/
│       └── helpers.py       # slugify, stable_id, chunk_list, etc.
├── tests/
│   └── test_health.py
├── scripts/
│   └── setup.sh             # One-shot local dev bootstrap
├── .vscode/
│   ├── settings.json
│   └── extensions.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml           # ruff + mypy + pytest config
├── .pre-commit-config.yaml
├── .env.example
└── .gitignore
```

## Quick Start (Local)

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- A Google Cloud project with Vertex AI and Drive APIs enabled
- OAuth 2.0 credentials or a service account key

### 2. One-command setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in GOOGLE_CLOUD_PROJECT, credentials paths, etc.
```

### 4. Run the development server

```bash
uvicorn app.main:app --reload --port 8080
```

Open the interactive API docs at <http://localhost:8080/docs>.

### 5. Run tests

```bash
pytest
```

## Docker

### Build & run with Docker Compose (recommended)

```bash
# Start API + ChromaDB
docker compose up --build

# Run in background
docker compose up -d --build

# Tail logs
docker compose logs -f api

# Stop everything
docker compose down
```

### Build the image only

```bash
docker build -t google-rag-agent:local .
```

### Run standalone container

```bash
docker run --rm \
  --env-file .env \
  -p 8080:8080 \
  google-rag-agent:local
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe |
| POST | `/api/v1/ingestion/sync` | Trigger Drive ingestion |
| GET | `/api/v1/ingestion/status/{job_id}` | Poll ingestion status |
| POST | `/api/v1/retrieval/search` | Semantic search |
| POST | `/api/v1/chat/message` | RAG chat turn |
| POST | `/api/v1/chat/webhook` | Google Chat event receiver |

## Google Cloud Setup

```bash
# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  drive.googleapis.com \
  chat.googleapis.com \
  --project=$GOOGLE_CLOUD_PROJECT

# Create a service account for Cloud Run
gcloud iam service-accounts create rag-agent \
  --display-name="RAG Agent" \
  --project=$GOOGLE_CLOUD_PROJECT

# Grant Vertex AI user role
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="serviceAccount:rag-agent@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

## Deploy to Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/rag-agent

# Deploy
gcloud run deploy rag-agent \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/rag-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT \
  --service-account rag-agent@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com
```

## Development Workflow

```bash
# Format & lint
ruff format app tests
ruff check app tests --fix

# Type check
mypy app

# Run tests with coverage
pytest --cov=app --cov-report=term-missing

# Run pre-commit on all files
pre-commit run --all-files
```

## Implementation Roadmap

Each module contains `# TODO:` stubs — implement in this order:

1. `app/ingestion/drive_client.py` — OAuth flow + file listing/download
2. `app/embeddings/vertex_embeddings.py` — Vertex AI embedding model init
3. `app/vectorstore/chroma_store.py` — ChromaDB client + upsert/query
4. `app/retrieval/retriever.py` — wire embeddings + vectorstore
5. `app/services/ingestion_service.py` — chunk, embed, store pipeline
6. `app/chat/agent.py` — RAG chain with Gemini LLM
7. `app/api/v1/*.py` — replace stub responses with service calls
8. `app/main.py` lifespan — initialise all clients on startup
