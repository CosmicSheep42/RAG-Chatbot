# Local RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139.2-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-prototype-orange)

A local Retrieval-Augmented Generation chatbot built as both a portfolio project and a reusable Python package.

The project demonstrates a complete RAG workflow with pluggable generation, embedding, and vector-storage providers, hybrid retrieval, citations, optional reranking, temporary conversation memory, query rewriting, and tool calling. It can run as a standalone FastAPI application or be embedded into another Python system.

> This is a learning-oriented prototype for local demonstrations and trusted environments. It is not a production-ready hosted service.

## Features

- Pluggable generation and embedding providers: Ollama (fully local) or OpenAI (server-side API).
- Hybrid retrieval with dense vector search and lexical matching.
- Pluggable vector storage with local Chroma or PostgreSQL/pgvector.
- Citations linked to retrieved document passages.
- Optional cross-encoder reranking.
- Temporary conversation memory and follow-up query rewriting.
- Configurable tool calling.
- Standalone FastAPI application.
- Embeddable FastAPI router.
- Programmatic asynchronous Python API.
- Support for TXT, Markdown, PDF, DOCX, XLSX, CSV, JSON, HTML, and PPTX files.

## Architecture

```text
Python caller or FastAPI adapter
            |
            v
      ChatbotEngine
            |
            +--> input cleaning and routing
            +--> memory and query rewriting
            +--> retrieval and context assembly
            +--> answer generation and tool calling
            +--> ChatResponse with citations
```

`ChatbotEngine` is the main public entry point. The FastAPI layer acts as an adapter around the engine.

The installed package contains the reusable chatbot code and its default prompt, routing, and tool-calling resources. The demo frontend, knowledge base, generated indexes, Docker configuration, and documentation remain repository resources.

Generation, embeddings, and vector storage are three independent, swappable boundaries:

```text
AppSettings
    |
    +--> create_chat_model()        --> Ollama or OpenAI      (LLM_PROVIDER)
    +--> create_embeddings()        --> Ollama or OpenAI       (EMBEDDING_PROVIDER)
    +--> create_vector_store_backend() --> Chroma or PGVector  (VECTOR_STORE_PROVIDER)
            |
            v
    create_application_services() --> ChatModel + Embeddings + VectorStoreBackend
            |
            v
        ChatbotEngine
```

`local_rag_chatbot/providers/llm/` and `local_rag_chatbot/providers/embeddings/` hold the provider contracts, factories, and adapters (`ollama.py`, `openai.py`). Domain code (`RAGChatbot`, `AnswerService`, `QueryRewriter`, `IndexManager`, the vector-store backends) only depends on the provider-neutral `ChatModel`/`Embeddings` contracts — it never imports an Ollama or OpenAI type directly. `ChatbotEngine(config=..., chat_model=..., embeddings=...)` still accepts explicit dependency injection, which always takes precedence over the provider selected by configuration.

## Requirements

- Python 3.11 or newer.
- Either Ollama installed and running locally, or an OpenAI API key — see [LLM & Embedding Providers](#llm--embedding-providers).
- Enough local memory for the selected models when running Ollama.

## Testing

Install the test dependencies and run the checks from the repository root:

```bash
python -m pip install -e ".[test]"
pytest --cov=local_rag_chatbot --cov-report=term-missing
ruff check .
```

## Evaluation dataset

The Datapizza AI Lab D&D 5e corpus contains 20 Markdown documents in `knowledge-base/` and 56 preserved QA examples in `evaluation/dataset/datapizza/`. Its deterministic manifest has 12 development examples and 44 held-out test examples. Validate it offline with:

```bash
python evaluation/validate_dataset.py
```

Run the offline BM25 retrieval benchmark with:

```bash
python -m evaluation.benchmark --configuration bm25 --split dev --output evaluation/results/datapizza/bm25-dev.json
```

The runner records per-question retrieval metrics and reproducibility metadata. The deterministic BM25 mode is CI-safe. When providers and a vector-store extra are configured, run a real configuration explicitly with `--live`; add `--end-to-end` for answer generation:

```bash
python -m evaluation.benchmark --live --configuration hybrid --split dev --end-to-end \
  --output /tmp/hybrid-dev.json
```

Standalone, dense, hybrid, reranking, and full-pipeline modes initialize their requested providers explicitly and fail when a required component is unavailable. Live experiments are never run by default in CI.
End-to-end records keep factual coverage, source-level citation support, abstention outcomes, and a manual-review payload separate; retrieved evidence presence is not treated as automatic groundedness.
The Docker scenario preparation and execution commands are in [docs/datapizza-benchmark.md](docs/datapizza-benchmark.md). The measured results are summarized in [evaluation/RESULTS.md](evaluation/RESULTS.md), with selected per-question records in [evaluation/published/datapizza/](evaluation/published/datapizza/).

### Experimental Evaluation

The Datapizza evaluation used 20 D&D 5e Markdown documents and 56 questions (12 development, 44 held-out test) on a Debian 13 host with an NVIDIA RTX 3050 Laptop GPU. The generation model was `llama3.2:3b`, embeddings used `nomic-embed-text`, and `BAAI/bge-reranker-base` ran on CPU. On the held-out split, retrieval results were:

| Method | Recall@5 | nDCG@5 | Weighted passage coverage |
|---|---:|---:|---:|
| BM25 | 0.6936 | 0.6734 | 0.3002 |
| Dense | 0.6605 | 0.6655 | 0.2472 |
| Hybrid | 0.7446 | 0.7360 | 0.3298 |
| Hybrid + rerank | 0.7576 | 0.7592 | 0.4017 |

These are measurements on this fixed corpus, not universal performance claims. End-to-end latency and limitations are documented in the [results report](evaluation/RESULTS.md). Answer correctness, completeness, and evidence grounding have not yet been independently evaluated. An LLM-assisted evaluation is planned; the repository publishes an unscored blind-review input and rubric, not completed judge scores. Datapizza attribution and licensing details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Default models (Ollama):

```text
llama3.2:3b
nomic-embed-text
```

## Quick Start

From the repository root:

```bash
cp .env.sample .env
# Set DOCUMENTS_DIR=./knowledge-base in .env to use the Datapizza demonstration corpus.

ollama pull llama3.2:3b
ollama pull nomic-embed-text

chmod +x run.sh
./run.sh
```

Open `http://127.0.0.1:8000`.

`run.sh` creates a virtual environment when necessary, installs the package in editable mode, loads `.env`, and starts the standalone FastAPI application.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[chroma,ollama]"
Copy-Item .env.sample .env

ollama pull llama3.2:3b
ollama pull nomic-embed-text

.venv\Scripts\python.exe -m uvicorn local_rag_chatbot.server:app --host 127.0.0.1 --port 8000
```

### Docker

```bash
cp .env.sample .env
docker compose up --build
```

By default (`.env.sample`'s `COMPOSE_PROFILES=ollama` and `COMPOSE_FILE=docker-compose.yml:docker-compose.ollama.yml`), this is unchanged from before: Compose starts Ollama, downloads the configured models, waits for the pull to finish, then starts `app`. The published port is bound to `127.0.0.1` by default.

Ollama is optional in Docker. To run with `LLM_PROVIDER=openai`/`EMBEDDING_PROVIDER=openai` and skip the Ollama containers entirely, clear the two Compose-only variables so the Ollama override file and profile aren't activated:

```bash
COMPOSE_PROFILES= COMPOSE_FILE=docker-compose.yml \
LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai \
docker compose up --build app
```

`COMPOSE_PROFILES` and `COMPOSE_FILE` are read by the Compose CLI itself (not by the application) — they control which compose files are merged and which profiles are active, so the `ollama`/`ollama-init` services are skipped and `app` no longer waits on them.

For PGVector with Docker Compose (works with either LLM/embedding provider). Note that once `COMPOSE_FILE` is
overridden on the command line, `COMPOSE_PROFILES` must be passed explicitly too — Compose stops reading it from
`.env` as soon as `COMPOSE_FILE` comes from the shell environment instead:

```bash
VECTOR_STORE_PROVIDER=pgvector COMPOSE_PROFILES=ollama \
COMPOSE_FILE=docker-compose.yml:docker-compose.pgvector.yml:docker-compose.ollama.yml \
docker compose up --build
```

Or, PGVector + OpenAI, no Ollama at all:

```bash
VECTOR_STORE_PROVIDER=pgvector LLM_PROVIDER=openai EMBEDDING_PROVIDER=openai \
COMPOSE_PROFILES= COMPOSE_FILE=docker-compose.yml:docker-compose.pgvector.yml \
docker compose up --build app
```

Changing `VECTOR_STORE_PROVIDER`, `LLM_PROVIDER`, or `EMBEDDING_PROVIDER` requires rebuilding the image, because the corresponding optional dependency extras are installed at build time (see [LLM & Embedding Providers](#llm--embedding-providers)).

## Python Usage

```python
import asyncio

from local_rag_chatbot import ChatbotConfig, ChatbotEngine


async def main() -> None:
    engine = ChatbotEngine(config=ChatbotConfig.from_env())

    try:
        await engine.initialize()
        response = await engine.chat("What does the knowledge base say about billing?")

        print(response.result)
        print(response.citations)
    finally:
        await engine.shutdown()


asyncio.run(main())
```

Calls without a `conversation_id` are stateless. Supply an identifier when temporary conversation memory is needed:

```python
response = await engine.chat(
    "What did it say about the deadline?",
    conversation_id="user-42:conversation-7",
)
```

A conversation identifier is only a memory key. It is not authentication or authorization.

## FastAPI Integration

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from local_rag_chatbot import ChatbotConfig, ChatbotEngine
from local_rag_chatbot.fastapi import create_chat_router


engine = ChatbotEngine(config=ChatbotConfig.from_env())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.initialize()
    try:
        yield
    finally:
        await engine.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(
    create_chat_router(engine),
    prefix="/api/assistant",
)
```

The host application should provide authentication, authorization, tenant isolation, rate limiting, and deployment-specific middleware.

## Configuration

Copy `.env.sample` to `.env` and change only the settings required by your environment.

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | Generation provider: `ollama` or `openai`. | `ollama` |
| `EMBEDDING_PROVIDER` | Embedding provider: `ollama` or `openai`. Independent of `LLM_PROVIDER`. | `ollama` |
| `GENERATION_MODEL` | Answer generation model name for the selected `LLM_PROVIDER`. | `llama3.2:3b` |
| `EMBEDDING_MODEL` | Embedding model name for the selected `EMBEDDING_PROVIDER`. | `nomic-embed-text` |
| `OLLAMA_BASE_URL` | Ollama server URL. Only used when a provider is set to `ollama`. | `http://localhost:11434` |
| `OPENAI_API_KEY` | OpenAI API key. Required when `LLM_PROVIDER` or `EMBEDDING_PROVIDER` is `openai`. Never logged, never returned by `/status`. | empty |
| `OPENAI_BASE_URL` | Optional custom endpoint compatible with the OpenAI Responses API and/or Embeddings API used by the provider (not a generic OpenAI-compatible/Chat Completions endpoint). | empty (SDK default) |
| `DOCUMENTS_DIR` | Knowledge-base directory | demo: `./knowledge-base` |
| `DATA_DIR` | Generated indexes and runtime data | demo: `./data` |
| `VECTOR_STORE_PROVIDER` | Vector-store backend provider: `chroma` or `pgvector`. | `chroma` |
| `PGVECTOR_DSN` | SQLAlchemy-style PostgreSQL DSN for PGVector. Required only for `pgvector`. | empty |
| `PGVECTOR_SCHEMA` | PostgreSQL schema for vector tables. | `rag` |
| `PGVECTOR_TABLE` | Shared document table for PGVector namespaces. | `vector_documents` |
| `PGVECTOR_COLLECTION_TABLE` | PGVector namespace/generation table. | `vector_collections` |
| `PGVECTOR_DISTANCE_STRATEGY` | PGVector distance metric: `cosine`, `l2`, or `inner_product`. | `cosine` |
| `PGVECTOR_INDEX_TYPE` | Optional PGVector index: `hnsw`, `ivfflat`, or `none`. | `hnsw` |
| `PGVECTOR_CREATE_EXTENSION` | Whether the app tries `CREATE EXTENSION IF NOT EXISTS vector`. | `true` |
| `PGVECTOR_CONNECT_MAX_ATTEMPTS` | Maximum PGVector connection attempts for transient startup failures. | `12` |
| `PGVECTOR_CONNECT_RETRY_SECONDS` | Delay between transient PGVector connection attempts. | `2` |
| `PGVECTOR_CONNECT_TIMEOUT_SECONDS` | Timeout for each PGVector connection attempt. | `5` |
| `WEB_DIR` | Optional static frontend | demo: `./web` |
| `SERVER_HOST` | Standalone server host | `127.0.0.1` |
| `SERVER_PORT` | Standalone server port | `8000` |
| `RERANKING_MODE` | `auto`, `true`, or `false` | `auto` |
| `AUTO_INDEX_ON_STARTUP` | Load or build the index on startup | `true` |
| `REPLY_TIMEOUT_SECONDS` | Full response timeout | `60` |

`CHROMA_DB_DIR` and `ROUTER_STORE_DIR` are derived from `DATA_DIR` unless overridden.

`TOOL_CALLING_MODEL` and `QUERY_REWRITING_MODEL` use `GENERATION_MODEL` unless configured separately.

## LLM & Embedding Providers

Generation and embeddings are selected independently through `LLM_PROVIDER` and `EMBEDDING_PROVIDER`. Both extras must be installed for the providers you use:

```bash
# Ollama (fully local)
pip install -e ".[chroma,ollama]"

# OpenAI (server-side API)
pip install -e ".[chroma,openai]"
```

"OpenAI" here means the OpenAI **API**, called server-side with the official `openai` Python SDK — it is not a dependency on the ChatGPT web app, and no request ever reaches the browser directly; the API key stays server-side and is never sent to the frontend, logged, or returned by `/status`/`/ready`.

Example combinations:

```env
# Fully local
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
VECTOR_STORE_PROVIDER=chroma
```

```env
# OpenAI generation + embeddings, local vector storage
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
VECTOR_STORE_PROVIDER=chroma
OPENAI_API_KEY=sk-...
```

```env
# OpenAI generation, local embeddings, PGVector storage
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=ollama
VECTOR_STORE_PROVIDER=pgvector
OPENAI_API_KEY=sk-...
```

`./run.sh` and the Docker build read `LLM_PROVIDER`/`EMBEDDING_PROVIDER` and install the matching optional extras automatically. Switching `EMBEDDING_PROVIDER` (or `EMBEDDING_MODEL`) changes the index fingerprint, so the next startup or reindex safely rebuilds incompatible indexes instead of serving mismatched vectors.

## Vector Store Backends

The project supports two optional vector-store backends:

```bash
# Local Chroma
pip install -e ".[chroma]"

# PostgreSQL with pgvector
pip install -e ".[pgvector]"
```

Select the backend in `.env`:

```env
VECTOR_STORE_PROVIDER=chroma
```

Use `pgvector` instead of `chroma` to connect to PostgreSQL. In that case, configure `PGVECTOR_DSN` and ensure the PostgreSQL `vector` extension is available.

`./run.sh` reads `VECTOR_STORE_PROVIDER` (and `LLM_PROVIDER`/`EMBEDDING_PROVIDER`) and installs the corresponding optional dependencies. When using Docker, follow the commands in the Docker section above.

## Optional Reranking

The base installation works without reranking.

```bash
pip install -e ".[rerank]"
```

```env
RERANKING_MODE=auto
RERANKER_MODEL=BAAI/bge-reranker-base
```

Reranking may improve difficult searches, but it increases installation size, memory usage, startup time, and response latency.

## Security and Limitations

- Do not expose the standalone application directly to the public internet.
- The standalone demo does not include chatbot-managed authentication.
- Embedded deployments must use the host application's security model.
- Do not expose Ollama, ChromaDB files, or PostgreSQL/pgvector directly to untrusted networks.
- Do not log or commit real PostgreSQL DSNs or `OPENAI_API_KEY` values with credentials. `.env` is git-ignored; only empty placeholders belong in `.env.sample`.
- All OpenAI API calls happen server-side (inside the LLM/embedding adapters); the API key is never sent to the frontend.
- Conversation history is temporary and stored in memory.
- The demo frontend renders responses as plain text.
- Ollama and OpenAI are the currently included model providers.
- Retrieval quality depends on the selected models, source documents, and tuning.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): architecture and index lifecycle.
- [`docs/retrieval-pipeline.md`](docs/retrieval-pipeline.md): retrieval and context selection.
- [`docs/security-model.md`](docs/security-model.md): deployment and security boundaries.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
