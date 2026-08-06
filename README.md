# Local RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139.2-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-prototype-orange)

A local Retrieval-Augmented Generation chatbot built as both a portfolio project and a reusable Python package.

The project demonstrates a complete RAG workflow with Ollama, ChromaDB, hybrid retrieval, citations, optional reranking, temporary conversation memory, query rewriting, and tool calling. It can run as a standalone FastAPI application or be embedded into another Python system.

> This is a learning-oriented prototype for local demonstrations and trusted environments. It is not a production-ready hosted service.

## Features

- Local generation and embeddings through Ollama.
- Hybrid retrieval with dense vector search and lexical matching.
- Persistent ChromaDB indexes.
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

## Requirements

- Python 3.11 or newer.
- Ollama installed and running.
- Enough local memory for the selected models.

Default models:

```text
llama3.2:3b
nomic-embed-text
```

## Quick Start

From the repository root:

```bash
cp .env.sample .env

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
.venv\Scripts\python.exe -m pip install -e .
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

Docker Compose starts Ollama, downloads the configured models, and runs the application. The published port is bound to `127.0.0.1` by default.

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
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `GENERATION_MODEL` | Answer generation model | `llama3.2:3b` |
| `EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `DOCUMENTS_DIR` | Knowledge-base directory | demo: `./knowledge-base` |
| `DATA_DIR` | Generated indexes and runtime data | demo: `./data` |
| `WEB_DIR` | Optional static frontend | demo: `./web` |
| `SERVER_HOST` | Standalone server host | `127.0.0.1` |
| `SERVER_PORT` | Standalone server port | `8000` |
| `RERANKING_MODE` | `auto`, `true`, or `false` | `auto` |
| `AUTO_INDEX_ON_STARTUP` | Load or build the index on startup | `true` |
| `REPLY_TIMEOUT_SECONDS` | Full response timeout | `60` |

`CHROMA_DB_DIR` and `ROUTER_STORE_DIR` are derived from `DATA_DIR` unless overridden.

`TOOL_CALLING_MODEL` and `QUERY_REWRITING_MODEL` use `GENERATION_MODEL` unless configured separately.

Legacy timeout variables remain accepted for compatibility, but new configurations should use `REPLY_TIMEOUT_SECONDS`.

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
- Do not expose Ollama or ChromaDB directly to untrusted networks.
- Conversation history is temporary and stored in memory.
- The demo frontend renders responses as plain text.
- Ollama is currently the only included model provider.
- Retrieval quality depends on the selected models, source documents, and tuning.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): architecture and index lifecycle.
- [`docs/retrieval-pipeline.md`](docs/retrieval-pipeline.md): retrieval and context selection.
- [`docs/security-model.md`](docs/security-model.md): deployment and security boundaries.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
