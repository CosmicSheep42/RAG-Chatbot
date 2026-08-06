# Architecture

Local RAG Chatbot is a reusable `local_rag_chatbot` Python package with FastAPI as an adapter and a repository-hosted demonstration app.

The primary integration entry points are:

- `ChatbotConfig`
- `ChatbotEngine`
- `create_app()`
- `create_chat_router()`

`ChatbotEngine` owns composition and lifecycle for the existing services: routing, input cleaning, query rewriting, memory, retrieval, answer generation, tools, and indexing.

FastAPI routes call the engine and do not construct Ollama, vector-store backends, or internal RAG services themselves.

Authentication, conversation identity, and runtime controls are separate. The core engine can be called as
`await engine.chat("Question")` without HTTP, authentication, or conversation state. When a caller supplies
`conversation_id`, it is used only as a temporary memory key for history and query rewriting. Calls without an ID use
empty history and store nothing; no shared `default` conversation is created. The legacy `session_id` name remains as a
compatibility alias, but internal flow uses `conversation_id`.

The standalone FastAPI app is a local anonymous demo with process-level runtime controls. It does not implement API-key
authentication. Embedded FastAPI hosts should protect the router with their own dependencies/middleware and may pass an
optional `conversation_resolver` that maps already-authorized host context to a chatbot conversation ID.

The reusable wheel includes only core JSON defaults under `local_rag_chatbot/resources/`: the master prompt, tool-calling configuration, and routing configuration. These are resolved with `importlib.resources` and do not depend on the current working directory, editable installs, or repository layout.

External overrides are explicit. `MASTER_PROMPT_FILE`, `TOOL_CALLING_CONFIG_FILE`, and `ROUTING_DIR`, or the matching `ChatbotConfig.with_overrides()` fields, replace packaged defaults. Invalid explicit paths raise configuration errors and do not fall back silently.

The reference repository keeps demonstration resources outside the package: `knowledge-base/` for user-provided demo documents, `web/` for the static frontend, `docs/` for human documentation, Docker files for local deployment, and `data/` for generated runtime artifacts. These are excluded from the Python wheel.

`documents_dir` and `data_dir` are external runtime paths. The demo launcher points them at repository `knowledge-base/` and `data/`; embedded applications should provide host-owned directories. If no documents directory exists, indexing starts from an empty corpus.

`create_app()` can run API-only. It mounts static frontend files only when `web_dir` or `WEB_DIR` is supplied explicitly.

RAG index lifecycle favors reliability over incremental update speed. When documents, parser/chunker settings, embeddings,
or vector-store backend settings require an index change, the application asks the configured backend to replace the RAG
namespace and writes a matching manifest through `<manifest>.staging`. The Chroma backend preserves the existing local
directory staging and backup behavior for its persisted index. The PGVector backend uses namespace generations in a
single shared document table: new documents are inserted under an inactive generation, validated, promoted as the active
generation in a transaction, and only then are older rows removed. A small journal at
`<data_dir>/rag_index_transaction.json` records manifest promotion state. Startup and indexing first run journal recovery
and backend artifact cleanup so incomplete staging files or directories do not get reported as a ready index.

A process lock at `<data_dir>/rag_index.lock` coordinates startup indexing and later reindexing across processes; timeout
errors state that another process may be indexing. The lock uses `fcntl` on Unix-like systems and `msvcrt` on Windows,
with platform-specific imports isolated behind a common context manager. A leftover lock file is diagnostic only and
does not mean the lock is held.

Backends may add their own distributed coordination. PGVector takes deterministic PostgreSQL advisory locks derived from
the application name and namespace during schema setup, namespace replacement, and generation promotion. This prevents
two application instances connected to the same database from publishing competing partial rebuilds.

Live in-process reindexing uses a maintenance policy rather than index generations. `ChatbotEngine.reindex()` marks the
engine as reindexing, `/ready` reports not ready, new RAG chats fail with a temporary reindexing error, and active RAG
queries are allowed to finish before the index directory and manifest are promoted. If the rebuild fails, the transaction
recovery path preserves or restores a consistent active pair. Engine readiness requires initialization, no active
reindexing operation, and an indexed RAG retriever.

The router index has its own lock at `<data_dir>/router_index.lock`. Its persisted fingerprint is versioned and includes
the routing JSON hash, vector backend provider/schema, embedding provider type, embedding model, known embedding
dimensions, input normalization settings, and router index strategy/schema versions. Changing any of those values causes
a full router namespace rebuild; unchanged fingerprints reuse the existing stores. Older router hash files without the
versioned fingerprint are treated as incompatible and rebuilt once. Secrets and credential-bearing URLs are not stored in
the fingerprint payload.

Vector-store selection is isolated in `local_rag_chatbot.vectorstores.factory`. The domain layer talks only to the
`VectorStoreBackend` contract, which returns normalized relevance scores where `0.0` means least relevant and `1.0`
means most relevant. Chroma-specific metadata serialization, local directory promotion, `chroma.sqlite3` checks, and
private collection access remain inside the Chroma adapter. PGVector-specific SQL, JSONB metadata storage, pgvector
operators, extension creation, schema validation, and DSN redaction remain inside the PGVector adapter.

Docker keeps the common/Chroma services in `docker-compose.yml`. The
`docker-compose.pgvector.yml` override adds PostgreSQL and merges a
`depends_on.postgres.condition: service_healthy` dependency into `app`, so
the database is not started or required by the Chroma command. The adapter
also applies bounded retries for transient connection-establishment failures
using the `PGVECTOR_CONNECT_*` settings.
