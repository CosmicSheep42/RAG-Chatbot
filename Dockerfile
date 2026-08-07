FROM python:3.12-slim

ARG INSTALL_RERANKER=false
ARG VECTOR_STORE_PROVIDER=chroma
ARG LLM_PROVIDER=ollama
ARG EMBEDDING_PROVIDER=ollama

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCUMENTS_DIR=/app/knowledge-base \
    DATA_DIR=/app/data \
    WEB_DIR=/app/web \
    VECTOR_STORE_PROVIDER=${VECTOR_STORE_PROVIDER} \
    LLM_PROVIDER=${LLM_PROVIDER} \
    EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER} \
    RERANKING_MODE=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN set -eu; \
    case "$VECTOR_STORE_PROVIDER" in \
      chroma|pgvector) ;; \
      *) echo "Unsupported VECTOR_STORE_PROVIDER: $VECTOR_STORE_PROVIDER" >&2; exit 1 ;; \
    esac; \
    case "$LLM_PROVIDER" in \
      ollama|openai) ;; \
      *) echo "Unsupported LLM_PROVIDER: $LLM_PROVIDER" >&2; exit 1 ;; \
    esac; \
    case "$EMBEDDING_PROVIDER" in \
      ollama|openai) ;; \
      *) echo "Unsupported EMBEDDING_PROVIDER: $EMBEDDING_PROVIDER" >&2; exit 1 ;; \
    esac; \
    extras="$VECTOR_STORE_PROVIDER,$LLM_PROVIDER,$EMBEDDING_PROVIDER"; \
    if [ "$INSTALL_RERANKER" = "true" ]; then \
      extras="$extras,rerank"; \
    fi; \
    pip install --no-cache-dir -e "/app[$extras]"

COPY web /app/web
COPY knowledge-base /app/knowledge-base

RUN mkdir -p /app/knowledge-base /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; port = os.getenv('SERVER_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health', timeout=3).read()"

# The module entry point reads SERVER_HOST and SERVER_PORT from AppSettings.
CMD ["python", "-m", "local_rag_chatbot.server"]
