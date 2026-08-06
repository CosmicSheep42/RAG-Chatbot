FROM python:3.12-slim

ARG INSTALL_RERANKER=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCUMENTS_DIR=/app/knowledge-base \
    DATA_DIR=/app/data \
    WEB_DIR=/app/web \
    RERANKING_MODE=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN if [ "$INSTALL_RERANKER" = "true" ]; then \
      pip install --no-cache-dir -e "/app[rerank]"; \
    else \
      pip install --no-cache-dir -e /app; \
    fi

COPY web /app/web
COPY knowledge-base /app/knowledge-base

RUN mkdir -p /app/knowledge-base /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; port = os.getenv('SERVER_PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health', timeout=3).read()"

# The module entry point reads SERVER_HOST and SERVER_PORT from AppSettings.
CMD ["python", "-m", "local_rag_chatbot.server"]
