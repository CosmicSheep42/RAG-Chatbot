#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

load_env_file() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    case "$line" in
      ""|\#*) continue ;;
    esac
    case "$line" in
      export\ *) line="${line#export }" ;;
    esac
    case "$line" in
      [A-Za-z_]*=*)
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
          *[!A-Za-z0-9_]*|"") continue ;;
        esac
        value="${value%"${value##*[![:space:]]}"}"
        if [ "${value#\"}" != "$value" ] && [ "${value%\"}" != "$value" ]; then
          value="${value#\"}"
          value="${value%\"}"
        elif [ "${value#\'}" != "$value" ] && [ "${value%\'}" != "$value" ]; then
          value="${value#\'}"
          value="${value%\'}"
        fi
        export "$key=$value"
        ;;
    esac
  done < "$env_file"
}

load_env_file "$PROJECT_ROOT/.env"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VECTOR_STORE_PROVIDER="${VECTOR_STORE_PROVIDER:-chroma}"
INSTALL_RERANKER="${INSTALL_RERANKER:-false}"

case "$VECTOR_STORE_PROVIDER" in
  chroma|pgvector) ;;
  *)
    echo "Unsupported VECTOR_STORE_PROVIDER: $VECTOR_STORE_PROVIDER" >&2
    echo "Supported values: chroma, pgvector" >&2
    exit 1
    ;;
esac

extras="$VECTOR_STORE_PROVIDER"
case "${INSTALL_RERANKER,,}" in
  1|true|yes|on) extras="$extras,rerank" ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.11 or newer."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"

  "$VENV_PYTHON" -m pip install --upgrade pip
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "The virtual environment exists but $VENV_PYTHON was not found."
  echo "Remove $VENV_DIR and run this script again."
  exit 1
fi

echo "Installing project in editable mode with extras: [$extras]"
"$VENV_PYTHON" -m pip install -e ".[$extras]"

export DOCUMENTS_DIR="${DOCUMENTS_DIR:-$PWD/knowledge-base}"
export DATA_DIR="${DATA_DIR:-$PWD/data}"
export WEB_DIR="${WEB_DIR:-$PWD/web}"

exec "$VENV_PYTHON" -m uvicorn \
  local_rag_chatbot.server:app \
  --host "${SERVER_HOST:-127.0.0.1}" \
  --port "${SERVER_PORT:-8000}"
