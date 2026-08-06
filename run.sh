#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.11 or newer."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"

  echo "Installing project in editable mode..."
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -e .
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "The virtual environment exists but $VENV_PYTHON was not found."
  echo "Remove $VENV_DIR and run this script again."
  exit 1
fi

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export DOCUMENTS_DIR="${DOCUMENTS_DIR:-$PWD/knowledge-base}"
export DATA_DIR="${DATA_DIR:-$PWD/data}"
export WEB_DIR="${WEB_DIR:-$PWD/web}"

exec "$VENV_PYTHON" -m uvicorn \
  local_rag_chatbot.server:app \
  --host "${SERVER_HOST:-127.0.0.1}" \
  --port "${SERVER_PORT:-8000}"
