#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python -m uvicorn avatar_engine.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --reload
