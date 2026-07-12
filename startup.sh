#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app:${FASTERLIVEPORTRAIT_ROOT:-/opt/FasterLivePortrait}:${PYTHONPATH:-}"
mkdir -p "${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}" /tmp/arevik-avatar-engine

if [[ "${AVATAR_ENGINE_DOWNLOAD_MODELS:-false}" == "true" ]]; then
  /app/scripts/download_models.sh
fi

exec python3.10 -m uvicorn avatar_engine.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --workers 1
