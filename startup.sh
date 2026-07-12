#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app:${FASTERLIVEPORTRAIT_ROOT:-/opt/FasterLivePortrait}:${PYTHONPATH:-}"
mkdir -p "${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}" /tmp/arevik-avatar-engine

/app/scripts/prepare_tensorrt.sh

exec python3.10 -m uvicorn avatar_engine.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --workers 1
