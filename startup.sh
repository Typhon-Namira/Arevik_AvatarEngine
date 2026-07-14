#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app:${FASTERLIVEPORTRAIT_ROOT:-/opt/FasterLivePortrait}:${PYTHONPATH:-}"
mkdir -p "${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}" /tmp/arevik-avatar-engine

if [[ "${AVATAR_ENGINE_DOWNLOAD_MODELS:-true}" == "true" || "${AVATAR_ENGINE_MODE:-trt}" == "trt" ]]; then
  (
    echo "[AvatarEngine] Starting non-blocking model preparation"
    if /app/scripts/prepare_tensorrt.sh; then
      echo "[AvatarEngine] Background model preparation completed"
      date -Is > /tmp/arevik-avatar-engine/model-prep-ready
      rm -f /tmp/arevik-avatar-engine/model-prep-error
    else
      echo "[AvatarEngine] WARNING: background model preparation failed; realtime fallback renderer remains available" >&2
      date -Is > /tmp/arevik-avatar-engine/model-prep-error
    fi
  ) &
else
  echo "[AvatarEngine] Model preparation disabled by AVATAR_ENGINE_DOWNLOAD_MODELS=${AVATAR_ENGINE_DOWNLOAD_MODELS:-unset}"
fi

exec python3.10 -m uvicorn avatar_engine.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --workers 1
