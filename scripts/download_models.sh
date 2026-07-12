#!/usr/bin/env bash
set -euo pipefail

python3.10 -m pip install --quiet --upgrade huggingface_hub
huggingface-cli download warmshao/FasterLivePortrait --local-dir "${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}"
