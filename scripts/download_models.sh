#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}"
ENABLE_ANIMAL="${AVATAR_ENGINE_ENABLE_ANIMAL:-false}"

mkdir -p "$CHECKPOINT_DIR"

REQUIRED_CHECKPOINTS=(
  "$CHECKPOINT_DIR/liveportrait_onnx/warping_spade-fix.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/landmark.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/motion_extractor.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/retinaface_det_static.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/face_2dpose_106_static.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/libgrid_sample_3d_plugin.so"
  "$CHECKPOINT_DIR/liveportrait_onnx/appearance_feature_extractor.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching_eye.onnx"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching_lip.onnx"
)

HF_INCLUDES=(
  "liveportrait_onnx/*"
)

if [[ "$ENABLE_ANIMAL" == "true" ]]; then
  REQUIRED_CHECKPOINTS+=(
    "$CHECKPOINT_DIR/liveportrait_animal_onnx/xpose.pth"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx/clip_embedding_9.pkl"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx/clip_embedding_68.pkl"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/warping_spade-fix-v1.1.onnx"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/motion_extractor-v1.1.onnx"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/appearance_feature_extractor-v1.1.onnx"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/stitching-v1.1.onnx"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/stitching_eye-v1.1.onnx"
    "$CHECKPOINT_DIR/liveportrait_animal_onnx_v1.1/stitching_lip-v1.1.onnx"
  )
  HF_INCLUDES+=(
    "liveportrait_animal_onnx/*"
    "liveportrait_animal_onnx_v1.1/*"
  )
fi

all_present=1
for path in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [[ ! -s "$path" ]]; then
    all_present=0
    break
  fi
done

if [[ "$all_present" -eq 1 ]]; then
  echo "[AvatarEngine] FasterLivePortrait checkpoints already exist in $CHECKPOINT_DIR"
  exit 0
fi

echo "[AvatarEngine] Downloading FasterLivePortrait checkpoints into $CHECKPOINT_DIR"
if ! command -v hf >/dev/null 2>&1; then
  echo "[AvatarEngine] ERROR: Hugging Face CLI 'hf' is not installed in the image" >&2
  exit 1
fi

if ! hf download warmshao/FasterLivePortrait --local-dir "$CHECKPOINT_DIR" --include "${HF_INCLUDES[@]}"; then
  echo "[AvatarEngine] ERROR: failed to download FasterLivePortrait checkpoints with 'hf download'" >&2
  echo "[AvatarEngine] Check network access, disk space, and HF_TOKEN if the model repository becomes gated." >&2
  exit 1
fi

missing=0
for path in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "[AvatarEngine] ERROR: missing required checkpoint after download: $path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "[AvatarEngine] Checkpoints ready in $CHECKPOINT_DIR"
