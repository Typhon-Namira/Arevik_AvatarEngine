#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}"

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
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/warping_spade-fix-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/motion_extractor-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/appearance_feature_extractor-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching_eye-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching_lip-v1.1.onnx"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/xpose.pth"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/clip_embedding_9.pkl"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/clip_embedding_68.pkl"
)

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
python3.10 -m pip install --quiet --upgrade huggingface_hub
huggingface-cli download warmshao/FasterLivePortrait --local-dir "$CHECKPOINT_DIR"

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
