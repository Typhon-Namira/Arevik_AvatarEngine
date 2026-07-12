#!/usr/bin/env bash
set -euo pipefail

FASTERLIVEPORTRAIT_ROOT="${FASTERLIVEPORTRAIT_ROOT:-/opt/FasterLivePortrait}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/FasterLivePortrait/checkpoints}"
MODE="${AVATAR_ENGINE_MODE:-trt}"
FORCE_REBUILD="${AVATAR_ENGINE_FORCE_TRT_REBUILD:-false}"

if [[ "$MODE" != "trt" ]]; then
  echo "[AvatarEngine] AVATAR_ENGINE_MODE=$MODE; skipping TensorRT preparation"
  exit 0
fi

if [[ ! -d "$FASTERLIVEPORTRAIT_ROOT" ]]; then
  echo "[AvatarEngine] ERROR: FasterLivePortrait root does not exist: $FASTERLIVEPORTRAIT_ROOT" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[AvatarEngine] ERROR: python command is required by FasterLivePortrait TensorRT scripts" >&2
  exit 1
fi

if ! python -c "import tensorrt" >/dev/null 2>&1; then
  echo "[AvatarEngine] ERROR: TensorRT Python bindings are unavailable; use a TensorRT-enabled NVIDIA image/runtime" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

EXPECTED_REPO_CHECKPOINTS="$FASTERLIVEPORTRAIT_ROOT/checkpoints"
if [[ -L "$EXPECTED_REPO_CHECKPOINTS" ]]; then
  CURRENT_TARGET="$(readlink "$EXPECTED_REPO_CHECKPOINTS")"
  if [[ "$CURRENT_TARGET" != "$CHECKPOINT_DIR" ]]; then
    rm "$EXPECTED_REPO_CHECKPOINTS"
    ln -s "$CHECKPOINT_DIR" "$EXPECTED_REPO_CHECKPOINTS"
  fi
elif [[ -e "$EXPECTED_REPO_CHECKPOINTS" ]]; then
  if [[ "$(cd "$EXPECTED_REPO_CHECKPOINTS" && pwd)" != "$(cd "$CHECKPOINT_DIR" && pwd)" ]]; then
    echo "[AvatarEngine] Moving existing repo checkpoints into persistent volume"
    shopt -s dotglob nullglob
    mv "$EXPECTED_REPO_CHECKPOINTS"/* "$CHECKPOINT_DIR"/ 2>/dev/null || true
    rmdir "$EXPECTED_REPO_CHECKPOINTS" 2>/dev/null || rm -rf "$EXPECTED_REPO_CHECKPOINTS"
    ln -s "$CHECKPOINT_DIR" "$EXPECTED_REPO_CHECKPOINTS"
  fi
else
  ln -s "$CHECKPOINT_DIR" "$EXPECTED_REPO_CHECKPOINTS"
fi

echo "[AvatarEngine] FasterLivePortrait checkpoint path: $EXPECTED_REPO_CHECKPOINTS -> $CHECKPOINT_DIR"

/app/scripts/download_models.sh

REQUIRED_ONNX=(
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
)

REQUIRED_TRT=(
  "$CHECKPOINT_DIR/liveportrait_onnx/warping_spade-fix.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/landmark.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/motion_extractor.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/retinaface_det_static.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/face_2dpose_106_static.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/appearance_feature_extractor.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching_eye.trt"
  "$CHECKPOINT_DIR/liveportrait_onnx/stitching_lip.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/warping_spade-fix-v1.1.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/motion_extractor-v1.1.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/appearance_feature_extractor-v1.1.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching-v1.1.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching_eye-v1.1.trt"
  "$CHECKPOINT_DIR/liveportrait_animal_onnx/stitching_lip-v1.1.trt"
)

missing_onnx=0
for path in "${REQUIRED_ONNX[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "[AvatarEngine] ERROR: missing required ONNX checkpoint: $path" >&2
    missing_onnx=1
  fi
done
if [[ "$missing_onnx" -ne 0 ]]; then
  exit 1
fi

engines_missing=0
if [[ "$FORCE_REBUILD" == "true" ]]; then
  engines_missing=1
else
  for path in "${REQUIRED_TRT[@]}"; do
    if [[ ! -s "$path" ]]; then
      engines_missing=1
      break
    fi
  done
fi

if [[ "$engines_missing" -eq 0 ]]; then
  echo "[AvatarEngine] TensorRT engines already exist in persistent volume; skipping rebuild"
  exit 0
fi

echo "[AvatarEngine] Building TensorRT engines using official FasterLivePortrait scripts"
cd "$FASTERLIVEPORTRAIT_ROOT"

if ! bash scripts/all_onnx2trt.sh; then
  echo "[AvatarEngine] ERROR: human TensorRT engine generation failed" >&2
  exit 1
fi

if ! bash scripts/all_onnx2trt_animal.sh; then
  echo "[AvatarEngine] ERROR: animal TensorRT engine generation failed" >&2
  exit 1
fi

missing_trt=0
for path in "${REQUIRED_TRT[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "[AvatarEngine] ERROR: TensorRT engine was not generated: $path" >&2
    missing_trt=1
  fi
done

if [[ "$missing_trt" -ne 0 ]]; then
  exit 1
fi

echo "[AvatarEngine] TensorRT engines ready in $CHECKPOINT_DIR"
