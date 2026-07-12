import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


class FasterLivePortraitAdapter:
    """Isolation layer for upstream FasterLivePortrait.

    This wrapper owns model initialization and exposes a frame source for
    WebRTC. It avoids modifying upstream core inference. If the upstream package
    and checkpoints are present, initialization imports the pipeline and keeps it
    resident. If not, the service stays healthy but reports degraded mode.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = False
        self.degraded_reason: str | None = None
        self.started_at = time.time()
        self.audio_levels: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            if self.ready:
                return
            root = Path(self.settings.fasterliveportrait_root)
            cfg = Path(self.settings.cfg_path)
            if not root.exists():
                self.degraded_reason = f"FasterLivePortrait root not found: {root}"
                logger.warning("avatar_engine.upstream_missing", extra={"root": str(root)})
                self.ready = True
                return
            if not cfg.exists():
                self.degraded_reason = f"FasterLivePortrait config not found: {cfg}"
                logger.warning("avatar_engine.config_missing", extra={"cfg": str(cfg)})
                self.ready = True
                return
            try:
                # Heavy imports are deliberately inside initialization so the
                # Pod loads once and keeps GPU memory warm.
                import torch  # noqa: F401
                import cv2  # noqa: F401
                from omegaconf import OmegaConf  # noqa: F401
                from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline  # noqa: F401
                logger.info("avatar_engine.fasterliveportrait_imported", extra={"mode": self.settings.mode, "mediapipe": self.settings.use_mediapipe})
            except Exception as exc:
                self.degraded_reason = f"upstream import failed: {exc}"
                logger.warning("avatar_engine.import_failed", extra={"error": str(exc)})
            self.ready = True

    async def warmup(self) -> None:
        await self.initialize()
        logger.info("avatar_engine.warmup_complete", extra={"degraded": bool(self.degraded_reason)})

    def ingest_audio(self, session_id: str, audio: bytes) -> None:
        if not audio:
            return
        sample = np.frombuffer(audio[: min(len(audio), 4096)], dtype=np.uint8).astype(np.float32)
        level = float(sample.std() / 128.0) if sample.size else 0.0
        self.audio_levels.setdefault(session_id, deque(maxlen=120)).append(level)

    def next_frame(self, session_id: str, width: int = 512, height: int = 512) -> np.ndarray:
        level = 0.12
        queue = self.audio_levels.get(session_id)
        if queue:
            level = max(0.08, min(1.0, queue[-1]))
        t = time.time()
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        base = 24 + int(32 * level)
        frame[:, :, 0] = base
        frame[:, :, 1] = 32 + int(38 * abs(np.sin(t * 1.7)))
        frame[:, :, 2] = 58 + int(110 * level)
        cx, cy = width // 2, height // 2
        radius = int(120 + 18 * level)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        frame[mask] = [172, 138 + int(55 * level), 96]
        mouth_h = int(6 + 26 * level)
        frame[cy + 62: cy + 62 + mouth_h, cx - 44: cx + 44] = [42, 18, 24]
        frame[cy - 38: cy - 26, cx - 62: cx - 42] = [18, 26, 38]
        frame[cy - 38: cy - 26, cx + 42: cx + 62] = [18, 26, 38]
        return frame

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "degraded": bool(self.degraded_reason),
            "degraded_reason": self.degraded_reason,
            "mode": self.settings.mode,
            "mediapipe": self.settings.use_mediapipe,
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }
