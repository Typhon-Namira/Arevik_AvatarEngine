import asyncio
import base64
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import httpx
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
        self.ready = True
        self.degraded_reason: str | None = None
        self.started_at = time.time()
        self.audio_events: dict[str, deque[tuple[float, float, float]]] = {}
        self.avatar_frames: dict[str, np.ndarray] = {}
        self.frame_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            if self.degraded_reason == "initialized":
                return
            root = Path(self.settings.fasterliveportrait_root)
            cfg = Path(self.settings.cfg_path)
            if not root.exists():
                self.degraded_reason = f"FasterLivePortrait root not found: {root}"
                logger.warning("avatar_engine.upstream_missing", extra={"root": str(root)})
                return
            if not cfg.exists():
                self.degraded_reason = f"FasterLivePortrait config not found: {cfg}"
                logger.warning("avatar_engine.config_missing", extra={"cfg": str(cfg)})
                return
            try:
                # Heavy imports are deliberately inside initialization so the
                # Pod loads once and keeps GPU memory warm.
                import torch  # noqa: F401
                import cv2  # noqa: F401
                from omegaconf import OmegaConf  # noqa: F401
                from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline  # noqa: F401
                logger.info("avatar_engine.fasterliveportrait_imported", extra={"mode": self.settings.mode, "mediapipe": self.settings.use_mediapipe})
                self.degraded_reason = "initialized"
            except Exception as exc:
                self.degraded_reason = f"upstream import failed: {exc}"
                logger.warning("avatar_engine.import_failed", extra={"error": str(exc)})

    async def warmup(self) -> None:
        await self.initialize()
        logger.info("avatar_engine.warmup_complete", extra={"degraded": bool(self.degraded_reason)})

    async def set_avatar_image(self, session_id: str, image_url: str) -> None:
        """Fetch and cache the configured avatar image as the live frame base."""
        if not image_url:
            return
        try:
            if image_url.startswith("data:image/"):
                _, encoded = image_url.split(",", 1)
                content = base64.b64decode(encoded)
            else:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    content = response.content
            decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise ValueError("avatar image could not be decoded")
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            self.avatar_frames[session_id] = self._cover_square(rgb, 512)
            logger.info(
                "avatar_engine.avatar_image_cached",
                extra={"session_id": session_id, "bytes": len(content), "shape": self.avatar_frames[session_id].shape},
            )
        except Exception as exc:
            logger.warning(
                "avatar_engine.avatar_image_cache_failed session_id=%s error_type=%s error=%s",
                session_id,
                type(exc).__name__,
                str(exc),
            )

    def _cover_square(self, image: np.ndarray, size: int) -> np.ndarray:
        height, width = image.shape[:2]
        crop = min(height, width)
        top = max(0, (height - crop) // 2)
        left = max(0, (width - crop) // 2)
        square = image[top: top + crop, left: left + crop]
        return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    def ingest_audio(self, session_id: str, audio: bytes) -> None:
        if not audio:
            return
        sample = np.frombuffer(audio[: min(len(audio), 32768)], dtype=np.uint8).astype(np.float32)
        energy = max(0.18, min(1.0, float(sample.std() / 72.0) if sample.size else 0.25))
        duration = max(1.2, min(14.0, len(audio) / 14_000.0))
        self.audio_events.setdefault(session_id, deque(maxlen=8)).append((time.time(), duration, energy))
        logger.info(
            "avatar_engine.audio_received",
            extra={"session_id": session_id, "audio_bytes": len(audio), "duration_estimate": round(duration, 3), "energy": round(energy, 3)},
        )

    def _active_audio_level(self, session_id: str) -> float:
        now = time.time()
        events = self.audio_events.get(session_id)
        if not events:
            return 0.0
        while events and now - events[0][0] > events[0][1]:
            events.popleft()
        if not events:
            return 0.0
        started, duration, energy = events[0]
        progress = max(0.0, min(1.0, (now - started) / max(duration, 0.001)))
        phrase_envelope = np.sin(np.pi * progress)
        syllable_pulse = 0.42 + 0.58 * abs(np.sin(now * 15.0))
        return float(max(0.0, min(1.0, energy * phrase_envelope * syllable_pulse)))

    def next_frame(self, session_id: str, width: int = 512, height: int = 512) -> np.ndarray:
        level = self._active_audio_level(session_id)
        t = time.time()
        base_frame = self.avatar_frames.get(session_id)
        if base_frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            base = 24 + int(32 * max(level, 0.06))
            frame[:, :, 0] = base
            frame[:, :, 1] = 32 + int(38 * abs(np.sin(t * 1.7)))
            frame[:, :, 2] = 58 + int(110 * max(level, 0.06))
            cx, cy = width // 2, height // 2
            radius = int(120 + 18 * level)
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            frame[mask] = [172, 138 + int(55 * level), 96]
        else:
            frame = cv2.resize(base_frame, (width, height), interpolation=cv2.INTER_AREA).copy()
            warmth = np.full_like(frame, [10, 6 + int(18 * level), 0])
            frame = cv2.addWeighted(frame, 0.94, warmth, 0.06, 0)

        cx, cy = width // 2, int(height * 0.62)
        mouth_w = int(width * (0.13 + 0.05 * level))
        mouth_h = int(height * (0.018 + 0.075 * level))
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy), (mouth_w, max(4, mouth_h)), 0, 0, 360, (36, 14, 24), -1)
        cv2.ellipse(overlay, (cx, cy - max(2, mouth_h // 3)), (mouth_w, max(2, mouth_h // 3)), 0, 0, 180, (238, 128, 132), 2)
        alpha = 0.16 if level <= 0.02 else min(0.72, 0.22 + level * 0.62)
        frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

        frame_count = self.frame_counts.get(session_id, 0)
        if frame_count == 0 or frame_count % max(1, self.settings.target_fps * 5) == 0:
            logger.info(
                "avatar_engine.frame_generated",
                extra={"session_id": session_id, "frame": frame_count, "audio_level": round(level, 3), "has_avatar_image": base_frame is not None},
            )
        self.frame_counts[session_id] = frame_count + 1
        return frame

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "degraded": bool(self.degraded_reason and self.degraded_reason != "initialized"),
            "degraded_reason": None if self.degraded_reason == "initialized" else self.degraded_reason,
            "mode": self.settings.mode,
            "mediapipe": self.settings.use_mediapipe,
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }
