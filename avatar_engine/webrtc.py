import logging
import time
from typing import Any

import av
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from .faster_liveportrait_adapter import FasterLivePortraitAdapter

logger = logging.getLogger(__name__)


class AvatarVideoTrack(VideoStreamTrack):
    def __init__(self, session_id: str, adapter: FasterLivePortraitAdapter, fps: int = 25) -> None:
        super().__init__()
        self.session_id = session_id
        self.adapter = adapter
        self.fps = fps
        self.counter = 0
        self.start = time.time()
        self._last_log_at = 0.0

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        frame_array = self.adapter.next_frame(self.session_id)
        if self.counter == 0 or time.time() - self._last_log_at > 5:
            logger.info(
                "avatar.webrtc.frame_generated",
                extra={
                    "session_id": self.session_id,
                    "frame": self.counter,
                    "shape": getattr(frame_array, "shape", None),
                    "min": int(frame_array.min()) if frame_array.size else None,
                    "max": int(frame_array.max()) if frame_array.size else None,
                },
            )
            self._last_log_at = time.time()
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24").reformat(format="yuv420p")
        frame.pts = pts
        frame.time_base = time_base
        self.counter += 1
        return frame


def rtc_configuration(ice_servers: list[dict[str, Any]]) -> RTCConfiguration:
    servers = []
    for item in ice_servers:
        urls = item.get("urls")
        if not urls:
            continue
        servers.append(RTCIceServer(urls=urls, username=item.get("username"), credential=item.get("credential")))
    return RTCConfiguration(iceServers=servers)


def parse_candidate(payload: dict[str, Any]):
    raw = payload.get("candidate") or {}
    candidate_line = raw.get("candidate")
    if not candidate_line:
        return None
    if candidate_line.startswith("candidate:"):
        candidate_line = candidate_line[len("candidate:"):]
    candidate = candidate_from_sdp(candidate_line)
    candidate.sdpMid = raw.get("sdpMid")
    candidate.sdpMLineIndex = raw.get("sdpMLineIndex")
    return candidate
