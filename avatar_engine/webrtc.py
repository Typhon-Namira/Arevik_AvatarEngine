import asyncio
import fractions
import time
from typing import Any

import av
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from .faster_liveportrait_adapter import FasterLivePortraitAdapter


class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, session_id: str, adapter: FasterLivePortraitAdapter, fps: int = 25) -> None:
        super().__init__()
        self.session_id = session_id
        self.adapter = adapter
        self.fps = fps
        self.counter = 0
        self.start = time.time()

    async def recv(self) -> av.VideoFrame:
        await asyncio.sleep(1 / max(1, self.fps))
        frame_array = self.adapter.next_frame(self.session_id)
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        frame.pts = self.counter
        frame.time_base = fractions.Fraction(1, self.fps)
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
