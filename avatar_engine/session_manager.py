import time
from dataclasses import dataclass, field
from typing import Any

from aiortc import RTCPeerConnection


@dataclass
class AvatarSession:
    session_id: str
    avatar_image_url: str
    ice_servers: list[dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    last_audio_at: float | None = None
    peer: RTCPeerConnection | None = None
    audio_segments: int = 0


class AvatarSessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, AvatarSession] = {}

    def start(self, session_id: str, avatar_image_url: str, ice_servers: list[dict[str, Any]]) -> AvatarSession:
        session = self.sessions.get(session_id) or AvatarSession(session_id=session_id, avatar_image_url=avatar_image_url, ice_servers=ice_servers)
        self.sessions[session_id] = session
        return session

    async def stop(self, session_id: str) -> bool:
        session = self.sessions.pop(session_id, None)
        if session and session.peer:
            await session.peer.close()
        return session is not None

    def get(self, session_id: str) -> AvatarSession | None:
        return self.sessions.get(session_id)

    def metrics(self) -> dict[str, Any]:
        now = time.time()
        return {
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "age_seconds": round(now - session.created_at, 3),
                    "audio_segments": session.audio_segments,
                    "peer_connection_state": session.peer.connectionState if session.peer else "none",
                }
                for sid, session in self.sessions.items()
            },
        }
