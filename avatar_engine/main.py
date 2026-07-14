import base64
import asyncio
import json
import logging
import time
from typing import Any

import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .config import Settings, get_settings
from .faster_liveportrait_adapter import FasterLivePortraitAdapter
from .logging_config import configure_logging
from .session_manager import AvatarSessionManager
from .webrtc import AvatarVideoTrack, parse_candidate, rtc_configuration

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()
adapter = FasterLivePortraitAdapter(settings)
sessions = AvatarSessionManager()
app = FastAPI(title=settings.service_name, version="1.0.0")


class SessionStartRequest(BaseModel):
    session_id: str
    avatar_image_url: str | None = None
    request_id: str | None = None
    metadata: dict[str, Any] | None = None
    ice_servers: list[dict[str, Any]] | None = None


class SessionStopRequest(BaseModel):
    session_id: str


class AudioRequest(BaseModel):
    session_id: str
    audio_base64: str
    audio_format: str = "mp3"
    request_id: str | None = None
    metadata: dict[str, Any] | None = None


async def authorize(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="invalid avatar engine API key")


def default_ice_servers() -> list[dict[str, Any]]:
    servers: list[dict[str, Any]] = []
    if settings.stun_url:
        servers.append({"urls": settings.stun_url})
    if settings.turn_url:
        servers.append({"urls": settings.turn_url, "username": settings.turn_username, "credential": settings.turn_password})
    return servers


@app.on_event("startup")
async def startup() -> None:
    await adapter.warmup()


@app.on_event("shutdown")
async def shutdown() -> None:
    for session_id in list(sessions.sessions):
        await sessions.stop(session_id)


@app.get("/health")
async def health() -> dict[str, Any]:
    runtime = adapter.health()
    return {
        "status": "ready" if runtime["ready"] else "starting",
        "ready": runtime["ready"],
        "runtime": runtime,
        "capabilities": {
            "webrtc": True,
            "websocket_signaling": True,
            "websocket_frame_transport": False,
            "generated_mp4": False,
            "mediapipe": settings.use_mediapipe,
            "tensorrt": settings.mode.lower() == "trt",
        },
    }


@app.get("/metrics")
async def metrics(_: None = Depends(authorize)) -> dict[str, Any]:
    return {"runtime": adapter.health(), "sessions": sessions.metrics()}


@app.post("/avatar/session/start")
async def start_session(payload: SessionStartRequest, _: None = Depends(authorize)) -> dict[str, Any]:
    image_url = payload.avatar_image_url or settings.avatar_image_url
    if not image_url:
        raise HTTPException(status_code=400, detail="avatar_image_url or AVATAR_IMAGE_URL is required")
    ice_servers = payload.ice_servers or default_ice_servers()
    session = sessions.start(payload.session_id, image_url, ice_servers)
    return {
        "success": True,
        "session_id": session.session_id,
        "status": "started",
        "request_id": payload.request_id,
        "avatar_url": session.avatar_image_url,
        "signaling_url": f"/avatar/signaling/{session.session_id}",
        "ice_servers": ice_servers,
    }


@app.post("/avatar/session/stop")
async def stop_session(payload: SessionStopRequest, _: None = Depends(authorize)) -> dict[str, Any]:
    existed = await sessions.stop(payload.session_id)
    return {"success": True, "session_id": payload.session_id, "status": "stopped" if existed else "not_found"}


@app.post("/avatar/audio")
async def audio(payload: AudioRequest, _: None = Depends(authorize)) -> dict[str, Any]:
    session = sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    audio_bytes = base64.b64decode(payload.audio_base64)
    adapter.ingest_audio(payload.session_id, audio_bytes)
    session.audio_segments += 1
    session.last_audio_at = time.time()
    return {"success": True, "session_id": payload.session_id, "status": "accepted", "request_id": payload.request_id}


@app.websocket("/avatar/signaling/{session_id}")
async def signaling(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "session not found", "session_id": session_id})
        await websocket.close(code=4404)
        return

    pc = RTCPeerConnection(configuration=rtc_configuration(session.ice_servers))
    session.peer = pc
    pc.addTrack(AvatarVideoTrack(session_id, adapter, fps=settings.target_fps))
    frame_stream_task: asyncio.Task | None = None

    async def stream_frames_over_websocket(fps: int = 12) -> None:
        frame_interval = 1.0 / max(1, min(fps, settings.target_fps))
        frame_index = 0
        logger.info("avatar.websocket_frame_stream.started", extra={"session_id": session_id, "fps": fps})
        try:
            while True:
                frame = adapter.next_frame(session_id)
                ok, encoded = cv2.imencode(".jpg", frame[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    await websocket.send_text(json.dumps({
                        "type": "video-frame",
                        "format": "jpeg",
                        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "index": frame_index,
                        "timestamp": time.time(),
                    }))
                    if frame_index == 0 or frame_index % (fps * 5) == 0:
                        logger.info("avatar.websocket_frame_stream.frame_sent", extra={"session_id": session_id, "frame": frame_index})
                    frame_index += 1
                await asyncio.sleep(frame_interval)
        except asyncio.CancelledError:
            logger.info("avatar.websocket_frame_stream.cancelled", extra={"session_id": session_id, "frames": frame_index})
            raise
        except Exception as exc:
            logger.warning("avatar.websocket_frame_stream.error session_id=%s error_type=%s error=%s", session_id, type(exc).__name__, str(exc))

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(
            "avatar.webrtc.ice_connection_state",
            extra={"session_id": session_id, "state": pc.iceConnectionState},
        )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(
            "avatar.webrtc.peer_connection_state",
            extra={"session_id": session_id, "state": pc.connectionState},
        )

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate is not None:
            await websocket.send_json({"type": "ice-candidate", "candidate": candidate.to_json()})

    try:
        while True:
            payload = await websocket.receive_json()
            if payload.get("type") == "offer":
                logger.info("avatar.webrtc.offer_received", extra={"session_id": session_id})
                await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type="offer"))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})
                logger.info("avatar.webrtc.answer_sent", extra={"session_id": session_id})
            elif payload.get("type") == "ice-candidate":
                candidate = parse_candidate(payload)
                if candidate:
                    await pc.addIceCandidate(candidate)
            elif payload.get("type") == "start-frame-stream":
                if frame_stream_task is None or frame_stream_task.done():
                    requested_fps = int(payload.get("fps") or 12)
                    frame_stream_task = asyncio.create_task(stream_frames_over_websocket(requested_fps))
                await websocket.send_json({"type": "frame-stream-started", "session_id": session_id})
            elif payload.get("type") == "stop-frame-stream":
                if frame_stream_task and not frame_stream_task.done():
                    frame_stream_task.cancel()
                await websocket.send_json({"type": "frame-stream-stopped", "session_id": session_id})
            elif payload.get("type") == "ping":
                await websocket.send_json({"type": "pong", "session_id": session_id, "timestamp": time.time()})
            else:
                await websocket.send_text(json.dumps({"type": "ignored", "session_id": session_id}))
    except WebSocketDisconnect:
        return
    finally:
        if frame_stream_task and not frame_stream_task.done():
            frame_stream_task.cancel()
        await pc.close()
