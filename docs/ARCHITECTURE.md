# Arevik AvatarEngine Architecture

The engine wraps FasterLivePortrait with a standards-compliant WebRTC service.

## Transport

- WebSocket: signaling only.
- WebRTC: all avatar media.
- HTTP: session lifecycle, audio ingestion, health, metrics.

No rendered avatar frames are transported over WebSocket.
No MP4 files are created in the runtime media path.

## Upstream limitations

FasterLivePortrait supports realtime webcam mode and API ZIP/MP4 generation, but does not ship a browser WebRTC service. Arevik_AvatarEngine implements that service around isolated adapter interfaces.

The current adapter keeps model initialization and frame production isolated. Production audio-motion integration should be completed inside `FasterLivePortraitAdapter` using upstream JoyVASA/audio-driving internals while keeping the external API stable.
