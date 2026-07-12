FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FASTERLIVEPORTRAIT_ROOT=/opt/FasterLivePortrait \
    CHECKPOINT_DIR=/models/FasterLivePortrait/checkpoints \
    AVATAR_ENGINE_MODE=trt \
    AVATAR_ENGINE_USE_MEDIAPIPE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg curl ca-certificates build-essential \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install -r /app/requirements.txt

RUN git clone --depth 1 https://github.com/warmshao/FasterLivePortrait.git /opt/FasterLivePortrait

COPY avatar_engine /app/avatar_engine
COPY scripts /app/scripts
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh /app/scripts/*.sh

EXPOSE 8000
ENTRYPOINT ["/app/startup.sh"]
