from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = "Arevik AvatarEngine"
    api_key: str = Field(default="", validation_alias="AVATAR_ENGINE_API_KEY")
    avatar_image_url: str = Field(default="", validation_alias="AVATAR_IMAGE_URL")
    fasterliveportrait_root: str = "/opt/FasterLivePortrait"
    checkpoint_dir: str = "/models/FasterLivePortrait/checkpoints"
    mode: str = Field(default="trt", validation_alias="AVATAR_ENGINE_MODE")
    use_mediapipe: bool = Field(default=True, validation_alias="AVATAR_ENGINE_USE_MEDIAPIPE")
    cfg_path: str = Field(default="/opt/FasterLivePortrait/configs/trt_mp_infer.yaml", validation_alias="AVATAR_ENGINE_CFG")
    stun_url: str = Field(default="stun:stun.l.google.com:19302", validation_alias="STUN_URL")
    turn_url: str = Field(default="", validation_alias="TURN_URL")
    turn_username: str = Field(default="", validation_alias="TURN_USERNAME")
    turn_password: str = Field(default="", validation_alias="TURN_PASSWORD")
    target_fps: int = 25
    session_timeout_seconds: int = 3600

    model_config = {"extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
