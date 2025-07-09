from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Optional, Dict
import yaml


class TTSConfig(BaseModel):
    index_tts: Dict[str, str]
    edge_tts: Dict[str, str]


class ASRConfig(BaseModel):
    FunASR: Dict[str, str]


class VADConfig(BaseModel):
    SileroVAD: Dict[str, float]


class LLMConfig(BaseModel):
    model: str
    base_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int
    streaming: bool
    timeout: Optional[int | None] = None


class AppConfig(BaseSettings):
    """
    example usage:
    ==============
    config = AppConfig.from_yaml(yaml_file=str(Path("configs/config.yaml")))

    """

    OPENAI_API_KEY: str

    TTS: TTSConfig
    ASR: ASRConfig
    VAD: VADConfig
    LLM: LLMConfig

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(stream=f)
        return cls(**config_data)
