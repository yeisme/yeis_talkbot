from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict
import yaml


class TTSConfig(BaseModel):
    index_tts: Dict[str, str] = Field({"config": "checkpoints/checkpoints-config.yaml"})
    edge_tts: Dict[str, str] = Field({"config": "config/edge-tts.yaml"})
    out_path: str = Field(
        default="tmp/tts/", description="Output path for TTS audio files"
    )


class ASRConfig(BaseModel):
    FunASR: Dict[str, str] = Field(
        default={
            "model": "paraformer-zh-streaming",
        },
        description="Configuration for FunASR ASR model(streaming enabled)",
    )


class VADConfig(BaseModel):
    FunASR: Dict[str, str] = Field(
        default={
            "model": "fsmn-vad",
        },
        description="Configuration for FunASR VAD model",
    )


class LLMConfig(BaseModel):
    model: str = Field("gpt4o", description="The model to use for LLM")
    base_url: str = Field(
        default="https://api.openai.com/v1/chat/completions",
        description="Base URL for the LLM API",
    )
    temperature: float = Field(0.7, description="Temperature for LLM responses")
    max_tokens: int = Field(..., description="Maximum tokens for LLM responses")
    streaming: bool = Field(False, description="Enable streaming for LLM responses")
    timeout: Optional[int | None] = Field(None, description="Timeout for LLM requests")


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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(stream=f)
        return cls(**config_data)
