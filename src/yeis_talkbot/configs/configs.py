from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional, Dict
import yaml


class TTSConfig(BaseModel):
    index_tts: Dict[str, str] = Field({"config": "checkpoints/checkpoints-config.yaml"})
    edge_tts: Dict[str, str] = Field({"config": "config/edge-tts.yaml"})


class ASRConfig(BaseModel):
    FunASR: Dict[str, str] = Field(
        {
            "model": "funasr/whisper-zh-cn-base",
            "output_dir": "tmp/",
        },
        description="Configuration for FunASR ASR model",
    )


class VADConfig(BaseModel):
    SileroVAD: Dict[str, float] = Field(
        {
            "sampling_rate": 16000,
            "threshold": 0.5,
            "min_silence_duration_ms": 200,
        },
        description="Configuration for Silero VAD",
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(stream=f)
        return cls(**config_data)
