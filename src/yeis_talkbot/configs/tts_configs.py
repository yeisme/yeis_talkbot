from pydantic import BaseModel, Field
import yaml


class EdgeTTSConfig(BaseModel):
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="Voice for Edge TTS")
    rate: str = Field(default="+0%", description="Speech rate for Edge TTS")
    volume: str = Field(default="+0%", description="Volume for Edge TTS")

    @classmethod
    def from_yaml(cls, yaml_file: str):
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(stream=f)
        return cls(**config_data)
