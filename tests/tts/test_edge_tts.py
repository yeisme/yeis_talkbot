import pytest
from src.yeis_talkbot.tts.edge_tts import EdgeTTS
from src.yeis_talkbot.configs import AppConfig
import os


@pytest.mark.asyncio
async def test_edge_tts_synthesize():
    app_config = AppConfig.from_yaml("configs/config.yaml")
    tts = EdgeTTS(app_config=app_config)
    text = "你好，世界"
    audio_output = await tts.synthesize(text)
    assert audio_output is not None
    os.remove(audio_output)  # Clean up the generated audio file
