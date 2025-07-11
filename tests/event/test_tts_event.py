import pytest
from yeis_talkbot.tts import (
    EdgeTTS,
    register_tts_handler,
    unregister_tts_handler,
    EdgeTTSHandler,
)
from yeis_talkbot.configs import AppConfig
from yeis_talkbot.event.event import TTSEvent
from yeis_talkbot.event.bus import event_bus
import os


@pytest.mark.asyncio
async def test_register_edge_tts_handler():
    app_config = AppConfig.from_yaml("configs/config.yaml")
    tts = EdgeTTS(app_config=app_config)
    handle = register_tts_handler(tts)

    assert handle is not None
    assert isinstance(handle, EdgeTTSHandler)

    text = "测试事件驱动 TTS"
    event = TTSEvent(text=text)
    await event_bus.publish(event=event)

    assert event.status == "completed"
    assert os.path.exists(event.audio_path)
    os.remove(path=event.audio_path)
    unregister_tts_handler(handle)


@pytest.mark.asyncio
async def test_batch_tts_events():
    app_config = AppConfig.from_yaml("configs/config.yaml")
    tts = EdgeTTS(app_config=app_config)
    handle = register_tts_handler(tts)

    assert handle is not None
    assert isinstance(handle, EdgeTTSHandler)

    texts = ["你好，世界", "第二条语音", "第三条语音"]
    events = [TTSEvent(text=text) for text in texts]

    delete_paths: list[str] = []

    for event in events:
        await event_bus.publish(event=event)
        assert event.status == "completed"
        assert os.path.exists(event.audio_path)
        delete_paths.append(event.audio_path)

    for path in delete_paths:
        try:
            os.remove(path=path)
        except Exception:
            pass

    unregister_tts_handler(handle)
