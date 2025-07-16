import pytest
import os
from src.yeis_talkbot.asr import (
    FunASR,
    register_asr_handler,
    unregister_asr_handler,
    FunASRHandler,
)
from src.yeis_talkbot.configs import AppConfig
from src.yeis_talkbot.event.event import ASREvent
from src.yeis_talkbot.event.bus import event_bus


@pytest.mark.asyncio
async def test_register_asr_handler():
    """测试注册 ASR 事件处理器"""
    app_config = AppConfig.from_yaml("configs/config.yaml")
    asr = FunASR(app_config=app_config)
    handler = register_asr_handler(asr)

    assert handler is not None
    assert isinstance(handler, FunASRHandler)

    audio_path = "tests/audio/test_16k.wav"
    event = ASREvent(audio_path=audio_path)

    await event_bus.publish(event=event)
    assert event.status in ["completed", "failed"]
    if event.status == "completed":
        assert isinstance(event.text, str)
        print(f"识别结果: {event.text}")

    unregister_asr_handler(handler)


@pytest.mark.asyncio
async def test_batch_asr_events():
    """测试批量 ASR 事件处理"""
    app_config = AppConfig.from_yaml("configs/config.yaml")
    asr = FunASR(app_config=app_config)
    handler = register_asr_handler(asr)

    assert handler is not None
    assert isinstance(handler, FunASRHandler)

    audio_files = ["tests/audio/test_16k.wav", "tests/audio/test.wav"]
    events: list[ASREvent] = []

    for audio_file in audio_files:
        if os.path.exists(path=audio_file):
            events.append(ASREvent(audio_path=audio_file))

    # 处理每个事件
    for event in events:
        await event_bus.publish(event=event)
        assert event.status in ["completed", "failed"]

        if event.status == "completed":
            assert isinstance(event.text, str)
            print(f"文件 {event.audio_path} 识别结果: {event.text}")

    unregister_asr_handler(handler)


@pytest.mark.asyncio
async def test_asr_event_with_invalid_audio():
    """测试无效音频路径的 ASR 事件"""
    app_config = AppConfig.from_yaml("configs/config.yaml")
    asr = FunASR(app_config=app_config)
    handler = register_asr_handler(asr)

    event = ASREvent(audio_path="nonexistent_file.wav")

    assert handler is not None
    assert isinstance(handler, FunASRHandler)

    await event_bus.publish(event=event)
    assert event.status == "failed"
    assert event.text == ""

    unregister_asr_handler(handler)


@pytest.mark.asyncio
async def test_asr_event_with_empty_audio_path():
    """测试空音频路径的 ASR 事件"""
    app_config = AppConfig.from_yaml("configs/config.yaml")
    asr = FunASR(app_config=app_config)
    handler = register_asr_handler(asr)

    event = ASREvent(audio_path="")

    assert handler is not None
    assert isinstance(handler, FunASRHandler)

    await event_bus.publish(event=event)
    assert event.status == "failed"
    assert event.text == ""

    unregister_asr_handler(handler)
