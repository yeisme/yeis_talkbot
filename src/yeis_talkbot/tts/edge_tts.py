from ..event import BaseEvent, event_bus, TTSEvent, TTSHandler
from .abc import TTS

import logging
from typing import Any
import os
import time
import uuid


logger = logging.getLogger(__name__)


class EdgeTTS(TTS):
    """
    Edge TTS implementation of the TTS abstract base class.
    """

    from ..configs.tts_configs import EdgeTTSConfig
    from ..configs.configs import AppConfig
    import edge_tts

    def __init__(self, app_config: AppConfig) -> None:
        """
        Initialize the EdgeTTS instance with configuration settings.
        """
        self.config = self.EdgeTTSConfig.from_yaml("configs/edge-tts.yaml")
        self.voice = self.config.voice
        self.rate = self.config.rate
        self.volume = self.config.volume
        self.output_path = app_config.TTS.out_path
        if self.output_path == "":
            self.output_path = "tmp/tts/"
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path)

    async def synthesize(self, text: str, **kwargs: str) -> Any:
        """
        Synthesize text to audio using Edge TTS.

        :param text: Text to be synthesized.
        :param kwargs: Additional parameters for synthesis.
        :return: Synthesized audio output.
        """

        logger.info(
            f"合成音频: {text} voice: {self.voice}, rate: {self.rate}, volume: {self.volume}"
        )

        # Create an Edge TTS client
        client = self.edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        audio_output = f"{self.output_path}output_edgetts_{timestamp}_{unique_id}.wav"

        try:
            await client.save(audio_output)
            logger.info(f"音频合成成功，保存到: {audio_output}")
        except Exception as e:
            logger.error(f"合成音频失败: {e}")
            return None

        return audio_output


class EdgeTTSHandler(TTSHandler):
    """
    Edge TTS 事件处理器

    初始化需要 EdgeTTS 实例
    """

    def __init__(self, tts: EdgeTTS) -> None:
        self.tts = tts

    async def handle_event(self, event: BaseEvent) -> None:
        # 类型检查和转换
        if not isinstance(event, TTSEvent):
            logger.error("事件类型错误，必须是 TTSEvent")
            return
        try:
            event.status = "processing"
            audio_path = await self.tts.synthesize(event.text)
            if audio_path:
                event.audio_path = audio_path
                event.status = "completed"
            else:
                event.status = "failed"
        except Exception as e:
            logger.error(f"处理 TTS 事件失败: {e}")
            event.status = "failed"


def register_edge_tts_handler(edge_tts: EdgeTTS) -> TTSHandler:
    handler = EdgeTTSHandler(edge_tts)
    event_bus.subscribe(TTSEvent, handler.handle_event)
    return handler


def unregister_edge_tts_handler(handler: EdgeTTSHandler):
    """
    Unregister the Edge TTS handler from the event bus.
    """
    event_bus.unsubscribe(TTSEvent, handler.handle_event)
