from .abc import TTS
import logging
from typing import Any
import os
import time
import uuid

logger = logging.getLogger(__name__)


class EdgeTTS(TTS):
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

    """
    Edge TTS implementation of the TTS abstract base class.
    """

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
