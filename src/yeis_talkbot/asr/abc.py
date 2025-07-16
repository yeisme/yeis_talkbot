from abc import ABC, abstractmethod
from typing import Optional
from ..types import audio_type


class ASR(ABC):
    """
    Abstract base class for Automatic Speech Recognition (ASR) systems.

    ASR 负责语音转文本，输入可以是完整音频或分割后的语音片段，子类需要实现具体的语音识别逻辑。
    """

    @abstractmethod
    def transcribe(self, chunk: Optional[audio_type], is_final: bool = False) -> str:
        """
        Transcribe audio input to text.

        :param chunk: Audio input to be transcribed.
        :param is_final: Whether this is the final chunk of audio.
        :return: Transcribed text.
        """
        pass
