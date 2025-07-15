from abc import ABC, abstractmethod
from typing import Any, Dict


class ASR(ABC):
    """
    Abstract base class for Automatic Speech Recognition (ASR) systems.

    ASR 负责语音转文本，输入可以是完整音频或分割后的语音片段，子类需要实现具体的语音识别逻辑。
    """

    @abstractmethod
    def transcribe(self, audio: Any, **kwargs: Dict[str, Any]) -> str:
        """
        Transcribe audio input to text.

        :param audio: Audio input to be transcribed.
        :param kwargs: Additional parameters for transcription.
        :return: Transcribed text.
        """
        pass
