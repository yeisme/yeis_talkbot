from abc import ABC, abstractmethod
from typing import Any, Dict


class ASR(ABC):
    """
    Abstract base class for Automatic Speech Recognition (ASR) systems.
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
