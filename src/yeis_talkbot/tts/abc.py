from abc import ABC, abstractmethod
from typing import Any, Dict


class TTS(ABC):
    """
    Abstract base class for Text-to-Speech (TTS) systems.
    """

    @abstractmethod
    def synthesize(self, text: str, **kwargs: Dict[str, Any]) -> Any:
        """
        Synthesize text to audio.

        :param text: Text to be synthesized.
        :param kwargs: Additional parameters for synthesis.
        :return: Synthesized audio output.
        """
        pass
