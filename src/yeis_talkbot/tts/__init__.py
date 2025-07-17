from .tts_handler import register_tts_handler, unregister_tts_handler
from .edge_tts import EdgeTTS, EdgeTTSHandler
from .abc import TTS

__all__ = [
    "TTS",
    "EdgeTTS",
    "EdgeTTSHandler",
    "register_tts_handler",
    "unregister_tts_handler",
]
