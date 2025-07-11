from .tts_handler import register_tts_handler, unregister_tts_handler
from .edge_tts import EdgeTTS, EdgeTTSHandler

__all__ = [
    "register_tts_handler",
    "EdgeTTS",
    "unregister_tts_handler",
    "EdgeTTSHandler",
]
