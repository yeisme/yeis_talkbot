from ..event import TTSHandler
from .edge_tts import (
    EdgeTTS,
    EdgeTTSHandler,
    register_edge_tts_handler,
    unregister_edge_tts_handler,
)
from .abc import TTS


def register_tts_handler(TTS: TTS) -> TTSHandler | None:
    """
    注册所有 TTS 事件处理器
    """
    if isinstance(TTS, EdgeTTS):
        handle: TTSHandler = register_edge_tts_handler(edge_tts=TTS)
        return handle
    return None


def unregister_tts_handler(TTSHandler: TTSHandler):
    if isinstance(TTSHandler, EdgeTTSHandler):
        unregister_edge_tts_handler(handler=TTSHandler)
