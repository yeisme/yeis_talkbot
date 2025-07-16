from ..event import ASRHandler
from .FunASR import (
    FunASR,
    FunASRHandler,
    register_FunASR_handler,
    unregister_FunASR_handler,
)
from .abc import ASR


def register_asr_handler(asr: ASR) -> ASRHandler | None:
    """
    注册所有 ASR 事件处理器

    Args:
        asr (ASR): ASR 实例

    Returns:
        ASRHandler | None: 注册的 ASR 事件处理器，如果不支持则返回 None
    """
    if isinstance(asr, FunASR):
        handler: ASRHandler = register_FunASR_handler(asr)
        return handler
    return None


def unregister_asr_handler(handler: ASRHandler) -> None:
    """
    取消注册 ASR 事件处理器

    Args:
        handler (ASRHandler): 要取消注册的 ASR 事件处理器
    """
    if isinstance(handler, FunASRHandler):
        unregister_FunASR_handler(handler)
