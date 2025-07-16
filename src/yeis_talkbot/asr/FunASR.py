import logging
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import numpy.typing as npt
from funasr import AutoModel  # type: ignore

from ..configs import AppConfig
from ..types import audio_type
from .abc import ASR
from ..event import BaseEvent, event_bus, ASREvent, ASRHandler

logger = logging.getLogger(__name__)

# 默认流式参数
DEFAULT_STREAMING_CHUNK_SIZE = [0, 10, 5]


class FunASR(ASR):
    """
    使用 FunASR 实现流式语音识别功能。

    这个类是**有状态的**，它会管理一个内部的缓存（cache）来处理连续的音频流。
    正确的处理流程:
    1. 调用 `reset()` 开始一个新的识别会话。
    2. 循环调用 `transcribe(chunk)` 来处理中间的音频块。
    3. 在音频流结束时，调用 `transcribe(last_chunk, is_final=True)` 来获取最后的结果。
    """

    def __init__(self, app_config: AppConfig) -> None:
        """
        初始化流式ASR模型和配置。

        Args:
            app_config (AppConfig): 包含模型路径和参数的应用程序配置。
        """
        logger.info("Initializing FunASR streaming model...")

        try:
            asr_model_path = app_config.ASR.FunASR["model"]
            vad_model_path = app_config.VAD.FunASR["model"]
            self.chunk_size = app_config.ASR.FunASR.get(
                "chunk_size", DEFAULT_STREAMING_CHUNK_SIZE
            )
        except (AttributeError, KeyError) as e:
            logger.error(f"Configuration missing for FunASR streaming model: {e}")
            raise ValueError(f"Configuration missing for FunASR: {e}") from e

        self.cache: Dict[str, Any] = {}

        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = AutoModel(
                model=asr_model_path,
                vad_model=vad_model_path,
                device=device,
                hub="hf",  # ModelScope为"ms", HuggingFace为"hf"
            )
            logger.info(f"FunASR streaming model loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load FunASR model: {e}", exc_info=True)
            raise

    def _normalize_chunk(self, chunk: audio_type) -> npt.NDArray[np.float32]:
        """
        内部辅助函数，将输入的Numpy音频块归一化为np.float32格式。
        """
        if chunk.dtype == np.float32:
            return chunk.astype(np.float32)
        if chunk.dtype == np.int16:
            # 将16-bit整数归一化到float32
            return chunk.astype(np.float32) / 32768.0

        raise TypeError(
            f"Unsupported numpy array dtype for normalization: {chunk.dtype}"
        )

    def reset(self) -> None:
        """
        重置内部状态缓存，开始一个新的语音识别会话。
        在处理每段新的独立语音前都应调用此方法。
        """
        logger.debug("Resetting ASR streaming cache.")
        self.cache = {}

    def transcribe(self, chunk: Optional[audio_type], is_final: bool = False) -> str:
        """
        对单个Numpy音频块（chunk）进行流式转录。
        """
        if chunk is None:
            if not is_final:
                return ""
            normalized_chunk = np.zeros(0, dtype=np.float32)
        else:
            try:
                normalized_chunk = self._normalize_chunk(chunk)
            except TypeError as e:
                logger.error(e)
                return ""

        try:
            res: List[Dict[str, Any]] = self.model.generate(  # type: ignore
                input=normalized_chunk,
                cache=self.cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                use_itn=True,
            )
            return res[0].get("text", "") if res else ""  # type: ignore
        except Exception as e:
            logger.error(
                f"An error occurred during chunk transcription: {e}", exc_info=True
            )
            self.reset()
            return ""


class FunASRHandler(ASRHandler):
    """
    FunASR 事件处理器

    处理 ASREvent 事件，使用 FunASR 模型进行语音识别
    """

    def __init__(self, asr: ASR) -> None:
        """
        初始化 FunASR 事件处理器

        Args:
            asr (ASR): FunASR 实例
        """
        super().__init__(asr)
        self.asr = asr
        logger.info("FunASR 事件处理器初始化完成")

    async def handle_event(self, event: BaseEvent) -> None:
        """
        处理 ASR 事件

        Args:
            event (BaseEvent): 要处理的事件，应该是 ASREvent 类型
        """
        if not isinstance(event, ASREvent):
            logger.error("事件类型错误，必须是 ASREvent")
            return

        try:
            event.status = "processing"
            logger.info(f"开始处理 ASR 事件: {event.event_id}")

            # 检查是否提供了音频路径
            if not event.audio_path:
                logger.error("音频路径为空")
                event.status = "failed"
                return

            self.asr.reset()

            text = self.asr.transcribe(chunk=None, is_final=True)

            # 如果没有识别到文本，可能需要特殊处理
            if not text:
                text = ""
                logger.warning(f"ASR 未识别到文本: {event.event_id}")

            event.text = text
            event.status = "completed"
            logger.info(f"ASR 识别完成: {event.event_id}, 识别结果: {text}")

        except Exception as e:
            logger.error(f"ASR 事件处理失败: {event.event_id}, 错误: {str(e)}")
            event.status = "failed"


def register_FunASR_handler(asr: ASR) -> FunASRHandler:
    """
    注册 ASR 事件处理器到事件总线

    Args:
        asr (ASR): ASR 实例

    Returns:
        FunASRHandler: 注册的处理器实例
    """
    handler = FunASRHandler(asr)
    event_bus.subscribe(ASREvent, handler.handle_event)
    logger.info("ASR 事件处理器已注册")
    return handler


def unregister_FunASR_handler(handler: FunASRHandler) -> None:
    """
    从事件总线取消注册 ASR 事件处理器

    Args:
        handler (FunASRHandler): 要取消注册的处理器
    """
    event_bus.unsubscribe(ASREvent, handler.handle_event)
    logger.info("ASR 事件处理器已取消注册")
