import torch
import io
import soundfile as sf  # type: ignore
from silero_vad import load_silero_vad, get_speech_timestamps, utils_vad  # type: ignore
import numpy as np
from typing import Any, List, Union, Optional, Dict, Type
import logging

# 假设 VAD 抽象基类已定义
from .abc import VAD

logger = logging.getLogger(__name__)


audio_type = Union[bytes, torch.Tensor, np.ndarray]


class SileroVAD(VAD):
    """
    Silero VAD 实现，使用 Silero VAD 模型进行语音活动检测。

    Attributes:
        sample_rate (int): 模型期望的音频采样率。
        device (str): 运行推理的设备 ('cpu' 或 'cuda')。
        model (utils_vad.OnnxWrapper): 已加载的 Silero VAD 模型。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_threads: int = 1,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Silero VAD 模型。

        :param sample_rate: 音频采样率 (Hz)，默认为 16000。
        :param num_threads: PyTorch 使用的线程数，默认为 1。
        :param device: 运行设备 ("cpu" 或 "cuda")，如果为 None，则自动检测。
        :param kwargs: 传递给 `load_silero_vad` 的其他参数 (例如 onnx=True)。
        :raises RuntimeError: 如果模型加载失败。
        """
        try:
            torch.set_num_threads(num_threads)
            self.sample_rate = sample_rate
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Initializing SileroVAD on device: {self.device}")

            # 直接加载模型，无需绑定不必要的函数
            onnx: bool = kwargs.pop("onnx", False)
            self.model: utils_vad.OnnxWrapper = load_silero_vad(onnx=onnx)
            self.model.to(self.device)  # type: ignore

        except Exception as e:
            raise RuntimeError(f"Failed to initialize SileroVAD: {e}") from e

    def _prepare_audio(self, audio: audio_type) -> torch.Tensor:
        """
        准备音频数据为模型可处理的格式 (单声道浮点 Tensor)。

        :param audio: 输入音频 (audio_type)。
        :return: 移至正确设备并处理后的音频 Tensor。
        :raises TypeError: 如果输入了不支持的音频类型。
        :raises ValueError: 如果音频处理失败或采样率不匹配。
        """
        if isinstance(audio, bytes):
            try:
                with io.BytesIO(audio) as bio:
                    # 指定 dtype 为 float32，避免类型转换问题
                    data, samplerate = sf.read(bio, dtype="float32")  # type: ignore

                if samplerate != self.sample_rate:
                    raise ValueError(
                        f"Sample rate mismatch: got {samplerate}, "
                        f"expected {self.sample_rate}"
                    )
                wav = torch.from_numpy(data)  # type: ignore
            except Exception as e:
                raise ValueError(f"Failed to process audio bytes: {e}") from e
        elif isinstance(audio, torch.Tensor):
            wav = audio.float()
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).float()  # type: ignore
        else:
            raise TypeError(f"Unsupported audio type: {type(audio).__name__}")

        if wav.ndim > 1:
            wav = wav.mean(dim=-1)

        return wav.to(device=self.device)

    def get_speech_segments(
        self,
        audio: audio_type,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        return_seconds: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Union[int, float]]]:
        """
        获取音频中的所有语音段及其时间戳。这是核心的 VAD 功能方法。

        :param audio: 输入音频 (audio_type)。
        :param threshold: 语音检测阈值 (0-1)，默认为 0.5。
        :param min_speech_duration_ms: 最小语音持续时间 (毫秒)，默认为 250。
        :param max_speech_duration_s: 最大语音持续时间 (秒)，默认为无限。
        :param min_silence_duration_ms: 最小静音持续时间 (毫秒)，默认为 100。
        :param window_size_samples: 窗口大小 (样本数)，默认为 512。
        :param speech_pad_ms: 语音填充时间 (毫秒)，默认为 30。
        :param return_seconds: 是否返回秒为单位的时间戳，默认为 True。
        :param kwargs: 其他传递给 `get_speech_timestamps` 的参数。
        :return: 语音段列表，每个元素是包含 'start' 和 'end' 的字典。
        :raises ValueError: 如果音频处理或检测失败。
        """
        logger.info("Getting speech segments from audio.")

        try:
            wav: torch.Tensor = self._prepare_audio(audio=audio)

            # 直接调用导入的函数，并将所有 VAD 参数传递下去
            speech_timestamps: List[Any] = get_speech_timestamps(  # type: ignore
                audio=wav,
                model=self.model,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                max_speech_duration_s=max_speech_duration_s,
                min_silence_duration_ms=min_silence_duration_ms,
                window_size_samples=window_size_samples,
                speech_pad_ms=speech_pad_ms,
                sampling_rate=self.sample_rate,  # 明确传递采样率
                return_seconds=return_seconds,
                **kwargs,
            )
            return speech_timestamps

        except Exception as e:
            logger.error(f"Failed to get speech segments: {e}", exc_info=True)
            raise ValueError(f"Failed to get speech segments: {e}") from e

    def detect(
        self,
        audio: audio_type,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        **kwargs: Any,
    ) -> bool:
        """
        检测给定音频中是否包含语音活动。

        :param audio: 输入音频 (audio_type)。
        :param ...: (参数与 get_speech_segments 相同)。
        :return: 如果检测到语音则返回 True，否则返回 False。
        :raises ValueError: 如果音频处理或检测失败。
        """
        logger.info("Detecting speech in audio.")

        speech_timestamps: List[Dict[str, int | float]] = self.get_speech_segments(
            audio=audio,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
            **kwargs,
        )

        logger.debug(f"Detected {len(speech_timestamps)} speech segments.")
        return len(speech_timestamps) > 0

    def cleanup(self) -> None:
        """执行所有必要的资源清理。"""
        logger.info("Cleaning up SileroVAD resources.")
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "SileroVAD":
        """支持上下文管理器协议。"""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """在退出上下文时调用清理方法，确保资源被释放。"""
        self.cleanup()
