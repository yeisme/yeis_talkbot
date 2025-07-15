from .abc import ASR
from funasr import AutoModel  # type: ignore
import torch
from ..configs import AppConfig


class FunASR(ASR):
    """FunASR 实现语音识别功能"""

    def __init__(self, app_config: AppConfig) -> None:
        self.model = AutoModel(
            model=app_config.ASR.FunASR["model"],
            vad_kwargs={"max_single_segment_time": 300},
            disable_update=True,
            hub="hf",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
