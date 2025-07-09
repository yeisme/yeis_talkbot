from abc import ABC, abstractmethod
from typing import Any, Dict


class VAD(ABC):
    """
    通过 VAD 模型检测给定音频块中是否包含语音活动。
    """

    @abstractmethod
    def detect(self, audio: Any, **kwargs: Dict[str, Any]) -> bool:
        """
        通过 VAD 模型检测给定音频块中是否包含语音活动。

        :param audio: 要分析的音频数据，应为 bytes 类型。
        :param kwargs: 额外参数，如采样率、音频格式等，具体取决于 VAD 实现。
        :return: 如果检测到语音活动事件（如开始或结束），则为 True，否则为 False。
        """
        pass
