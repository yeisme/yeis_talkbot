from abc import ABC, abstractmethod
from typing import Any, List, Dict


class VAD(ABC):
    """
    通过 VAD 模型检测给定音频块中是否包含语音活动。
    """

    @abstractmethod
    def detect(self, audio: Any, **kwargs: Any) -> bool:
        """
        持续检测给定音频块中是否包含语音活动。

        :param audio: 要分析的音频数据，应为 bytes 类型。
        :param kwargs: 额外参数，如采样率、音频格式等，具体取决于 VAD 实现。
        :return: 如果检测到语音活动事件（如开始或结束），则为 True，否则为 False。
        """
        pass

    @abstractmethod
    def get_speech_segments(self, audio: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        获取音频中的所有语音段及其时间戳。
        :param audio: 音频数据
        :param kwargs: 其他参数
        :return: 语音段列表，每个元素是包含 'start' 和 'end' 的字典
        """
        pass
