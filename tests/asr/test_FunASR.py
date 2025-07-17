import pytest
import numpy as np
import soundfile as sf  # type: ignore
import numpy.typing as npt
from typing import List

# 确保导入路径正确，这里假设 AppConfig 和 FunASRStreaming 的路径如下
from src.yeis_talkbot.configs import AppConfig
from src.yeis_talkbot.asr import FunASR

# 使用 Pytest 的类型别名，让测试函数更简洁
ASRTestFixture = FunASR


@pytest.fixture(scope="module")
def app_config() -> AppConfig:
    """从 YAML 文件加载应用配置。"""
    # 确保配置文件路径相对于项目根目录是正确的
    app_config_instance = AppConfig.from_yaml("configs/config.yaml")
    return app_config_instance


@pytest.fixture(scope="module")
def asr(app_config: AppConfig) -> ASRTestFixture:
    """初始化 FunASRStreaming 实例，供所有测试使用。"""
    return FunASR(app_config=app_config)


def test_transcribe_silence(asr: ASRTestFixture) -> None:
    """
    测试1: 验证静音音频是否返回空字符串。
    """
    # Arrange: 创建一个 1 秒的静音音频块
    silent_chunk: npt.NDArray[np.float32] = np.zeros(16000, dtype=np.float32)
    asr.reset()  # 确保状态是干净的

    # Act: 对静音块进行转录
    result: str = asr.transcribe(chunk=silent_chunk, is_final=True)

    # Assert: 结果应为空字符串
    assert result == ""


def test_transcribe_audio_streaming(asr: ASRTestFixture) -> None:
    """
    测试2: 验证对真实音频的流式识别结果是否正确。
    """
    audio_path = "tests/audio/test_16k.wav"

    expected_text = "你好欢迎使用语音服务"

    data: npt.NDArray[np.int16]
    samplerate: int
    data, samplerate = sf.read(audio_path, dtype="int16")  # type: ignore

    assert samplerate == 16000
    asr.reset()

    # Act: 模拟流式处理
    chunk_size = 9600
    full_text: List[str] = []
    num_samples = len(data)

    for i in range(0, num_samples, chunk_size):
        chunk: npt.NDArray[np.int16] = data[i : i + chunk_size]
        is_final = (i + chunk_size) >= num_samples

        text_piece: str = asr.transcribe(chunk=chunk, is_final=is_final)
        if text_piece:
            full_text.append(text_piece)

    result: str = "".join(full_text)

    # Assert: 验证最终识别结果是否符合预期
    print(f"识别结果: {result}")
    assert expected_text in result


def test_reset_functionality(asr: ASRTestFixture) -> None:
    """
    测试3: 验证 reset() 方法是否能有效清除缓存，开始新的会话。
    """
    # Arrange: 加载音频并分成两半
    data: npt.NDArray[np.int16]
    data, _ = sf.read("tests/audio/test_16k.wav", dtype="int16")  # type: ignore

    mid_point: int = len(data) // 2
    first_half: npt.NDArray[np.int16] = data[:mid_point]
    second_half: npt.NDArray[np.int16] = data[mid_point:]

    # 假设 test_16k.wav 后半段的核心内容是 "语音服务"
    expected_text_of_second_half = "语音服务"
    # 假设前半段的核心内容是 "你好欢迎使用"
    unexpected_text_from_first_half = "你好"

    # Act Part 1: 处理第一段音频，但不结束会话
    asr.reset()
    _ = asr.transcribe(chunk=first_half, is_final=False)

    # Act Part 2: 重置状态，然后只处理第二段音频
    asr.reset()
    result_after_reset: str = asr.transcribe(chunk=second_half, is_final=True)

    # Assert: 验证重置后的结果只包含第二段的内容，不受第一段的影响
    print(f"重置后识别结果: {result_after_reset}")
    assert expected_text_of_second_half in result_after_reset
    assert unexpected_text_from_first_half not in result_after_reset
