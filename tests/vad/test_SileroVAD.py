import pytest
import torch
import soundfile as sf  # type: ignore

from src.yeis_talkbot.vad.SileroVAD import SileroVAD


@pytest.fixture
def vad():
    return SileroVAD(sample_rate=16000)


def test_detect_no_speech(vad: SileroVAD):
    audio = torch.zeros(16000)  # 静音
    result = vad.detect(audio)
    assert result is False


def test_detect_speech(vad: SileroVAD):
    data, samplerate = sf.read("tests/audio/test_16k.wav")  # type: ignore

    result = vad.detect(data, 0.5)  # type: ignore
    assert result is True
