from typing import Union
import numpy as np
import numpy.typing as npt
import torch

audio_type = Union[bytes, torch.Tensor, npt.NDArray[np.float32], npt.NDArray[np.int16]]
""" 
| 数据类型                    | 适用场景            | 优点        | 缺点            |
|-------------------------|-----------------|-----------|---------------|
| npt.NDArray[np.float32] | 已归一化的音频(-1~1范围) | 兼容大多数ML框架 | 需要转换原始数据      |
| npt.NDArray[np.int16]   | 原始PCM/WAV数据     | 保持原始精度    | 需要手动归一化       |
| torch.Tensor            | PyTorch模型处理     | GPU加速支持   | 非PyTorch环境不适用 |
| bytes                   | 原始音频文件/流        | 无需解析      | 需要解码才能处理      |
"""
