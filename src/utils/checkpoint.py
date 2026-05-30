"""模型权重与 Checkpoint 解析工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """从预训练文件或训练 checkpoint 中提取参数字典。"""
    
    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件内容不是参数字典。")
    for key in (
        "state_dict_ema",
        "state_dict",
        "model",
        "model_state_dict",
        "ema_state_dict",
    ):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_weights_flexible(model: nn.Module, path: Path) -> tuple[int, list[str]]:
    """加载预训练参数，并跳过分类头等维度冲突的层。"""
    
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()
    
    compatible_state = {}
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            compatible_state[key] = value
            
    skipped = sorted(set(source_state) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), skipped