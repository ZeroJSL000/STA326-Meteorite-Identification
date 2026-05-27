"""基于 timm 的 ConvNeXtV2 + GeM 单 logit 二分类模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import timm
except ImportError as error:
    raise ImportError("缺少 timm，请先执行 `uv add timm`。") from error

from config import ModelConfig


class GeM(nn.Module):
    """可学习广义均值池化，强调高响应主体区域。"""

    def __init__(self, p: float = 3.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError("GeM 初始 p 必须大于 0。")
        self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, features: Tensor) -> Tensor:
        p = self.p.clamp_min(self.eps)
        pooled = F.adaptive_avg_pool2d(
            features.clamp_min(self.eps).pow(p),
            output_size=1,
        )
        return pooled.pow(1.0 / p)


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """从预训练文件或训练 checkpoint 中提取参数字典。"""

    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件内容不是参数字典。")
    for key in ("state_dict", "model", "model_state_dict", "ema_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_weights_flexible(model: nn.Module, path: Path) -> tuple[int, list[str]]:
    """加载骨干参数，并跳过新增池化参数或分类头维度冲突。"""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    skipped = sorted(set(source_state) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    return len(compatible_state), skipped


def build_model(
    config: ModelConfig,
    pretrained_path: Path | None = None,
    require_pretrained: bool = False,
) -> nn.Module:
    """创建以 GeM 替换默认 GAP 的 ConvNeXtV2 二分类模型。"""

    model = timm.create_model(
        config.name,
        pretrained=False,
        num_classes=1,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
    )
    model.head.global_pool = GeM(p=config.gem_p)
    if pretrained_path is not None and pretrained_path.is_file():
        loaded_count, skipped = load_weights_flexible(model, pretrained_path)
        print(
            f"已载入预训练参数 {loaded_count} 项: {pretrained_path}; "
            f"跳过 {len(skipped)} 项不兼容参数。"
        )
    elif require_pretrained:
        raise FileNotFoundError(
            f"缺少预训练参数文件: {pretrained_path}，请先按 README.md 下载到 weights/。"
        )
    return model
