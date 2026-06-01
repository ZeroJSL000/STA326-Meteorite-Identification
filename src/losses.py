"""配置驱动的二分类损失函数。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """数值稳定的二分类 Focal Loss，兼容动态正类权重。"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("focal_gamma 必须大于等于 0。")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("focal_alpha 必须在 [0, 1] 范围内。")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype)
        probabilities = torch.sigmoid(logits)
        log_probabilities = F.logsigmoid(logits)
        log_negative_probabilities = F.logsigmoid(-logits)
        log_pt = targets * log_probabilities + (1.0 - targets) * log_negative_probabilities
        pt = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1.0 - targets)
            loss = loss * class_weight
        return loss.mean()


class LogitAdjustedLoss(nn.Module):
    """在底层损失之前应用可选的二分类训练先验偏移。"""

    def __init__(self, loss: nn.Module, train_pos_prior: float) -> None:
        super().__init__()
        if not 0.0 < train_pos_prior < 1.0:
            raise ValueError("train_pos_prior 必须在 (0, 1) 范围内。")
        self.loss = loss
        adjustment = math.log((1.0 - train_pos_prior) / train_pos_prior)
        self.register_buffer(
            "adjustment",
            torch.tensor(adjustment, dtype=torch.float32),
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        adjustment = self.adjustment.to(dtype=logits.dtype, device=logits.device)
        return self.loss(logits + adjustment, targets)
