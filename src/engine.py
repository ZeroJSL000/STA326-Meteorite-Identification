"""严格 Baseline 的单 epoch 训练、验证和预测逻辑。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TrainConfig
from metrics import sigmoid


def build_loss(pos_weight: float, device: torch.device) -> nn.Module:
    """按当前训练折类别分布创建加权二元交叉熵损失。"""

    if not np.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError(f"pos_weight 必须为正有限值，收到: {pos_weight}")
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


def _autocast_context(device: torch.device, enabled: bool) -> Callable:
    return lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    train_config: TrainConfig,
) -> float:
    """执行一次 AMP 与梯度累积训练，不引入额外标签混合策略。"""

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=False)
    autocast = _autocast_context(device, train_config.amp)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
            loss = criterion(logits, targets)
            scaled_loss = loss / train_config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        is_update_step = (step + 1) % train_config.accumulation_steps == 0
        is_last_step = step + 1 == len(loader)
        if is_update_step or is_last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach().item()) * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    description: str = "Valid",
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回验证损失、正类概率和真实标签。"""

    model.eval()
    running_loss = 0.0
    logits_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
            loss = criterion(logits, targets)
        running_loss += float(loss.item()) * images.size(0)
        logits_list.append(logits.float().cpu().numpy())
        target_list.append(targets.cpu().numpy())
    probabilities = sigmoid(np.concatenate(logits_list))
    targets = np.concatenate(target_list).astype(np.int64)
    return running_loss / len(loader.dataset), probabilities, targets


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    description: str,
) -> np.ndarray:
    """对无标签图像输出正类概率。"""

    model.eval()
    logits_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with autocast():
            logits = model(images).flatten()
        logits_list.append(logits.float().cpu().numpy())
    return sigmoid(np.concatenate(logits_list))
