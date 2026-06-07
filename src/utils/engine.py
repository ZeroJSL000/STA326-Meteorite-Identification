"""严格 Baseline 的单 epoch 训练、验证和预测逻辑。"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from losses import FocalLoss, LogitAdjustedLoss
from utils.config import AugmentConfig, InferenceConfig, TrainConfig
from utils.metrics import sigmoid


class SmoothedBCEWithLogitsLoss(nn.Module):
    """支持手动 label smoothing 与可选 pos_weight 的 BCEWithLogitsLoss。"""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing 必须在 [0, 1) 范围内。")
        self.label_smoothing = float(label_smoothing)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.label_smoothing > 0.0:
            targets = (
                targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


def build_loss(
    train_config: TrainConfig,
    device: torch.device,
    pos_weight: float | None = None,
) -> nn.Module:
    """按配置创建 BCE 或 Focal Loss；none 模式不注入 pos_weight。"""

    weight_tensor: Tensor | None = None
    if train_config.pos_weight_mode == "dynamic":
        if pos_weight is None or not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError(f"动态 pos_weight 必须为正有限值，收到: {pos_weight}")
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    elif train_config.pos_weight_mode != "none":
        raise ValueError(f"不支持的 pos_weight_mode: {train_config.pos_weight_mode}")
    if train_config.loss_type == "bce":
        criterion: nn.Module = SmoothedBCEWithLogitsLoss(
            label_smoothing=train_config.label_smoothing,
            pos_weight=weight_tensor,
        )
    elif train_config.loss_type == "focal":
        criterion = FocalLoss(
            gamma=train_config.focal_gamma,
            alpha=train_config.focal_alpha,
            pos_weight=weight_tensor,
        )
    else:
        raise ValueError(f"不支持的 loss_type: {train_config.loss_type}")
    if train_config.use_logit_adjustment:
        criterion = LogitAdjustedLoss(criterion, train_config.train_pos_prior)
    return criterion


def _autocast_context(
    device: torch.device,
    enabled: bool,
    amp_dtype: str = "float16",
) -> Callable:
    """Create a CUDA autocast context with an explicitly selected dtype."""
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = dtype_by_name[amp_dtype]
    except KeyError as exc:
        raise ValueError(f"不支持的 AMP dtype: {amp_dtype}") from exc
    return lambda: torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled and device.type == "cuda",
    )


def _require_finite(tensor: Tensor, description: str) -> None:
    """Stop immediately instead of allowing NaN values to poison checkpoints."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"检测到非有限值: {description}")


def apply_batch_mixup_cutmix(
    images: Tensor,
    targets: Tensor,
    config: AugmentConfig | None,
) -> tuple[Tensor, Tensor]:
    """按样本互斥执行 MixUp 或 CutMix，并返回软标签。"""

    if config is None:
        return images, targets
    mixup_probability = config.mixup_probability if config.mixup_enabled else 0.0
    cutmix_probability = config.cutmix_probability if config.cutmix_enabled else 0.0
    if mixup_probability + cutmix_probability <= 0.0:
        return images, targets

    batch_size = images.size(0)
    decisions = torch.rand(batch_size, device=images.device)
    mixup_mask = decisions < mixup_probability
    cutmix_mask = (decisions >= mixup_probability) & (
        decisions < mixup_probability + cutmix_probability
    )
    if not mixup_mask.any() and not cutmix_mask.any():
        return images, targets

    permutation = torch.randperm(batch_size, device=images.device)
    source_images = images
    source_targets = targets
    mixed_images = images.clone()
    mixed_targets = targets.clone()

    if mixup_mask.any():
        concentration = torch.tensor(config.mixup_alpha, device=images.device)
        distribution = torch.distributions.Beta(concentration, concentration)
        lambdas = distribution.sample((batch_size,)).to(dtype=images.dtype)
        lambdas = torch.where(mixup_mask, lambdas, torch.ones_like(lambdas))
        image_lambdas = lambdas.view(batch_size, 1, 1, 1)
        mixed_images = mixed_images * image_lambdas + source_images[permutation] * (
            1.0 - image_lambdas
        )
        mixed_targets = mixed_targets * lambdas + source_targets[permutation] * (
            1.0 - lambdas
        )

    if cutmix_mask.any():
        height, width = images.shape[-2:]
        concentration = torch.tensor(config.cutmix_alpha, device=images.device)
        distribution = torch.distributions.Beta(concentration, concentration)
        indices = torch.nonzero(cutmix_mask, as_tuple=False).flatten().tolist()
        lambdas = distribution.sample((len(indices),)).tolist()
        for sample_index, sampled_lambda in zip(indices, lambdas, strict=True):
            cut_ratio = math.sqrt(1.0 - sampled_lambda)
            cut_height = int(height * cut_ratio)
            cut_width = int(width * cut_ratio)
            center_y = int(torch.randint(height, (1,), device=images.device).item())
            center_x = int(torch.randint(width, (1,), device=images.device).item())
            y1 = max(center_y - cut_height // 2, 0)
            y2 = min(center_y + cut_height // 2, height)
            x1 = max(center_x - cut_width // 2, 0)
            x2 = min(center_x + cut_width // 2, width)
            mixed_images[sample_index, :, y1:y2, x1:x2] = source_images[
                permutation[sample_index], :, y1:y2, x1:x2
            ]
            adjusted_lambda = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
            mixed_targets[sample_index] = (
                source_targets[sample_index] * adjusted_lambda
                + source_targets[permutation[sample_index]] * (1.0 - adjusted_lambda)
            )
    return mixed_images, mixed_targets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    train_config: TrainConfig,
    model_ema: Any | None = None,
    augment_config: AugmentConfig | None = None,
) -> float:
    """执行一次 AMP 与梯度累积训练，并在优化器更新后同步 EMA。"""

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"Train {epoch + 1}", leave=False)
    autocast = _autocast_context(device, train_config.amp, train_config.amp_dtype)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        images, targets = apply_batch_mixup_cutmix(images, targets, augment_config)
        with autocast():
            logits = model(images).flatten()
            _require_finite(logits, f"Train epoch={epoch + 1} step={step + 1} logits")
            loss = criterion(logits, targets)
            _require_finite(loss, f"Train epoch={epoch + 1} step={step + 1} loss")
            scaled_loss = loss / train_config.accumulation_steps
        scaler.scale(scaled_loss).backward()
        is_update_step = (step + 1) % train_config.accumulation_steps == 0
        is_last_step = step + 1 == len(loader)
        if is_update_step or is_last_step:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                train_config.max_grad_norm,
            )
            gradient_is_finite = bool(torch.isfinite(gradient_norm).item())
            if not gradient_is_finite and not scaler.is_enabled():
                _require_finite(
                    gradient_norm,
                    f"Train epoch={epoch + 1} step={step + 1} grad_norm",
                )
            old_scale = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = old_scale is not None and scaler.get_scale() < old_scale
            if not gradient_is_finite:
                progress.write(
                    "FP16 梯度溢出，GradScaler 已跳过更新并降低 scale: "
                    f"epoch={epoch + 1}, step={step + 1}, "
                    f"scale={old_scale:.1f}->{scaler.get_scale():.1f}"
                )
            if model_ema is not None and not step_was_skipped:
                model_ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach().item()) * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def _tta_views(images: Tensor, config: InferenceConfig) -> list[Tensor]:
    """按配置生成尺度与水平翻转的笛卡尔积。"""

    views: list[Tensor] = []
    for scale in config.tta_scales:
        if images.shape[-2:] == (scale, scale):
            scaled_images = images
        else:
            scaled_images = F.interpolate(
                images,
                size=(scale, scale),
                mode="bilinear",
                align_corners=False,
            )
        for flip in config.tta_flips:
            if flip == "none":
                views.append(scaled_images)
            elif flip == "horizontal":
                views.append(torch.flip(scaled_images, dims=(-1,)))
            else:
                raise ValueError(f"不支持的 TTA flip: {flip}")
    return views


def _predict_tta_probabilities(
    model: nn.Module,
    images: Tensor,
    autocast: Callable,
    config: InferenceConfig,
) -> Tensor:
    probabilities: list[Tensor] = []
    for view in _tta_views(images, config):
        with autocast():
            logits = model(view).flatten()
        probabilities.append(torch.sigmoid(logits.float()))
    return torch.stack(probabilities).mean(dim=0)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    description: str = "Valid",
    inference_config: InferenceConfig | None = None,
    amp_dtype: str = "float16",
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回验证损失、正类概率和真实标签。"""

    model.eval()
    running_loss = 0.0
    logits_list: list[np.ndarray] = []
    probability_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp, amp_dtype)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if inference_config is not None and inference_config.tta_enabled:
            probabilities = _predict_tta_probabilities(
                model, images, autocast, inference_config
            )
            _require_finite(probabilities, f"{description} TTA probabilities")
            epsilon = torch.finfo(probabilities.dtype).eps
            logits = torch.logit(probabilities.clamp(epsilon, 1.0 - epsilon))
            loss = criterion(logits, targets)
            _require_finite(loss, f"{description} TTA loss")
            probability_list.append(probabilities.cpu().numpy())
        else:
            with autocast():
                logits = model(images).flatten()
                _require_finite(logits, f"{description} logits")
                loss = criterion(logits, targets)
                _require_finite(loss, f"{description} loss")
            logits_list.append(logits.float().cpu().numpy())
        running_loss += float(loss.item()) * images.size(0)
        target_list.append(targets.cpu().numpy())
    if inference_config is not None and inference_config.tta_enabled:
        probabilities = np.concatenate(probability_list)
    else:
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
    inference_config: InferenceConfig | None = None,
    amp_dtype: str = "float16",
) -> np.ndarray:
    """对无标签图像输出正类概率。"""

    model.eval()
    logits_list: list[np.ndarray] = []
    probability_list: list[np.ndarray] = []
    autocast = _autocast_context(device, amp, amp_dtype)
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if inference_config is not None and inference_config.tta_enabled:
            probabilities = _predict_tta_probabilities(model, images, autocast, inference_config)
            _require_finite(probabilities, f"{description} TTA probabilities")
            probability_list.append(probabilities.cpu().numpy())
        else:
            with autocast():
                logits = model(images).flatten()
            _require_finite(logits, f"{description} logits")
            logits_list.append(logits.float().cpu().numpy())
    if inference_config is not None and inference_config.tta_enabled:
        return np.concatenate(probability_list)
    return sigmoid(np.concatenate(logits_list))
