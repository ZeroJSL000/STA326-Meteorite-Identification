"""分层 K-Fold Baseline 训练入口，并自动维护 Markdown 实验日志。"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from timm.utils import ModelEmaV2

from sampler import create_weighted_sampler
from utils.checkpoint import load_weights_flexible
from utils.config import ExperimentConfig, load_config, resolve_image_dir
from utils.dataset import BinaryImageDataset, build_train_transforms, build_valid_transforms
from utils.engine import build_loss, train_one_epoch, validate_one_epoch
from utils.metrics import BinaryMetrics, find_optimal_threshold
from models.builder import build_model


def parse_args() -> argparse.Namespace:
    """提供调试与实验命名所需的命令行覆盖入口。"""

    parser = argparse.ArgumentParser(description="训练 ConvNeXtV2 Baseline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """将命令行显式传入的选项覆盖到集中配置。"""

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.folds is not None:
        config.train.folds = args.folds
    if args.model_name is not None:
        config.model.name = args.model_name
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.no_pretrained:
        config.model.use_pretrained = False
    config.create_output_dirs()
    return config


def seed_everything(seed: int) -> None:
    """固定主要随机源，并允许 cudnn 选择快速实现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """为各数据加载进程生成可复现的独立随机状态。"""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset: BinaryImageDataset,
    batch_size: int,
    shuffle: bool,
    config: ExperimentConfig,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    """统一训练与验证的数据加载性能配置。"""

    generator = torch.Generator()
    generator.manual_seed(config.train.seed)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle and len(dataset) >= batch_size,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": config.data.num_workers > 0,
    }
    if config.data.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def create_grad_scaler(
    device: torch.device,
    amp: bool,
    amp_dtype: str,
) -> torch.amp.GradScaler:
    """创建只在 CUDA 训练时启用的混合精度 scaler。"""

    return torch.amp.GradScaler(
        "cuda",
        enabled=amp and amp_dtype == "float16" and device.type == "cuda",
    )


def build_optimizer(
    model: torch.nn.Module,
    config: ExperimentConfig,
) -> torch.optim.Optimizer:
    """按 head/backbone 分组创建 LLRD AdamW 优化器。"""

    head_parameters = []
    backbone_parameters = []
    head_names: list[str] = []
    backbone_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "head" in name:
            head_parameters.append(parameter)
            head_names.append(name)
        else:
            backbone_parameters.append(parameter)
            backbone_names.append(name)
    if not head_parameters:
        raise RuntimeError("未找到名称包含 `head` 的分类头参数，无法应用 LLRD。")
    if not backbone_parameters:
        raise RuntimeError("未找到 backbone 参数，无法应用 LLRD。")
    head_lr = config.train.learning_rate
    backbone_lr = head_lr * config.train.backbone_lr_factor
    print(
        "LLRD 参数分组: "
        f"head={len(head_names)} tensors lr={head_lr:.3e}; "
        f"backbone={len(backbone_names)} tensors lr={backbone_lr:.3e}"
    )
    return torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": head_lr,
                "weight_decay": config.train.weight_decay,
                "name": "head",
            },
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "weight_decay": config.train.weight_decay,
                "name": "backbone",
            },
        ]
    )


def save_json(payload: object, path: Path) -> None:
    """以 UTF-8 可读 JSON 保存元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def validate_inputs(config: ExperimentConfig, frame: pd.DataFrame) -> Path:
    """在训练开始前校验 schema、标签分布、权重和图像目录。"""

    required_columns = {config.data.id_col, config.data.label_col}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"训练 CSV 缺少列: {sorted(missing_columns)}")
    labels = frame[config.data.label_col].astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("当前任务仅支持标签为 0/1 的二分类数据。")
    smallest_class = int(labels.value_counts().min())
    if smallest_class < config.train.folds:
        raise ValueError(
            f"最小类别仅有 {smallest_class} 张图，无法执行 "
            f"{config.train.folds} 折分层验证。"
        )
    if config.model.use_pretrained and not config.pretrained_path.is_file():
        raise FileNotFoundError(
            f"缺少离线预训练权重 {config.pretrained_path}；"
            "请执行 README.md 的下载指令，或仅在 smoke test 使用 "
            "--no-pretrained。"
        )
    first_image = str(frame.iloc[0][config.data.id_col])
    return resolve_image_dir(config.paths.train_image_dir, first_image)


def augmentation_summary(config: ExperimentConfig) -> str:
    """返回写入实验日志的增强组合简述。"""

    dropout = "no CoarseDropout"
    if config.augment.coarse_dropout_enabled:
        dropout = (
            "CoarseDropout("
            f"max_holes={config.augment.coarse_dropout_max_holes}, "
            f"max_height={config.augment.coarse_dropout_max_height}, "
            f"max_width={config.augment.coarse_dropout_max_width})"
        )
    return (
        "LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + "
        "ColorJitter + RandomGamma + HueSaturationValue + CLAHE + "
        f"Blur/GaussNoise/ImageCompression + {dropout}; "
        "valid/test: aspect-safe resize-pad + Normalize only"
    )


def validate_multiscale_train_sizes(config: ExperimentConfig) -> tuple[int, ...]:
    """校验多尺度训练尺寸与模型 patch size 的整除关系。"""

    if not config.augment.multiscale_train_enabled:
        return ()
    invalid_sizes = [
        size
        for size in config.augment.multiscale_train_sizes
        if size <= 0 or size % config.model.patch_size != 0
    ]
    if invalid_sizes:
        raise ValueError(f"多尺度尺寸必须为正数且能被 patch size 整除: {invalid_sizes}")
    return config.augment.multiscale_train_sizes


def append_experiment_log(
    config: ExperimentConfig,
    experiment_id: str,
    completed_at: str,
    fold_metrics: list[dict[str, float | int]],
    oof_metrics: BinaryMetrics,
    threshold_file: Path,
) -> None:
    """在完整训练成功结束后向 Markdown 日志追加本次实验记录。"""

    log_path = config.experiment_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Experiment Log\n\n", encoding="utf-8")
    loss_description = "BCEWithLogitsLoss()"
    if config.train.pos_weight_mode == "dynamic":
        loss_description = "BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)"
    if config.train.loss_type == "focal":
        loss_description = (
            f"FocalLoss(gamma={config.train.focal_gamma:.4f}, "
            f"alpha={config.train.focal_alpha:.4f})"
        )
    lines = [
        f"## {experiment_id}",
        "",
        f"- 时间: {completed_at}",
        f"- 实验名: `{config.experiment_name}`",
        f"- Config: `{config.config_path}`",
        f"- Backbone: `{config.model.name}`",
        "- Pooling: `GAP (timm default)`",
        f"- 图像尺寸: `{config.data.image_size} x {config.data.image_size}`",
        f"- Loss: `{loss_description}`",
        f"- pos_weight_mode: `{config.train.pos_weight_mode}`",
        f"- Label Smoothing: `{config.train.label_smoothing:.4f}`",
        f"- Logit Adjustment: `{config.train.use_logit_adjustment}`",
        f"- Train Positive Prior: `{config.train.train_pos_prior:.6f}`",
        f"- EMA: `enabled={config.train.ema_enabled}, decay={config.train.ema_decay:.6f}`",
        f"- Optimizer: `{config.train.optimizer_name}` with LLRD",
        f"- LLRD: `head_lr={config.train.learning_rate:.2e}`, `backbone_lr={config.train.learning_rate * config.train.backbone_lr_factor:.2e}`",
        f"- Scheduler: `{config.train.scheduler_name}`",
        f"- 数据增强: {augmentation_summary(config)}",
        "",
        "| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metrics in fold_metrics:
        loss_weight = metrics.get("loss_pos_weight")
        loss_weight_text = "plain_bce" if loss_weight is None else f"{loss_weight:.6f}"
        lines.append(
            f"| {int(metrics['fold'])} | {int(metrics['positive_count'])} | "
            f"{int(metrics['negative_count'])} | {metrics['class_ratio']:.6f} | "
            f"{loss_weight_text} | {metrics['f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- OOF F1: `{oof_metrics.f1:.6f}`",
            f"- OOF 最优阈值: `{oof_metrics.threshold:.6f}`",
            "- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`",
            f"- 阈值文件: `{threshold_file}`",
            "- Kaggle LB Score: `待补充`",
            "- 后续推断改进方向: 待补充",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def apply_pseudo_training_schedule(config: ExperimentConfig) -> None:
    """伪标签重训练时按配置缩小学习率和训练轮数。"""

    if not config.pseudo_labeling.enabled:
        return
    config.train.learning_rate *= config.pseudo_labeling.pseudo_lr_factor
    config.train.epochs = max(
        1,
        int(round(config.train.epochs * config.pseudo_labeling.pseudo_epoch_factor)),
    )
    print(
        "伪标签训练日程已启用: "
        f"lr={config.train.learning_rate:.3e}; epochs={config.train.epochs}"
    )


def load_pseudo_training_frame(config: ExperimentConfig) -> pd.DataFrame | None:
    """读取严格筛选后的伪标签，并映射到掩码测试图目录。"""

    if not config.pseudo_labeling.enabled:
        return None
    path = config.paths.pseudo_labels_csv
    if not path.is_file():
        raise FileNotFoundError(f"找不到伪标签文件: {path}")
    frame = pd.read_csv(path)
    id_col = "image_name" if "image_name" in frame else config.data.id_col
    required_columns = {id_col, config.data.label_col, "is_pseudo"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"伪标签 CSV 缺少列: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("伪标签 CSV 为空，无法启动伪标签训练。")
    frame = frame.rename(columns={id_col: config.data.id_col}).copy()
    frame[config.data.id_col] = frame[config.data.id_col].astype(str)
    frame[config.data.label_col] = frame[config.data.label_col].astype(int)
    if not set(frame[config.data.label_col].unique()).issubset({0, 1}):
        raise ValueError("伪标签必须为 0/1。")
    first_image = str(frame.iloc[0][config.data.id_col])
    test_image_dir = resolve_image_dir(config.paths.test_image_dir, first_image)
    frame["image_dir"] = str(test_image_dir)
    frame["sample_weight"] = config.pseudo_labeling.pseudo_sample_weight
    frame["is_pseudo"] = 1
    frame["use_soft_masking"] = False
    print(f"载入伪标签: {len(frame)} 行; 测试图目录={test_image_dir}")
    return frame


def run_training(config: ExperimentConfig) -> None:
    """训练全部折，保存 OOF、最优阈值、折模型和实验日志。"""

    apply_pseudo_training_schedule(config)
    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(config.paths.train_csv)
    train_image_dir = validate_inputs(config, frame)
    pseudo_frame = load_pseudo_training_frame(config)
    original_image_dir: Path | None = None
    if config.preprocessing.soft_masking_enabled:
        first_image = str(frame.iloc[0][config.data.id_col])
        try:
            original_image_dir = resolve_image_dir(
                config.paths.original_image_dir,
                first_image,
            )
        except FileNotFoundError as error:
            warnings.warn(
                f"Soft-Masking 原图目录不可用；回退到纯掩码图。详情: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    print(f"设备: {device}; 训练图像目录: {train_image_dir}")
    if original_image_dir is not None:
        print(
            "Soft-Masking 已启用: "
            f"原图目录={original_image_dir}; "
            f"mask_alpha={config.preprocessing.soft_masking_alpha:.3f}"
        )
    print(f"实验: {config.experiment_name}; 模型: {config.model.name}")
    multiscale_train_sizes = validate_multiscale_train_sizes(config)

    splitter = StratifiedKFold(
        n_splits=config.train.folds,
        shuffle=True,
        random_state=config.train.seed,
    )
    frame = frame.copy()
    frame["fold"] = -1
    labels = frame[config.data.label_col].astype(int).to_numpy()
    for fold, (_, valid_indices) in enumerate(splitter.split(frame, labels)):
        frame.loc[valid_indices, "fold"] = fold

    oof_probabilities = np.zeros(len(frame), dtype=np.float64)
    training_log: list[dict[str, float | int]] = []
    fold_metrics: list[dict[str, float | int]] = []
    checkpoint_files: list[str] = []
    fold_count = config.train.fold_limit or config.train.folds
    if not 0 < fold_count <= config.train.folds:
        raise ValueError("train.fold_limit 必须在 (0, train.folds] 范围内。")
    if fold_count != config.train.folds:
        print(f"调试模式: 仅训练前 {fold_count}/{config.train.folds} 个 fold。")

    for fold in range(fold_count):
        print(f"\n===== Fold {fold + 1}/{config.train.folds} =====")
        real_train_frame = frame[frame["fold"] != fold].reset_index(drop=True)
        train_frame = real_train_frame
        if pseudo_frame is not None:
            real_train_frame = real_train_frame.copy()
            real_train_frame["image_dir"] = str(train_image_dir)
            real_train_frame["sample_weight"] = 1.0
            real_train_frame["is_pseudo"] = 0
            real_train_frame["use_soft_masking"] = True
            train_frame = pd.concat([real_train_frame, pseudo_frame], ignore_index=True, sort=False)
        valid_frame = frame[frame["fold"] == fold].reset_index(drop=True)
        valid_indices = frame.index[frame["fold"] == fold].to_numpy()
        positive_count = int(train_frame[config.data.label_col].sum())
        negative_count = int(len(train_frame) - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"Fold {fold} 训练集必须同时包含正负样本。")
        class_ratio = negative_count / positive_count
        loss_pos_weight = (
            class_ratio if config.train.pos_weight_mode == "dynamic" else None
        )
        loss_weight_text = (
            "plain_bce" if loss_pos_weight is None else f"{loss_pos_weight:.6f}"
        )
        print(
            f"Fold {fold} 类别统计: positive={positive_count}, "
            f"negative={negative_count}, class_ratio={class_ratio:.6f}, "
            f"loss_weight={loss_weight_text}"
        )
        train_dataset = BinaryImageDataset(
            train_frame,
            train_image_dir,
            build_train_transforms(config.data.image_size, config.augment),
            config.data.id_col,
            config.data.label_col,
            soft_masking_enabled=config.preprocessing.soft_masking_enabled,
            soft_masking_alpha=config.preprocessing.soft_masking_alpha,
            original_image_dir=original_image_dir,
        )
        valid_dataset = BinaryImageDataset(
            valid_frame,
            train_image_dir,
            build_valid_transforms(config.data.image_size),
            config.data.id_col,
            config.data.label_col,
            soft_masking_enabled=config.preprocessing.soft_masking_enabled,
            soft_masking_alpha=config.preprocessing.soft_masking_alpha,
            original_image_dir=original_image_dir,
        )
        train_sampler: WeightedRandomSampler | None = None
        sampling_weights = np.ones(len(train_frame), dtype=np.float64)
        if config.train.hard_mining_enabled:
            oof_path = (
                config.train.hard_mining_oof_path
                or config.experiment_output_dir / "oof_predictions.csv"
            )
            hard_mining_sampler = create_weighted_sampler(
                oof_path,
                topk=config.train.hard_mining_topk,
                boost=config.train.hard_mining_boost,
                sample_ids=real_train_frame[config.data.id_col].astype(str).tolist(),
                id_col=config.data.id_col,
                label_col=config.data.label_col,
            )
            sampling_weights[: len(real_train_frame)] *= hard_mining_sampler.weights.numpy()
            print(f"Fold {fold} 困难样本重采样已启用: OOF={oof_path}")
        if pseudo_frame is not None:
            sampling_weights *= train_frame["sample_weight"].to_numpy(dtype=np.float64)
        if config.train.hard_mining_enabled or pseudo_frame is not None:
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sampling_weights, dtype=torch.double),
                num_samples=len(train_frame),
                replacement=True,
            )
        train_loader = build_loader(
            train_dataset,
            config.train.batch_size,
            True,
            config,
            device,
            sampler=train_sampler,
        )
        valid_loader = build_loader(
            valid_dataset,
            config.train.valid_batch_size,
            False,
            config,
            device,
        )

        pretrained_path = (
            config.pretrained_path if config.model.use_pretrained else None
        )
        model = build_model(
            config.model,
            pretrained_path=pretrained_path,
            require_pretrained=config.model.use_pretrained,
        ).to(device)
        if pseudo_frame is not None and config.pseudo_labeling.init_checkpoint_dir is not None:
            init_checkpoint = config.pseudo_labeling.init_checkpoint_dir / f"fold_{fold}_best.pth"
            if not init_checkpoint.is_file():
                raise FileNotFoundError(f"找不到伪标签微调初始化 checkpoint: {init_checkpoint}")
            loaded_count, skipped = load_weights_flexible(model, init_checkpoint)
            print(
                "伪标签微调初始化完成: "
                f"loaded={loaded_count}; skipped={len(skipped)}; checkpoint={init_checkpoint}"
            )
        model_ema = None
        if config.train.ema_enabled:
            model_ema = ModelEmaV2(model, decay=config.train.ema_decay)
            print(f"EMA 已启用: decay={config.train.ema_decay:.6f}")
        criterion = build_loss(config.train, device, loss_pos_weight)
        optimizer = build_optimizer(model, config)
        warmup_epochs = max(1, config.train.epochs // 10)
        if config.train.epochs == 1:
            scheduler: torch.optim.lr_scheduler.LRScheduler = (
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    total_iters=warmup_epochs,
                )
            )
        else:
            cosine_epochs = config.train.epochs - warmup_epochs
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=config.train.min_learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        scaler = create_grad_scaler(device, config.train.amp, config.train.amp_dtype)
        best_f1 = -1.0
        best_probabilities: np.ndarray | None = None
        best_metrics: BinaryMetrics | None = None
        bad_epochs = 0
        checkpoint_path = config.checkpoint_dir / f"fold_{fold}_best.pth"
        checkpoint_files.append(checkpoint_path.name)

        for epoch in range(config.train.epochs):
            if multiscale_train_sizes:
                epoch_image_size = random.choice(multiscale_train_sizes)
                train_dataset.transform = build_train_transforms(
                    epoch_image_size,
                    config.augment,
                )
                train_loader = build_loader(
                    train_dataset,
                    config.train.batch_size,
                    True,
                    config,
                    device,
                    sampler=train_sampler,
                )
                print(f"Fold {fold} Epoch {epoch + 1}: train_image_size={epoch_image_size}")
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                train_config=config.train,
                model_ema=model_ema,
                augment_config=config.augment,
            )
            eval_model = model_ema.module if model_ema is not None else model
            valid_loss, probabilities, targets = validate_one_epoch(
                eval_model,
                valid_loader,
                criterion,
                device,
                config.train.amp,
                description=f"Valid fold {fold + 1}",
                inference_config=config.inference,
                amp_dtype=config.train.amp_dtype,
            )
            metrics = find_optimal_threshold(targets, probabilities)
            lr_by_group = {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            training_log.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "class_ratio": class_ratio,
                    "loss_pos_weight": loss_pos_weight,
                    "label_smoothing": config.train.label_smoothing,
                    "ema_enabled": config.train.ema_enabled,
                    "ema_decay": config.train.ema_decay
                    if config.train.ema_enabled
                    else None,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "f1": metrics.f1,
                    "threshold": metrics.threshold,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "head_learning_rate": lr_by_group["head"],
                    "backbone_learning_rate": lr_by_group["backbone"],
                }
            )
            pd.DataFrame(training_log).to_csv(
                config.experiment_output_dir / "training_log.csv",
                index=False,
            )
            print(
                f"Fold {fold} Epoch {epoch + 1}: train={train_loss:.5f} "
                f"valid={valid_loss:.5f} F1={metrics.f1:.5f} "
                f"thr={metrics.threshold:.2f}"
            )
            if metrics.f1 > best_f1:
                best_f1 = metrics.f1
                best_probabilities = probabilities.copy()
                best_metrics = metrics
                bad_epochs = 0
                save_model = model_ema.module if model_ema is not None else model
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in save_model.state_dict().items()
                        },
                        "fold": fold,
                        "model_name": config.model.name,
                        "image_size": config.data.image_size,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "class_ratio": class_ratio,
                        "loss_pos_weight": loss_pos_weight,
                        "pos_weight_mode": config.train.pos_weight_mode,
                        "label_smoothing": config.train.label_smoothing,
                        "ema_enabled": config.train.ema_enabled,
                        "ema_decay": config.train.ema_decay
                        if config.train.ema_enabled
                        else None,
                        "checkpoint_source": "ema"
                        if model_ema is not None
                        else "model",
                        "metrics": metrics.to_dict(),
                    },
                    checkpoint_path,
                )
            else:
                bad_epochs += 1
            scheduler.step()
            if bad_epochs >= config.train.patience:
                print(f"Fold {fold} 在 epoch {epoch + 1} 触发 F1 早停。")
                break
        if best_probabilities is None or best_metrics is None:
            raise RuntimeError(f"Fold {fold} 未产生可保存的验证预测。")
        oof_probabilities[valid_indices] = best_probabilities
        fold_metrics.append(
            {
                "fold": fold,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "class_ratio": class_ratio,
                "loss_pos_weight": loss_pos_weight,
                **best_metrics.to_dict(),
            }
        )

    completed_mask = (frame["fold"] < fold_count).to_numpy()
    completed_probabilities = oof_probabilities[completed_mask]
    completed_labels = labels[completed_mask]
    oof_frame = frame.loc[completed_mask, [config.data.id_col, config.data.label_col, "fold"]].copy()
    oof_frame["probability"] = completed_probabilities
    oof_frame["prob"] = completed_probabilities
    oof_metrics = find_optimal_threshold(completed_labels, completed_probabilities)
    oof_frame["prediction"] = (
        oof_frame["probability"] >= oof_metrics.threshold
    ).astype(int)
    oof_frame.to_csv(config.experiment_output_dir / "oof_predictions.csv", index=False)
    completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"EXP-{timestamp}-{config.experiment_name}"
    threshold_path = config.experiment_output_dir / "optimal_threshold.json"
    latest_threshold_path = config.paths.outputs_dir / "optimal_threshold.json"
    threshold_payload = {
        "experiment_id": experiment_id,
        "experiment_name": config.experiment_name,
        "source": "out_of_fold_predictions",
        "metric": "f1_score",
        "search_min": 0.10,
        "search_max": 0.90,
        "search_step": 0.01,
        "optimal_threshold": oof_metrics.threshold,
        "oof_metrics": oof_metrics.to_dict(),
    }
    save_json(threshold_payload, threshold_path)
    save_json(threshold_payload, latest_threshold_path)
    metadata = {
        "experiment_id": experiment_id,
        "completed_at": completed_at,
        "experiment_name": config.experiment_name,
        "model_name": config.model.name,
        "image_size": config.data.image_size,
        "folds": fold_count,
        "validation_folds": config.train.folds,
        "checkpoint_files": checkpoint_files,
        "fold_metrics": fold_metrics,
        "oof_metrics": oof_metrics.to_dict(),
        "optimal_threshold_file": str(threshold_path),
        "ema_enabled": config.train.ema_enabled,
        "ema_decay": config.train.ema_decay if config.train.ema_enabled else None,
        "checkpoint_source": "ema" if config.train.ema_enabled else "model",
        "config": config.to_dict(),
    }
    save_json(metadata, config.checkpoint_dir / "metadata.json")
    save_json(metadata, config.experiment_output_dir / "metadata.json")
    append_experiment_log(
        config,
        experiment_id,
        completed_at,
        fold_metrics,
        oof_metrics,
        threshold_path,
    )
    print(
        f"\nOOF F1={oof_metrics.f1:.5f}; "
        f"最佳阈值={oof_metrics.threshold:.2f}; "
        f"阈值文件={threshold_path}; 日志={config.experiment_log_path}"
    )


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    save_json(config.to_dict(), config.experiment_output_dir / "config.json")
    run_training(config)


if __name__ == "__main__":
    main()
