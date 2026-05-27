"""分层 K-Fold Baseline 训练入口，并自动维护 Markdown 实验日志。"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from config import ExperimentConfig, load_config, resolve_image_dir
from dataset import BinaryImageDataset, build_train_transforms, build_valid_transforms
from engine import build_loss, train_one_epoch, validate_one_epoch
from metrics import BinaryMetrics, find_optimal_threshold
from model import build_model


def parse_args() -> argparse.Namespace:
    """提供调试与实验命名所需的命令行覆盖入口。"""

    parser = argparse.ArgumentParser(description="训练 ConvNeXtV2 Baseline")
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
) -> DataLoader:
    """统一训练与验证的数据加载性能配置。"""

    generator = torch.Generator()
    generator.manual_seed(config.train.seed)
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
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


def create_grad_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    """创建只在 CUDA 训练时启用的混合精度 scaler。"""

    return torch.amp.GradScaler(
        "cuda",
        enabled=amp and device.type == "cuda",
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


def augmentation_summary() -> str:
    """返回写入实验日志的增强组合简述。"""

    return (
        "LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + "
        "ColorJitter + RandomGamma + HueSaturationValue + CLAHE + "
        "Blur/GaussNoise/ImageCompression + Strong CoarseDropout(max_holes=8, "
        "max_size=64); "
        "valid/test: aspect-safe resize-pad + Normalize only"
    )


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
    lines = [
        f"## {experiment_id}",
        "",
        f"- 时间: {completed_at}",
        f"- 实验名: `{config.experiment_name}`",
        f"- Backbone: `{config.model.name}`",
        f"- Pooling: `GeM(p_init={config.model.gem_p:.2f})`",
        f"- 图像尺寸: `{config.data.image_size} x {config.data.image_size}`",
        "- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`",
        f"- Optimizer: `{config.train.optimizer_name}`",
        f"- Scheduler: `{config.train.scheduler_name}`",
        f"- 数据增强: {augmentation_summary()}",
        "",
        "| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for metrics in fold_metrics:
        lines.append(
            f"| {int(metrics['fold'])} | {int(metrics['positive_count'])} | "
            f"{int(metrics['negative_count'])} | {metrics['pos_weight']:.6f} | "
            f"{metrics['f1']:.6f} |"
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


def run_training(config: ExperimentConfig) -> None:
    """训练全部折，保存 OOF、最优阈值、折模型和实验日志。"""

    seed_everything(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(config.paths.train_csv)
    train_image_dir = validate_inputs(config, frame)
    print(f"设备: {device}; 训练图像目录: {train_image_dir}")
    print(f"实验: {config.experiment_name}; 模型: {config.model.name}")

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

    for fold in range(config.train.folds):
        print(f"\n===== Fold {fold + 1}/{config.train.folds} =====")
        train_frame = frame[frame["fold"] != fold].reset_index(drop=True)
        valid_frame = frame[frame["fold"] == fold].reset_index(drop=True)
        valid_indices = frame.index[frame["fold"] == fold].to_numpy()
        positive_count = int(train_frame[config.data.label_col].sum())
        negative_count = int(len(train_frame) - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise ValueError(f"Fold {fold} 训练集必须同时包含正负样本。")
        pos_weight = negative_count / positive_count
        print(
            f"Fold {fold} 类别统计: positive={positive_count}, "
            f"negative={negative_count}, pos_weight={pos_weight:.6f}"
        )
        train_dataset = BinaryImageDataset(
            train_frame,
            train_image_dir,
            build_train_transforms(config.data.image_size, config.augment),
            config.data.id_col,
            config.data.label_col,
        )
        valid_dataset = BinaryImageDataset(
            valid_frame,
            train_image_dir,
            build_valid_transforms(config.data.image_size),
            config.data.id_col,
            config.data.label_col,
        )
        train_loader = build_loader(
            train_dataset,
            config.train.batch_size,
            True,
            config,
            device,
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
        criterion = build_loss(pos_weight, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs,
            eta_min=config.train.min_learning_rate,
        )
        scaler = create_grad_scaler(device, config.train.amp)
        best_f1 = -1.0
        best_probabilities: np.ndarray | None = None
        best_metrics: BinaryMetrics | None = None
        bad_epochs = 0
        checkpoint_path = config.checkpoint_dir / f"fold_{fold}_best.pth"
        checkpoint_files.append(checkpoint_path.name)

        for epoch in range(config.train.epochs):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                train_config=config.train,
            )
            valid_loss, probabilities, targets = validate_one_epoch(
                model,
                valid_loader,
                criterion,
                device,
                config.train.amp,
                description=f"Valid fold {fold + 1}",
            )
            metrics = find_optimal_threshold(targets, probabilities)
            current_lr = float(optimizer.param_groups[0]["lr"])
            training_log.append(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "pos_weight": pos_weight,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "f1": metrics.f1,
                    "threshold": metrics.threshold,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "learning_rate": current_lr,
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
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "fold": fold,
                        "model_name": config.model.name,
                        "image_size": config.data.image_size,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "pos_weight": pos_weight,
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
                "pos_weight": pos_weight,
                **best_metrics.to_dict(),
            }
        )

    oof_frame = frame[[config.data.id_col, config.data.label_col, "fold"]].copy()
    oof_frame["probability"] = oof_probabilities
    oof_metrics = find_optimal_threshold(labels, oof_probabilities)
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
        "folds": config.train.folds,
        "checkpoint_files": checkpoint_files,
        "fold_metrics": fold_metrics,
        "oof_metrics": oof_metrics.to_dict(),
        "optimal_threshold_file": str(threshold_path),
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
    config = apply_cli_overrides(load_config(), parse_args())
    save_json(config.to_dict(), config.experiment_output_dir / "config.json")
    run_training(config)


if __name__ == "__main__":
    main()
