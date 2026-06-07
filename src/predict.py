"""加载 K-Fold 权重并用平均概率生成提交文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.config import ExperimentConfig, load_config, resolve_image_dir
from utils.dataset import BinaryImageDataset, build_valid_transforms
from utils.engine import predict_probabilities
from models.builder import build_model
from utils.checkpoint import load_weights_flexible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行折模型 Baseline 集成推理")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def build_test_loader(
    frame: pd.DataFrame,
    image_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    """构建严格 aspect-safe 的测试数据加载器。"""

    dataset = BinaryImageDataset(
        frame=frame,
        image_dir=image_dir,
        transform=build_valid_transforms(config.data.image_size),
        id_col=config.data.id_col,
        label_col=None,
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": config.inference.batch_size,
        "shuffle": False,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.data.num_workers > 0,
    }
    if config.data.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def apply_target_positive_count(
    submission: pd.DataFrame,
    probabilities: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """按模型概率排序，将提交校准为固定正类数量。"""

    target_count = config.inference.target_positive_count
    if target_count is None:
        return submission
    if target_count > len(submission):
        raise ValueError(
            f"target_positive_count={target_count} 超过提交行数 {len(submission)}。"
        )
    calibrated = submission.copy()
    order = np.argsort(probabilities)[::-1]
    labels = np.zeros(len(calibrated), dtype=int)
    labels[order[:target_count]] = 1
    calibrated[config.data.label_col] = labels
    path = config.experiment_output_dir / "submission_count_calibrated.csv"
    calibrated.to_csv(path, index=False)
    print(
        f"正类数量校准已应用: target_positive={target_count}, "
        f"target_negative={len(calibrated) - target_count}; saved={path}"
    )
    return calibrated


def load_optimal_threshold(
    metadata: dict[str, object], config: ExperimentConfig
) -> float:
    """读取 OOF 阈值，并兼容旧 checkpoint 中保存的绝对路径。"""

    candidate_paths = [config.experiment_output_dir / "optimal_threshold.json"]
    path_value = metadata.get("optimal_threshold_file")
    if path_value:
        candidate_paths.append(Path(str(path_value)))

    checked: list[str] = []
    for threshold_path in candidate_paths:
        checked.append(str(threshold_path))
        if not threshold_path.is_file():
            continue
        with threshold_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("experiment_name") != metadata.get("experiment_name"):
            raise ValueError("阈值文件与 checkpoint 元数据的实验名不一致。")
        threshold = float(payload["optimal_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"OOF 阈值非法: {threshold}")
        return threshold

    oof_metrics = metadata.get("oof_metrics")
    if isinstance(oof_metrics, dict) and "threshold" in oof_metrics:
        threshold = float(oof_metrics["threshold"])
        if 0.0 < threshold < 1.0:
            print(
                "未找到本地 OOF 阈值文件，使用 checkpoint metadata 中的 "
                f"threshold={threshold:.4f}。"
            )
            return threshold

    raise FileNotFoundError(
        "找不到 OOF 最优阈值文件，且 checkpoint metadata 中没有可用阈值。"
        f"已检查: {checked}"
    )


def run_prediction(
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> None:
    """平均各折预测概率，并按配置做固定正类数量校准。"""

    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到训练元数据: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    config.model.name = metadata["model_name"]
    config.data.image_size = int(metadata["image_size"])
    checkpoint_files = metadata["checkpoint_files"]
    threshold = load_optimal_threshold(metadata, config)

    sample_submission = pd.read_csv(config.paths.sample_submission_csv)
    required_columns = {config.data.id_col, config.data.label_col}
    if not required_columns.issubset(sample_submission.columns):
        raise KeyError(f"提交模板必须包含列: {sorted(required_columns)}")
    first_image = str(sample_submission.iloc[0][config.data.id_col])
    test_image_dir = resolve_image_dir(config.paths.test_image_dir, first_image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability_sum = np.zeros(len(sample_submission), dtype=np.float64)
    print(f"设备: {device}; 测试图像目录: {test_image_dir}; OOF 阈值: {threshold:.2f}")

    loader = build_test_loader(sample_submission, test_image_dir, config, device)
    for filename in checkpoint_files:
        checkpoint_path = checkpoint_dir / filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"缺少折模型权重: {checkpoint_path}")
        model = build_model(config.model).to(device)
        loaded_count, _ = load_weights_flexible(model, checkpoint_path)
        print(f"载入 {filename}: {loaded_count} 项参数")
        probability_sum += predict_probabilities(
            model,
            loader,
            device,
            config.train.amp,
            description=filename,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = probability_sum / len(checkpoint_files)
    model_labels = (probabilities >= threshold).astype(int)
    model_submission = sample_submission.copy()
    model_submission[config.data.label_col] = model_labels
    probability_frame = model_submission[[config.data.id_col]].copy()
    probability_frame["probability"] = probabilities
    probability_frame["prediction"] = model_submission[config.data.label_col]
    model_submission_path = config.experiment_output_dir / "submission_model_only.csv"
    probability_path = config.experiment_output_dir / "submission_probabilities.csv"
    model_submission.to_csv(model_submission_path, index=False)
    probability_frame.to_csv(probability_path, index=False)

    submission = model_submission
    submission = apply_target_positive_count(submission, probabilities, config)
    submission_path = config.experiment_output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(
        f"提交文件已保存: {submission_path}; "
        f"模型单独提交: {model_submission_path}; "
        f"预测正类数: {int(submission[config.data.label_col].sum())}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
        config.create_output_dirs()
    checkpoint_dir = args.checkpoint_dir or config.checkpoint_dir
    run_prediction(config, checkpoint_dir)


if __name__ == "__main__":
    main()
