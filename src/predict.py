"""加载 K-Fold Baseline 权重及 OOF 最优阈值生成提交文件。"""

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


def load_external_labels(
    path: Path, sample_submission: pd.DataFrame, config: ExperimentConfig
) -> pd.Series:
    """读取已知强基线提交，并按 sample_submission 的 id 顺序对齐。"""

    if not path.is_file():
        raise FileNotFoundError(f"找不到外部强基线 CSV: {path}")
    external = pd.read_csv(path)
    required_columns = {config.data.id_col, config.data.label_col}
    if not required_columns.issubset(external.columns):
        raise KeyError(f"外部 CSV 必须包含列: {sorted(required_columns)}")
    duplicated = external[config.data.id_col].duplicated()
    if duplicated.any():
        duplicated_ids = (
            external.loc[duplicated, config.data.id_col].astype(str).tolist()
        )
        raise ValueError(f"外部 CSV 存在重复 id: {duplicated_ids[:5]}")
    aligned = sample_submission[[config.data.id_col]].merge(
        external[[config.data.id_col, config.data.label_col]],
        on=config.data.id_col,
        how="left",
        validate="one_to_one",
    )
    if aligned[config.data.label_col].isna().any():
        missing = aligned.loc[
            aligned[config.data.label_col].isna(), config.data.id_col
        ].astype(str).tolist()
        raise ValueError(f"外部 CSV 缺少 sample_submission 中的 id: {missing[:10]}")
    labels = aligned[config.data.label_col].astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("外部 CSV 的 label 必须为 0/1。")
    return labels


def save_external_submission_variants(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_labels: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame | None:
    """以 0.81818 强基线为锚点，保存可提交的融合候选。"""

    external_path = config.inference.external_label_csv
    if external_path is None or config.inference.external_strategy == "none":
        return None

    external_labels = load_external_labels(external_path, sample_submission, config)
    output_dir = config.experiment_output_dir
    id_col = config.data.id_col
    label_col = config.data.label_col

    baseline_submission = sample_submission[[id_col]].copy()
    baseline_submission[label_col] = external_labels.to_numpy(dtype=int)
    baseline_path = output_dir / "submission_external_baseline.csv"
    baseline_submission.to_csv(baseline_path, index=False)

    disagreement_mask = model_labels != external_labels.to_numpy(dtype=int)
    disagreements = sample_submission[[id_col]].copy()
    disagreements["model_probability"] = probabilities
    disagreements["model_prediction"] = model_labels
    disagreements["external_prediction"] = external_labels.to_numpy(dtype=int)
    disagreements["model_confidence_from_threshold"] = np.abs(
        probabilities - threshold
    )
    disagreements = disagreements.loc[disagreement_mask].sort_values(
        "model_confidence_from_threshold", ascending=False
    )
    disagreements.to_csv(output_dir / "external_disagreements.csv", index=False)

    if config.inference.external_strategy == "blend":
        weight = config.inference.external_blend_weight
        blended_probabilities = (
            (1.0 - weight) * probabilities
            + weight * external_labels.to_numpy(dtype=float)
        )
        final_labels = (blended_probabilities >= 0.5).astype(int)
        variant_name = "submission_external_blend.csv"
    else:
        margin = config.inference.external_override_margin
        confident_model = np.abs(probabilities - threshold) >= margin
        final_labels = external_labels.to_numpy(dtype=int).copy()
        replace_mask = disagreement_mask & confident_model
        final_labels[replace_mask] = model_labels[replace_mask]
        variant_name = "submission_external_guarded.csv"

    fused = sample_submission[[id_col]].copy()
    fused[label_col] = final_labels
    fused_path = output_dir / variant_name
    fused.to_csv(fused_path, index=False)
    print(
        f"外部强基线已接入: {external_path}; "
        f"strategy={config.inference.external_strategy}; "
        f"baseline_score={config.inference.external_label_score}; "
        f"disagreements={int(disagreement_mask.sum())}; saved={fused_path}"
    )
    return fused



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
    """强制从训练阶段生成的 OOF 阈值文件读取判定阈值。"""

    path_value = metadata.get("optimal_threshold_file")
    threshold_path = (
        Path(str(path_value))
        if path_value
        else config.experiment_output_dir / "optimal_threshold.json"
    )
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"找不到 OOF 最优阈值文件: {threshold_path}。请先用当前管线完成训练。"
        )
    with threshold_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("experiment_name") != metadata.get("experiment_name"):
        raise ValueError("阈值文件与 checkpoint 元数据的实验名不一致。")
    threshold = float(payload["optimal_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"OOF 阈值非法: {threshold}")
    return threshold


def run_prediction(
    config: ExperimentConfig,
    checkpoint_dir: Path,
) -> None:
    """平均各折预测概率并以持久化 OOF 阈值输出提交文件。"""

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

    fused_submission = save_external_submission_variants(
        sample_submission=sample_submission,
        probabilities=probabilities,
        threshold=threshold,
        model_labels=model_labels,
        config=config,
    )
    submission = fused_submission if fused_submission is not None else model_submission
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
