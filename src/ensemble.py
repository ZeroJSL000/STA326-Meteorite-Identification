"""对齐多个实验的 OOF 与测试概率，并输出三种合法集成结果。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.metrics import BinaryMetrics, find_optimal_threshold


@dataclass
class ExperimentPredictions:
    """保存单实验对齐前的 OOF、测试预测和 OOF 指标。"""

    directory: Path
    name: str
    oof: pd.DataFrame
    submission: pd.DataFrame
    oof_metrics: BinaryMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="集成多个二分类图像实验")
    parser.add_argument("--dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble"))
    parser.add_argument("--exponent-scale", type=float, default=5.0)
    parser.add_argument("--stacking-c", type=float, default=1.0)
    parser.add_argument("--stacking-max-iter", type=int, default=1000)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"概率文件缺少{name}列，候选字段: {candidates}")


def _normalize_probability_frame(path: Path, require_label: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到概率文件: {path}")
    frame = pd.read_csv(path)
    id_col = _find_column(frame, ("image_name", "id"), "图像 ID")
    probability_col = _find_column(frame, ("prob", "probability"), "概率")
    columns = [id_col, probability_col]
    if require_label:
        if "label" not in frame:
            raise KeyError(f"OOF 文件缺少 label 列: {path}")
        columns.append("label")
    normalized = frame[columns].copy()
    normalized = normalized.rename(columns={id_col: "image_name", probability_col: "prob"})
    normalized["image_name"] = normalized["image_name"].astype(str)
    normalized["prob"] = normalized["prob"].astype(np.float64)
    if not np.isfinite(normalized["prob"]).all():
        raise ValueError(f"概率文件包含非有限值: {path}")
    if not normalized["prob"].between(0.0, 1.0).all():
        raise ValueError(f"概率文件包含 [0, 1] 之外的值: {path}")
    if require_label:
        normalized["label"] = normalized["label"].astype(np.int64)
        label_counts = normalized.groupby("image_name", sort=False)["label"].nunique()
        if (label_counts > 1).any():
            raise ValueError(f"OOF 文件同一图像存在冲突标签: {path}")
        return (
            normalized.groupby("image_name", sort=False, as_index=False)
            .agg(label=("label", "first"), prob=("prob", "mean"))
        )
    return normalized.groupby("image_name", sort=False, as_index=False).agg(prob=("prob", "mean"))


def _find_metrics(labels: np.ndarray, probabilities: np.ndarray, args: argparse.Namespace) -> BinaryMetrics:
    return find_optimal_threshold(
        labels,
        probabilities,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        step=args.threshold_step,
    )


def load_experiment(directory: Path, args: argparse.Namespace) -> ExperimentPredictions:
    oof = _normalize_probability_frame(directory / "oof_predictions.csv", require_label=True)
    submission = _normalize_probability_frame(
        directory / "submission_probabilities.csv",
        require_label=False,
    )
    metrics = _find_metrics(oof["label"].to_numpy(), oof["prob"].to_numpy(), args)
    return ExperimentPredictions(
        directory=directory,
        name=directory.name,
        oof=oof,
        submission=submission,
        oof_metrics=metrics,
    )


def _align_experiments(
    experiments: list[ExperimentPredictions],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if len(experiments) < 2:
        raise ValueError("集成至少需要两个实验目录。")
    reference_oof = experiments[0].oof
    reference_submission = experiments[0].submission
    oof_ids = reference_oof["image_name"].to_numpy()
    submission_ids = reference_submission["image_name"].to_numpy()
    labels = reference_oof["label"].to_numpy(dtype=np.int64)
    oof_features: list[np.ndarray] = []
    submission_features: list[np.ndarray] = []
    for experiment in experiments:
        if set(experiment.oof["image_name"]) != set(oof_ids):
            raise ValueError(f"实验 {experiment.name} 的 OOF 图像集合不一致。")
        if set(experiment.submission["image_name"]) != set(submission_ids):
            raise ValueError(f"实验 {experiment.name} 的测试图像集合不一致。")
        aligned_oof = experiment.oof.set_index("image_name").loc[oof_ids]
        aligned_submission = experiment.submission.set_index("image_name").loc[submission_ids]
        if not np.array_equal(aligned_oof["label"].to_numpy(dtype=np.int64), labels):
            raise ValueError(f"实验 {experiment.name} 的 OOF 标签不一致。")
        oof_features.append(aligned_oof["prob"].to_numpy(dtype=np.float64))
        submission_features.append(aligned_submission["prob"].to_numpy(dtype=np.float64))
    return (
        oof_ids,
        submission_ids,
        labels,
        np.column_stack(oof_features),
        np.column_stack(submission_features),
    )


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("集成权重必须为正有限值。")
    return values / values.sum()


def _rank_features(oof_features: np.ndarray, submission_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof_rank_columns: list[np.ndarray] = []
    submission_rank_columns: list[np.ndarray] = []
    for column in range(oof_features.shape[1]):
        oof_column = oof_features[:, column]
        submission_column = submission_features[:, column]
        oof_ranks = pd.Series(oof_column).rank(method="average", pct=True).to_numpy()
        sorted_oof = np.sort(oof_column)
        submission_ranks = np.searchsorted(sorted_oof, submission_column, side="right") / len(sorted_oof)
        oof_rank_columns.append(oof_ranks)
        submission_rank_columns.append(submission_ranks)
    return np.column_stack(oof_rank_columns), np.column_stack(submission_rank_columns)


def _write_probabilities(
    path: Path,
    image_names: np.ndarray,
    probabilities: np.ndarray,
    metrics: BinaryMetrics,
) -> None:
    frame = pd.DataFrame({"image_name": image_names, "prob": probabilities})
    frame["probability"] = frame["prob"]
    frame["prediction"] = (frame["prob"] >= metrics.threshold).astype(np.int64)
    frame.to_csv(path, index=False)


def _write_oof_probabilities(
    path: Path,
    image_names: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: BinaryMetrics,
) -> None:
    frame = pd.DataFrame({"image_name": image_names, "label": labels, "prob": probabilities})
    frame["probability"] = frame["prob"]
    frame["prediction"] = (frame["prob"] >= metrics.threshold).astype(np.int64)
    frame.to_csv(path, index=False)


def run_ensemble(args: argparse.Namespace) -> dict[str, object]:
    experiments = [load_experiment(directory, args) for directory in args.dirs]
    oof_ids, submission_ids, labels, oof_features, submission_features = _align_experiments(experiments)

    model_f1 = np.array([experiment.oof_metrics.f1 for experiment in experiments])
    if args.weights is None:
        blend_weights = _normalize_weights(np.exp(args.exponent_scale * model_f1))
        weight_source = "exp_oof_f1"
    else:
        if len(args.weights) != len(experiments):
            raise ValueError("--weights 数量必须与 --dirs 数量一致。")
        blend_weights = _normalize_weights(np.asarray(args.weights, dtype=np.float64))
        weight_source = "manual"
    weighted_oof = oof_features @ blend_weights
    weighted_submission = submission_features @ blend_weights
    weighted_metrics = _find_metrics(labels, weighted_oof, args)

    stacker = LogisticRegression(C=args.stacking_c, max_iter=args.stacking_max_iter)
    stacker.fit(oof_features, labels)
    stacking_oof = stacker.predict_proba(oof_features)[:, 1]
    stacking_submission = stacker.predict_proba(submission_features)[:, 1]
    stacking_metrics = _find_metrics(labels, stacking_oof, args)

    oof_rank_features, submission_rank_features = _rank_features(oof_features, submission_features)
    rank_oof = oof_rank_features.mean(axis=1)
    rank_submission = submission_rank_features.mean(axis=1)
    rank_metrics = _find_metrics(labels, rank_oof, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_probabilities(args.output_dir / "weighted_blend.csv", submission_ids, weighted_submission, weighted_metrics)
    _write_probabilities(args.output_dir / "stacking.csv", submission_ids, stacking_submission, stacking_metrics)
    _write_probabilities(args.output_dir / "rank_avg.csv", submission_ids, rank_submission, rank_metrics)
    _write_oof_probabilities(
        args.output_dir / "weighted_blend_oof_predictions.csv",
        oof_ids,
        labels,
        weighted_oof,
        weighted_metrics,
    )
    _write_oof_probabilities(
        args.output_dir / "stacking_oof_predictions.csv",
        oof_ids,
        labels,
        stacking_oof,
        stacking_metrics,
    )
    _write_oof_probabilities(
        args.output_dir / "rank_avg_oof_predictions.csv",
        oof_ids,
        labels,
        rank_oof,
        rank_metrics,
    )
    log = {
        "models": [
            {
                "name": experiment.name,
                "directory": str(experiment.directory),
                "oof_f1": experiment.oof_metrics.f1,
                "oof_threshold": experiment.oof_metrics.threshold,
                "blend_weight": float(weight),
            }
            for experiment, weight in zip(experiments, blend_weights, strict=True)
        ],
        "weight_source": weight_source,
        "strategies": {
            "weighted_blend": weighted_metrics.to_dict(),
            "stacking": stacking_metrics.to_dict(),
            "rank_avg": rank_metrics.to_dict(),
        },
        "stacking": {
            "c": args.stacking_c,
            "max_iter": args.stacking_max_iter,
            "coefficients": stacker.coef_.tolist(),
            "intercept": stacker.intercept_.tolist(),
        },
        "oof_rows": len(oof_ids),
        "submission_rows": len(submission_ids),
    }
    with (args.output_dir / "ensemble_log.json").open("w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2)
    return log


def main() -> None:
    args = parse_args()
    log = run_ensemble(args)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
