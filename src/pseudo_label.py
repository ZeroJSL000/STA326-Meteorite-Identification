"""根据两个强模型的测试概率生成严格筛选的 co-teaching 伪标签。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import _normalize_probability_frame
from utils.config import PROJECT_ROOT, ExperimentConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 co-teaching 高置信伪标签")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dirs", nargs="+", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve_directory(directory: str | Path) -> Path:
    path = Path(directory)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_submission(directory: Path) -> pd.DataFrame:
    return _normalize_probability_frame(
        directory / "submission_probabilities.csv",
        require_label=False,
    )


def generate_pseudo_labels(
    config: ExperimentConfig,
    model_dirs: list[Path] | None = None,
) -> pd.DataFrame:
    """按高低置信阈值筛选伪标签，模糊样本直接丢弃。"""

    configured_dirs = config.pseudo_labeling.co_teaching_model_dirs
    directories = model_dirs or [_resolve_directory(directory) for directory in configured_dirs]
    if not directories:
        raise ValueError("至少需要一个伪标签模型目录。")
    if config.pseudo_labeling.co_teaching_enabled and len(directories) < 2:
        raise ValueError("启用 co-teaching 时至少需要两个模型目录。")

    model_a = _load_submission(directories[0])
    model_a = model_a.rename(columns={"prob": "prob_a"})
    aligned = model_a
    if config.pseudo_labeling.co_teaching_enabled:
        model_b = _load_submission(directories[1])
        model_b = model_b.rename(columns={"prob": "prob_b"})
        if set(model_a["image_name"]) != set(model_b["image_name"]):
            raise ValueError("Co-teaching 模型的测试图像集合不一致。")
        aligned = model_a.merge(model_b, on="image_name", how="inner", validate="one_to_one")

    high = config.pseudo_labeling.high_confidence_threshold
    low = config.pseudo_labeling.low_confidence_threshold
    agreement = config.pseudo_labeling.co_teaching_agreement_threshold
    if config.pseudo_labeling.co_teaching_enabled:
        positive_mask = (aligned["prob_a"] > high) & (aligned["prob_b"] > agreement)
        negative_mask = (aligned["prob_a"] < low) & (aligned["prob_b"] < 1.0 - agreement)
        positive_confidence = np.minimum(aligned["prob_a"], aligned["prob_b"])
        negative_confidence = np.minimum(1.0 - aligned["prob_a"], 1.0 - aligned["prob_b"])
    else:
        positive_mask = aligned["prob_a"] > high
        negative_mask = aligned["prob_a"] < low
        positive_confidence = aligned["prob_a"]
        negative_confidence = 1.0 - aligned["prob_a"]

    positive = pd.DataFrame(
        {
            "image_name": aligned.loc[positive_mask, "image_name"],
            "label": 1,
            "confidence": positive_confidence[positive_mask],
            "is_pseudo": 1,
        }
    )
    negative = pd.DataFrame(
        {
            "image_name": aligned.loc[negative_mask, "image_name"],
            "label": 0,
            "confidence": negative_confidence[negative_mask],
            "is_pseudo": 1,
        }
    )
    return pd.concat([positive, negative], ignore_index=True).sort_values("image_name")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_dirs = [_resolve_directory(directory) for directory in args.model_dirs] if args.model_dirs else None
    output_path = args.output or config.paths.pseudo_labels_csv
    pseudo_labels = generate_pseudo_labels(config, model_dirs=model_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_labels.to_csv(output_path, index=False)
    positive_count = int(pseudo_labels["label"].sum()) if len(pseudo_labels) else 0
    negative_count = len(pseudo_labels) - positive_count
    print(f"伪标签已保存: {output_path}; positive={positive_count}; negative={negative_count}; total={len(pseudo_labels)}")


if __name__ == "__main__":
    main()
