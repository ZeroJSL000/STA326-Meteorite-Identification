"""基于上一轮 OOF 误差的困难样本重采样。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(f"OOF 文件缺少{name}列，候选字段: {candidates}")


def create_weighted_sampler(
    oof_csv_path: str | Path,
    topk: float = 0.1,
    boost: float = 3.0,
    sample_ids: Sequence[str] | None = None,
    id_col: str | None = None,
    label_col: str = "label",
) -> WeightedRandomSampler:
    """根据 OOF BCE loss 提升两档困难样本的抽样权重。"""

    if not 0.0 < topk <= 0.5:
        raise ValueError("hard_mining_topk 必须在 (0, 0.5] 范围内。")
    if boost <= 0.0:
        raise ValueError("hard_mining_boost 必须大于 0。")
    path = Path(oof_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到困难样本 OOF 文件: {path}")

    frame = pd.read_csv(path)
    resolved_id_col = id_col or _find_column(frame, ("image_name", "id"), "图像 ID")
    probability_col = _find_column(frame, ("prob", "probability"), "概率")
    required_columns = {resolved_id_col, label_col, probability_col}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise KeyError(f"OOF 文件缺少列: {sorted(missing_columns)}")
    frame = frame[[resolved_id_col, label_col, probability_col]].copy()
    frame[resolved_id_col] = frame[resolved_id_col].astype(str)
    if frame[resolved_id_col].duplicated().any():
        raise ValueError("OOF 文件包含重复图像 ID，无法进行困难样本对齐。")
    if sample_ids is not None:
        normalized_ids = [str(image_id) for image_id in sample_ids]
        indexed = frame.set_index(resolved_id_col)
        missing_ids = sorted(set(normalized_ids) - set(indexed.index))
        if missing_ids:
            raise KeyError(f"OOF 文件缺少训练样本 ID: {missing_ids[:5]}")
        frame = indexed.loc[normalized_ids].reset_index()

    labels = frame[label_col].astype(np.float64).to_numpy()
    probabilities = frame[probability_col].astype(np.float64).to_numpy()
    if not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("OOF 标签必须为 0/1。")
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF 概率必须为有限值。")
    epsilon = np.finfo(np.float64).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    losses = -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))

    order = np.argsort(losses)[::-1]
    sample_count = len(frame)
    hard_count = max(1, int(np.ceil(sample_count * topk)))
    medium_count = max(hard_count, int(np.ceil(sample_count * min(2.0 * topk, 1.0))))
    weights = np.ones(sample_count, dtype=np.float64)
    weights[order[:hard_count]] *= boost
    weights[order[hard_count:medium_count]] *= boost / 1.5
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=sample_count,
        replacement=True,
    )
