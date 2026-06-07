"""以 F1-Score 为核心的验证指标与阈值搜索。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class BinaryMetrics:
    """保存二分类决策阈值对应的关键分数。"""

    f1: float
    precision: float
    recall: float
    threshold: float
    positive_predictions: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """稳定地将 logit 转成正类概率。"""

    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    """使用给定阈值计算 F1、精确率和召回率。"""

    predictions = (probabilities >= threshold).astype(np.int64)
    return BinaryMetrics(
        f1=float(f1_score(targets, predictions, zero_division=0)),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        recall=float(recall_score(targets, predictions, zero_division=0)),
        threshold=float(threshold),
        positive_predictions=int(predictions.sum()),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.40,
    threshold_max: float = 0.60,
    step: float = 0.01,
) -> BinaryMetrics:
    """在固定网格上搜索 F1 最大的概率阈值。

    固定搜索网格避免通过测试集类别数施加人为约束；本轮实验将搜索区间
    收缩到 0.5 附近，减少验证泄露或 OOF 波动造成的极端阈值。
    """

    targets = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if targets.size == 0 or targets.shape != probabilities.shape:
        raise ValueError("阈值搜索需要长度相同的非空标签与概率数组。")
    if np.unique(targets).size < 2:
        raise ValueError("阈值搜索需要同时包含正负样本。")
    if not 0.0 <= threshold_min < threshold_max <= 1.0 or step <= 0:
        raise ValueError("阈值区间或步长非法。")

    candidates = np.round(
        np.arange(threshold_min, threshold_max + step / 2.0, step),
        decimals=10,
    )
    best = compute_binary_metrics(targets, probabilities, float(candidates[0]))
    for threshold in candidates[1:]:
        current = compute_binary_metrics(targets, probabilities, float(threshold))
        if current.f1 > best.f1:
            best = current
        elif np.isclose(current.f1, best.f1) and abs(current.threshold - 0.5) < abs(
            best.threshold - 0.5
        ):
            best = current
    return best


def optimize_f1_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> BinaryMetrics:
    """兼容现有调用的 F1 阈值搜索入口。"""

    return find_optimal_threshold(targets, probabilities)
