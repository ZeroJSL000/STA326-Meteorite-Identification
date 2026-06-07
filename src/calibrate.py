"""使用 OOF 概率拟合温度，并执行合法的 label-shift 全局校准。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, precision_score, recall_score

from ensemble import _normalize_probability_frame
from utils.config import CalibrationConfig, ExperimentConfig, load_config
from utils.metrics import BinaryMetrics, sigmoid


def parse_args() -> argparse.Namespace:
    """允许仅使用 YAML 运行，也保留显式路径覆盖入口。"""

    parser = argparse.ArgumentParser(description="在 OOF 概率上拟合温度并生成校准提交")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oof", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _probabilities_to_logits(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    """从 sigmoid 概率稳定反推 raw logit。"""

    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        epsilon,
        1.0 - epsilon,
    )
    return np.log(clipped / (1.0 - clipped))


def _negative_log_likelihood(
    labels: np.ndarray,
    probabilities: np.ndarray,
    epsilon: float,
) -> float:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(
            labels * np.log(clipped)
            + (1.0 - labels) * np.log(1.0 - clipped)
        )
    )


def fit_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibration: CalibrationConfig,
) -> tuple[float, float, float]:
    """在 raw logit 上最小化 OOF NLL；关闭开关时保持 T=1。"""

    labels = np.asarray(labels, dtype=np.float64)
    logits = _probabilities_to_logits(probabilities, calibration.probability_epsilon)
    raw_nll = _negative_log_likelihood(
        labels,
        sigmoid(logits),
        calibration.probability_epsilon,
    )
    if not calibration.enabled or not calibration.temperature_scaling:
        return 1.0, raw_nll, raw_nll

    result = minimize_scalar(
        lambda temperature: _negative_log_likelihood(
            labels,
            sigmoid(logits / temperature),
            calibration.probability_epsilon,
        ),
        bounds=(calibration.temperature_min, calibration.temperature_max),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"温度缩放拟合失败: {result.message}")
    temperature = float(result.x)
    calibrated_nll = _negative_log_likelihood(
        labels,
        sigmoid(logits / temperature),
        calibration.probability_epsilon,
    )
    return temperature, raw_nll, calibrated_nll


def apply_calibration(
    probabilities: np.ndarray,
    temperature: float,
    config: ExperimentConfig,
) -> np.ndarray:
    """先执行温度缩放，再按配置执行 label-shift logit 修正。"""

    raw_probabilities = np.asarray(probabilities, dtype=np.float64)
    if not config.calibration.enabled:
        return raw_probabilities.copy()

    logits = _probabilities_to_logits(
        raw_probabilities,
        config.calibration.probability_epsilon,
    )
    if config.calibration.temperature_scaling:
        logits = logits / temperature
    if config.calibration.use_bayesian_correction:
        train_prior = config.calibration.train_pos_prior
        test_prior = config.calibration.prior_pos_rate
        train_log_odds = np.log(train_prior / (1.0 - train_prior))
        test_log_odds = np.log(test_prior / (1.0 - test_prior))
        logits = logits - train_log_odds + test_log_odds
    return sigmoid(logits)


def _compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    predictions = (probabilities > threshold).astype(np.int64)
    return BinaryMetrics(
        f1=float(f1_score(labels, predictions, zero_division=0)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        threshold=float(threshold),
        positive_predictions=int(predictions.sum()),
    )


def find_f1_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibration: CalibrationConfig,
) -> BinaryMetrics:
    """在配置化细粒度网格上选择 OOF F1 最优全局阈值。"""

    candidates = np.arange(
        0.0,
        1.0 + calibration.f1_threshold_step / 2.0,
        calibration.f1_threshold_step,
    )
    best = _compute_metrics(labels, probabilities, float(candidates[0]))
    for threshold in candidates[1:]:
        current = _compute_metrics(labels, probabilities, float(threshold))
        if current.f1 > best.f1:
            best = current
        elif np.isclose(current.f1, best.f1) and abs(
            current.threshold - calibration.threshold_tie_breaker
        ) < abs(best.threshold - calibration.threshold_tie_breaker):
            best = current
    return best


def find_prior_aligned_threshold(
    probabilities: np.ndarray,
    calibration: CalibrationConfig,
) -> float:
    """二分搜索单个全局阈值，使正类率尽量接近 YAML 先验。"""

    low = 0.0
    high = 1.0
    for _ in range(calibration.threshold_iterations):
        midpoint = (low + high) / 2.0
        if float(np.mean(probabilities > midpoint)) > calibration.prior_pos_rate:
            low = midpoint
        else:
            high = midpoint
    candidates = (low, high)
    return min(
        candidates,
        key=lambda threshold: abs(
            float(np.mean(probabilities > threshold))
            - calibration.prior_pos_rate
        ),
    )


def _select_final_metrics(
    f1_metrics: BinaryMetrics,
    prior_metrics: BinaryMetrics,
    calibration: CalibrationConfig,
) -> tuple[str, BinaryMetrics]:
    if calibration.threshold_strategy == "f1_optimal":
        return "f1_optimal", f1_metrics
    if calibration.threshold_strategy == "prior_aligned":
        return "prior_aligned", prior_metrics
    if f1_metrics.f1 >= prior_metrics.f1:
        return "f1_optimal", f1_metrics
    return "prior_aligned", prior_metrics


def _write_oof(
    path: Path,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    output = pd.DataFrame(
        {
            "image_name": frame["image_name"],
            "label": frame["label"],
            "prob": probabilities,
        }
    )
    output["probability"] = output["prob"]
    output["prediction"] = (output["prob"] > threshold).astype(np.int64)
    output.to_csv(path, index=False)


def _write_submission(
    output_dir: Path,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    probability_filename: str,
    submission_filename: str,
) -> None:
    probability_frame = pd.DataFrame(
        {
            "image_name": frame["image_name"],
            "prob": probabilities,
        }
    )
    probability_frame["probability"] = probability_frame["prob"]
    probability_frame["prediction"] = (
        probability_frame["prob"] > threshold
    ).astype(np.int64)
    probability_frame.to_csv(output_dir / probability_filename, index=False)
    probability_frame[["image_name", "prediction"]].rename(
        columns={"prediction": "label"}
    ).to_csv(output_dir / submission_filename, index=False)


def _resolve_input_path(
    cli_path: Path | None,
    yaml_path: Path | None,
    description: str,
) -> Path:
    path = cli_path or yaml_path
    if path is None:
        raise ValueError(f"缺少 {description} 路径，请在 ensemble 节点或 CLI 中配置。")
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {description}: {path}")
    return path


def run_calibration(
    config: ExperimentConfig,
    oof_path: Path,
    submission_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """拟合温度、应用修正、比较全局阈值并写出提交文件。"""

    oof = _normalize_probability_frame(oof_path, require_label=True)
    submission = _normalize_probability_frame(submission_path, require_label=False)
    labels = oof["label"].to_numpy(dtype=np.int64)
    oof_raw = oof["prob"].to_numpy(dtype=np.float64)
    submission_raw = submission["prob"].to_numpy(dtype=np.float64)

    if not config.calibration.enabled:
        baseline_metrics = _compute_metrics(labels, oof_raw, config.inference.threshold)
        nll = _negative_log_likelihood(
            labels,
            oof_raw,
            config.calibration.probability_epsilon,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_oof(
            output_dir / "calibrated_oof_predictions.csv",
            oof,
            oof_raw,
            baseline_metrics.threshold,
        )
        _write_submission(
            output_dir,
            submission,
            submission_raw,
            baseline_metrics.threshold,
            "submission_probabilities.csv",
            "submission.csv",
        )
        temperature_payload = {"T": 1.0, "nll_before": nll, "nll_after": nll}
        threshold_payload = {
            "f1_threshold": None,
            "prior_threshold": None,
            "final_threshold": baseline_metrics.threshold,
            "final_strategy": "calibration_disabled",
            "oof_f1_final": baseline_metrics.f1,
            "expected_pos_rate": None,
            "actual_pos_rate_prior_threshold": None,
            "calibration_enabled": False,
        }
        save_json(temperature_payload, output_dir / "temperature.json")
        save_json(threshold_payload, output_dir / "optimal_threshold.json")
        return {"temperature": temperature_payload, "threshold": threshold_payload}

    temperature, nll_before, nll_after = fit_temperature(
        labels,
        oof_raw,
        config.calibration,
    )
    oof_calibrated = apply_calibration(oof_raw, temperature, config)
    submission_calibrated = apply_calibration(submission_raw, temperature, config)

    f1_metrics = find_f1_optimal_threshold(
        labels,
        oof_calibrated,
        config.calibration,
    )
    oof_prior_threshold = find_prior_aligned_threshold(
        oof_calibrated,
        config.calibration,
    )
    prior_metrics = _compute_metrics(labels, oof_calibrated, oof_prior_threshold)
    selected_name, selected_metrics = _select_final_metrics(
        f1_metrics,
        prior_metrics,
        config.calibration,
    )

    # OOF 与测试概率分布可能不同，因此对比提交单独执行合法的全局阈值对齐。
    submission_prior_threshold = find_prior_aligned_threshold(
        submission_calibrated,
        config.calibration,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_oof(
        output_dir / "calibrated_oof_predictions.csv",
        oof,
        oof_calibrated,
        selected_metrics.threshold,
    )
    _write_submission(
        output_dir,
        submission,
        submission_calibrated,
        selected_metrics.threshold,
        "submission_probabilities.csv",
        "submission.csv",
    )
    _write_submission(
        output_dir,
        submission,
        submission_calibrated,
        submission_prior_threshold,
        "prior_aligned_submission_probabilities.csv",
        "prior_aligned_submission.csv",
    )

    temperature_payload = {
        "T": temperature,
        "nll_before": nll_before,
        "nll_after": nll_after,
    }
    threshold_payload = {
        "f1_threshold": f1_metrics.threshold,
        "prior_threshold": oof_prior_threshold,
        "final_threshold": selected_metrics.threshold,
        "final_strategy": selected_name,
        "oof_f1_final": selected_metrics.f1,
        "expected_pos_rate": config.calibration.prior_pos_rate,
        "actual_pos_rate_prior_threshold": float(
            np.mean(oof_calibrated > oof_prior_threshold)
        ),
        "f1_threshold_oof_f1": f1_metrics.f1,
        "prior_threshold_oof_f1": prior_metrics.f1,
        "submission_final_positive_predictions": int(
            np.sum(submission_calibrated > selected_metrics.threshold)
        ),
        "submission_final_positive_ratio": float(
            np.mean(submission_calibrated > selected_metrics.threshold)
        ),
        "submission_prior_threshold": submission_prior_threshold,
        "submission_prior_aligned_positive_predictions": int(
            np.sum(submission_calibrated > submission_prior_threshold)
        ),
        "submission_prior_aligned_positive_ratio": float(
            np.mean(submission_calibrated > submission_prior_threshold)
        ),
        "calibration_enabled": config.calibration.enabled,
        "temperature_scaling": config.calibration.temperature_scaling,
        "use_bayesian_correction": config.calibration.use_bayesian_correction,
        "train_pos_prior": config.calibration.train_pos_prior,
    }
    save_json(temperature_payload, output_dir / "temperature.json")
    save_json(threshold_payload, output_dir / "optimal_threshold.json")
    return {"temperature": temperature_payload, "threshold": threshold_payload}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    oof_path = _resolve_input_path(
        args.oof,
        config.ensemble.oof_predictions,
        "OOF 概率文件",
    )
    submission_path = _resolve_input_path(
        args.submission,
        config.ensemble.test_probabilities,
        "测试概率文件",
    )
    output_dir = args.output_dir or config.experiment_output_dir
    report = run_calibration(config, oof_path, submission_path, output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
