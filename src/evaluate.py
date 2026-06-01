"""Evaluate preprocessing mask success rates on a labeled dataset."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
WARNING_GAP = 0.10


@dataclass(frozen=True)
class EvaluationCounts:
    """Masking outcomes split by binary label."""

    success_label_1: int
    success_label_0: int
    failed_label_1: int
    failed_label_0: int

    @property
    def total_success(self) -> int:
        return self.success_label_1 + self.success_label_0

    @property
    def total_failed(self) -> int:
        return self.failed_label_1 + self.failed_label_0

    @property
    def total_processed(self) -> int:
        return self.total_success + self.total_failed


def _absolute_csv_file(value: str) -> Path:
    """Validate the absolute CSV path supplied on the command line."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--csv_path 必须是绝对路径。")
    path = path.resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"找不到 CSV 文件: {path}")
    return path


def _result_directory(value: str) -> Path:
    """Validate the masking output root supplied on the command line."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"找不到掩码输出目录: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="评估预处理器在训练集上的掩码成功率。"
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        type=_absolute_csv_file,
        help="train_labels.csv 的绝对路径",
    )
    parser.add_argument(
        "--result_dir",
        required=True,
        type=_result_directory,
        help="掩码输出根目录，其中应包含 images/ 和 failed/",
    )
    return parser.parse_args(argv)


def load_label_mapping(csv_path: Path) -> dict[str, int]:
    """Load and validate the image ID to binary label mapping."""
    try:
        frame = pd.read_csv(csv_path, dtype={"id": "string"})
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"无法读取 CSV 文件 {csv_path}: {exc}") from exc

    required_columns = {"id", "label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV 缺少必要列: {names}")

    if frame["id"].isna().any():
        raise ValueError("CSV 的 id 列包含空值。")

    image_ids = frame["id"].str.strip()
    if image_ids.eq("").any():
        raise ValueError("CSV 的 id 列包含空字符串。")
    duplicate_ids = image_ids[image_ids.duplicated()].unique().tolist()
    if duplicate_ids:
        preview = ", ".join(str(image_id) for image_id in duplicate_ids[:5])
        raise ValueError(f"CSV 包含重复图片 ID: {preview}")

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    invalid_labels = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    if invalid_labels.any():
        invalid_values = frame.loc[invalid_labels, "label"].astype(str).unique().tolist()
        preview = ", ".join(invalid_values[:5])
        raise ValueError(f"CSV 的 label 列只能包含 0 或 1，发现非法值: {preview}")

    return dict(zip(image_ids.tolist(), numeric_labels.astype(int).tolist(), strict=True))


def _iter_images(directory: Path) -> Iterator[Path]:
    """Yield supported image files recursively in deterministic order."""
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _collect_labels(
    directory: Path,
    label_mapping: dict[str, int],
    *,
    outcome_name: str,
) -> tuple[dict[str, int], int]:
    """Collect labels for one outcome directory and count ignored files."""
    labels: dict[str, int] = {}
    unknown_ids: list[str] = []

    for image_path in _iter_images(directory):
        image_id = image_path.name
        if image_id in labels:
            raise ValueError(f"{outcome_name} 目录包含重复图片文件名: {image_id}")
        try:
            labels[image_id] = label_mapping[image_id]
        except KeyError:
            unknown_ids.append(image_id)

    if unknown_ids:
        preview = ", ".join(unknown_ids[:5])
        raise ValueError(f"{outcome_name} 目录中存在 CSV 未记录的图片 ID: {preview}")

    ignored_count = sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES
    )
    return labels, ignored_count


def evaluate(csv_path: Path, result_dir: Path) -> tuple[EvaluationCounts, int]:
    """Calculate masking outcome counts from a CSV and output directories."""
    images_dir = result_dir / "images"
    failed_dir = result_dir / "failed"
    missing_directories = [
        str(directory) for directory in (images_dir, failed_dir) if not directory.is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(f"找不到必要目录: {', '.join(missing_directories)}")

    label_mapping = load_label_mapping(csv_path)
    success_labels, ignored_success = _collect_labels(
        images_dir,
        label_mapping,
        outcome_name="images",
    )
    failed_labels, ignored_failed = _collect_labels(
        failed_dir,
        label_mapping,
        outcome_name="failed",
    )

    overlapping_ids = sorted(success_labels.keys() & failed_labels.keys())
    if overlapping_ids:
        preview = ", ".join(overlapping_ids[:5])
        raise ValueError(f"以下图片同时出现在 images 和 failed 目录中: {preview}")

    counts = EvaluationCounts(
        success_label_1=sum(label == 1 for label in success_labels.values()),
        success_label_0=sum(label == 0 for label in success_labels.values()),
        failed_label_1=sum(label == 1 for label in failed_labels.values()),
        failed_label_0=sum(label == 0 for label in failed_labels.values()),
    )
    return counts, ignored_success + ignored_failed


def _format_rate(success: int, total: int) -> str:
    """Render a readable success rate, including its numerator and denominator."""
    if total == 0:
        return f"N/A ({success}/{total})"
    return f"{success / total:.2%} ({success}/{total})"


def print_report(counts: EvaluationCounts, ignored_count: int = 0) -> None:
    """Print a concise masking evaluation report."""
    label_1_total = counts.success_label_1 + counts.failed_label_1
    label_0_total = counts.success_label_0 + counts.failed_label_0

    print("\n" + "=" * 72)
    print("训练集掩码评估报告")
    print("=" * 72)
    print(f"总处理图片数                  : {counts.total_processed}")
    print(
        "整体掩码成功率 (Total Success Rate): "
        f"{_format_rate(counts.total_success, counts.total_processed)}"
    )
    print(
        "陨石 (Label 1) 掩码成功率        : "
        f"{_format_rate(counts.success_label_1, label_1_total)}"
    )
    print(
        "非陨石 (Label 0) 掩码成功率      : "
        f"{_format_rate(counts.success_label_0, label_0_total)}"
    )
    print("-" * 72)
    print(
        f"成功: Label 1 = {counts.success_label_1}, "
        f"Label 0 = {counts.success_label_0}, 合计 = {counts.total_success}"
    )
    print(
        f"失败: Label 1 = {counts.failed_label_1}, "
        f"Label 0 = {counts.failed_label_0}, 合计 = {counts.total_failed}"
    )
    if ignored_count:
        print(f"提示: 已忽略 {ignored_count} 个非图片文件。")
    print("=" * 72)

    if label_1_total and label_0_total:
        label_1_rate = counts.success_label_1 / label_1_total
        label_0_rate = counts.success_label_0 / label_0_total
        if label_1_rate <= label_0_rate - WARNING_GAP:
            print(
                "\n⚠️ 警告：模型对陨石的召回率显著低于普通干扰石，"
                "存在过滤掉过多正样本的风险！"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI evaluator."""
    args = parse_args(argv)
    try:
        counts, ignored_count = evaluate(args.csv_path, args.result_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"❌ 评估失败: {exc}", file=sys.stderr)
        return 1

    print_report(counts, ignored_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
