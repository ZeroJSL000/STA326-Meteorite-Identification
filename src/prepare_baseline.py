"""Assemble a clean baseline dataset from successfully masked images."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
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


def existing_file(value: str) -> Path:
    """Resolve and validate a CLI input file."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"文件不存在: {path}")
    return path


def existing_directory(value: str) -> Path:
    """Resolve and validate a CLI input directory."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"目录不存在: {path}")
    return path


def _find_masked_images(masked_dir: Path) -> dict[str, Path]:
    """Return successful masked images indexed by file name."""
    images = {
        image_path.name: image_path
        for image_path in sorted(masked_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not images:
        raise ValueError(f"掩码图片目录中没有可用图片: {masked_dir}")
    return images


def _load_excluded_image_ids(exclude_file: Path | None) -> set[str]:
    """Load manually excluded image IDs, allowing names or bare stems."""
    if exclude_file is None:
        return set()

    excluded_ids: set[str] = set()
    for raw_line in exclude_file.read_text(encoding="utf-8").splitlines():
        image_id = raw_line.partition("#")[0].strip()
        if image_id:
            excluded_ids.add(Path(image_id).name)
    return excluded_ids


def _load_filtered_labels(
    orig_csv: Path,
    available_image_names: set[str],
    excluded_image_ids: set[str],
) -> pd.DataFrame:
    """Load labels and keep only rows backed by successful masked images."""
    labels = pd.read_csv(orig_csv, dtype={"id": str}, keep_default_na=False)
    required_columns = {"id", "label"}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"标签文件缺少必要列: {missing}")

    if (labels["id"] == "").any():
        raise ValueError("标签文件的 id 列包含空值")

    is_available = labels["id"].isin(available_image_names)
    is_excluded = labels["id"].map(
        lambda image_id: image_id in excluded_image_ids
        or Path(image_id).stem in excluded_image_ids
    )
    filtered = labels[is_available & ~is_excluded].copy()
    if filtered.empty:
        raise ValueError("没有掩码图片能与标签文件中的 id 匹配")

    duplicate_ids = filtered.loc[filtered["id"].duplicated(), "id"].unique()
    if len(duplicate_ids) > 0:
        examples = ", ".join(str(image_id) for image_id in duplicate_ids[:5])
        raise ValueError(f"过滤后的标签文件包含重复 id: {examples}")

    numeric_labels = pd.to_numeric(filtered["label"], errors="raise")
    invalid_labels = sorted(set(numeric_labels).difference({0, 1}))
    if invalid_labels:
        raise ValueError(f"label 列只能包含 0 或 1，发现: {invalid_labels}")
    if numeric_labels.isna().any():
        raise ValueError("标签文件的 label 列包含空值")

    return filtered


def _reset_output_directory(train_images_dir: Path) -> None:
    """Create an empty training image directory without following symlinks."""
    if train_images_dir.is_symlink():
        raise ValueError(f"拒绝清理符号链接目录: {train_images_dir}")
    if train_images_dir.exists():
        if not train_images_dir.is_dir():
            raise NotADirectoryError(f"输出路径不是目录: {train_images_dir}")
        shutil.rmtree(train_images_dir)
    train_images_dir.mkdir(parents=True)


def assemble_baseline_dataset(
    *,
    orig_csv: Path,
    masked_dir: Path,
    out_root: Path,
    exclude_file: Path | None = None,
) -> tuple[int, int, int]:
    """Build the baseline dataset and return image and binary-label counts."""
    masked_images = _find_masked_images(masked_dir)
    excluded_image_ids = _load_excluded_image_ids(exclude_file)
    filtered_labels = _load_filtered_labels(
        orig_csv,
        set(masked_images),
        excluded_image_ids,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    train_images_dir = out_root / "train_images"
    output_csv = out_root / "train_labels.csv"
    _reset_output_directory(train_images_dir)

    if output_csv.exists() and not output_csv.is_file():
        raise ValueError(f"标签输出路径不是普通文件: {output_csv}")

    try:
        for image_name in filtered_labels["id"]:
            shutil.copy2(masked_images[image_name], train_images_dir / image_name)
        filtered_labels.to_csv(output_csv, index=False)
    except (OSError, ValueError):
        shutil.rmtree(train_images_dir, ignore_errors=True)
        output_csv.unlink(missing_ok=True)
        raise

    label_values = pd.to_numeric(filtered_labels["label"])
    image_count = len(filtered_labels)
    label_one_count = int((label_values == 1).sum())
    label_zero_count = int((label_values == 0).sum())
    return image_count, label_one_count, label_zero_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="使用成功生成的掩码图片组装纯净 Baseline 数据集",
    )
    parser.add_argument(
        "--orig_csv",
        required=True,
        type=existing_file,
        help="原始训练集标签 CSV 路径",
    )
    parser.add_argument(
        "--masked_dir",
        required=True,
        type=existing_directory,
        help="成功预处理的掩码图片目录",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("preData/cascade_dataset"),
        help="组装后的数据集根目录，默认为 preData/cascade_dataset",
    )
    parser.add_argument(
        "--exclude_file",
        type=existing_file,
        help="可选的人工毒样本排除列表，每行填写图片文件名或纯编号",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline dataset assembly CLI."""
    args = parse_args(argv)
    out_root = args.out_root.expanduser().resolve()
    try:
        image_count, label_one_count, label_zero_count = assemble_baseline_dataset(
            orig_csv=args.orig_csv,
            masked_dir=args.masked_dir,
            out_root=out_root,
            exclude_file=args.exclude_file,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"组装失败: {error}") from error

    print("Baseline 数据集组装完成")
    print(f"成功组装图片数量: {image_count}")
    print(f"Label 1 的数量: {label_one_count}")
    print(f"Label 0 的数量: {label_zero_count}")
    print(f"目标数据集路径: {out_root}")


if __name__ == "__main__":
    main()
