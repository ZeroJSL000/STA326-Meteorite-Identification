"""Assemble a complete masked test dataset with original-image fallbacks."""

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


def _index_images(directory: Path) -> dict[str, Path]:
    """Index supported images recursively and reject ambiguous file names."""
    if not directory.is_dir():
        raise FileNotFoundError(f"目录不存在: {directory}")

    images: dict[str, Path] = {}
    for image_path in sorted(directory.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.name in images:
            raise ValueError(f"目录中存在重复图片文件名: {image_path.name}")
        images[image_path.name] = image_path
    return images


def _load_test_ids(sample_submission_csv: Path) -> list[str]:
    """Load ordered test IDs from the submission template."""
    frame = pd.read_csv(sample_submission_csv, dtype={"id": str}, keep_default_na=False)
    if "id" not in frame:
        raise ValueError("提交模板缺少必要列: id")
    if (frame["id"] == "").any():
        raise ValueError("提交模板的 id 列包含空值")
    duplicate_ids = frame.loc[frame["id"].duplicated(), "id"].unique()
    if len(duplicate_ids) > 0:
        examples = ", ".join(str(image_id) for image_id in duplicate_ids[:5])
        raise ValueError(f"提交模板包含重复 id: {examples}")
    return frame["id"].tolist()


def _reset_output_directory(output_dir: Path) -> None:
    """Create an empty output directory without following symlinks."""
    if output_dir.is_symlink():
        raise ValueError(f"拒绝清理符号链接目录: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"输出路径不是目录: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def assemble_masked_test_dataset(
    *,
    sample_submission_csv: Path,
    source_dir: Path,
    result_dir: Path,
    output_dir: Path,
) -> tuple[int, int, int]:
    """Copy masks when available and original images for explicit failures."""
    test_ids = _load_test_ids(sample_submission_csv)
    source_images = _index_images(source_dir)
    masked_images = _index_images(result_dir / "images")
    failed_images = _index_images(result_dir / "failed")

    overlapping_ids = sorted(masked_images.keys() & failed_images.keys())
    if overlapping_ids:
        examples = ", ".join(overlapping_ids[:5])
        raise ValueError(f"以下图片同时存在于 images 和 failed: {examples}")

    selected_images: list[tuple[str, Path]] = []
    masked_count = 0
    fallback_count = 0
    missing_ids: list[str] = []
    for image_id in test_ids:
        if image_id in masked_images:
            selected_images.append((image_id, masked_images[image_id]))
            masked_count += 1
        elif image_id in failed_images and image_id in source_images:
            selected_images.append((image_id, source_images[image_id]))
            fallback_count += 1
        else:
            missing_ids.append(image_id)
    if missing_ids:
        examples = ", ".join(missing_ids[:5])
        raise ValueError(f"以下测试图片没有完整预处理结果: {examples}")

    _reset_output_directory(output_dir)
    try:
        for image_id, source_path in selected_images:
            shutil.copy2(source_path, output_dir / image_id)
    except OSError:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    return len(selected_images), masked_count, fallback_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="组装完整测试集：掩码成功使用掩码图，失败使用原图",
    )
    parser.add_argument(
        "--sample_submission_csv",
        required=True,
        type=existing_file,
        help="测试集提交模板 CSV",
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        type=existing_directory,
        help="原始测试图片目录",
    )
    parser.add_argument(
        "--result_dir",
        required=True,
        type=existing_directory,
        help="测试集 cascade 输出根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("preData/cascade_dataset/test_images"),
        help="完整推理图片输出目录",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the masked test dataset assembly CLI."""
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        total_count, masked_count, fallback_count = assemble_masked_test_dataset(
            sample_submission_csv=args.sample_submission_csv,
            source_dir=args.source_dir,
            result_dir=args.result_dir,
            output_dir=output_dir,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"测试集组装失败: {error}") from error

    print("测试集掩码目录组装完成")
    print(f"总图片数量: {total_count}")
    print(f"使用掩码图片: {masked_count}")
    print(f"失败回退原图: {fallback_count}")
    print(f"推理图片目录: {output_dir}")


if __name__ == "__main__":
    main()
