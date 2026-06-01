"""Preprocessing entry point for isolated image outputs."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRE_DATA_DIR = PROJECT_ROOT / "preData"
MASK_TASK_PREFIX = "mask_"


@dataclass(frozen=True)
class MaskOutputDirectories:
    """Output directories owned by one masking method."""

    root: Path
    images: Path
    visualizations: Path
    failed: Path


class Masker(ABC):
    """Interface implemented by concrete masking methods."""

    @abstractmethod
    def run(self, *, data_dir: Path, output_dirs: MaskOutputDirectories) -> None:
        """Process images from data_dir and write only to output_dirs."""


MaskerFactory = Callable[[], Masker]
MASKER_FACTORIES: dict[str, MaskerFactory] = {}


def setup_directories(
    method_name: str,
    output_root: Path | None = None,
) -> MaskOutputDirectories:
    """Create isolated output directories for one masking method."""
    method_path = Path(method_name)
    if (
        not method_name
        or method_path.is_absolute()
        or len(method_path.parts) != 1
        or method_name in {".", ".."}
    ):
        raise ValueError(f"Invalid masking method name: {method_name!r}")

    root = output_root or PRE_DATA_DIR / method_name
    output_dirs = MaskOutputDirectories(
        root=root,
        images=root / "images",
        visualizations=root / "visualizations",
        failed=root / "failed",
    )
    for directory in (
        output_dirs.images,
        output_dirs.visualizations,
        output_dirs.failed,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def register_masker(method_name: str, factory: MaskerFactory) -> None:
    """Register a concrete masker without coupling it to the CLI."""
    MASKER_FACTORIES[method_name] = factory


def run_masking(
    method_name: str,
    data_dir: Path,
    output_root: Path | None = None,
) -> None:
    """Initialize outputs and dispatch to a registered masking method."""
    if __name__ == "__main__":
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        sys.modules.setdefault("src.preedit", sys.modules[__name__])

    import src.preprocess  # noqa: F401

    try:
        factory = MASKER_FACTORIES[method_name]
    except KeyError as exc:
        raise ValueError(f"No masker registered for method: {method_name!r}") from exc

    output_dirs = setup_directories(method_name, output_root)
    masker = factory()
    masker.run(data_dir=data_dir, output_dirs=output_dirs)


def run_deduplication(data_dir: Path) -> None:
    """Dispatch the future deduplication implementation."""
    raise NotImplementedError(f"Deduplication is not implemented for {data_dir}")


def existing_directory(value: str) -> Path:
    """Resolve and validate a CLI input directory."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image preprocessing task entry point")
    parser.add_argument(
        "--task",
        required=True,
        help="Task name: mask_<method_name> (for example mask_cascade) or dedup",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=existing_directory,
        help="Source training or test image directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Optional isolated output root; defaults to preData/<method_name>",
    )
    args = parser.parse_args(argv)

    if args.task != "dedup" and not args.task.startswith(MASK_TASK_PREFIX):
        parser.error("--task must be dedup or start with mask_")
    if args.task == MASK_TASK_PREFIX:
        parser.error("--task must include a masking method name after mask_")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.task == "dedup":
        run_deduplication(args.data_dir)
        return

    method_name = args.task.removeprefix(MASK_TASK_PREFIX)
    output_root = args.output_dir.expanduser().resolve() if args.output_dir else None
    run_masking(method_name, args.data_dir, output_root)


if __name__ == "__main__":
    main()
