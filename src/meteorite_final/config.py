"""Single-source path and inference configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_dir: Path = PROJECT_ROOT / "weights" / "checkpoints" / "final"
    output_dir: Path = PROJECT_ROOT / "results" / "final"
    config_file: Path = PROJECT_ROOT / "configs" / "final.json"

    @property
    def test_image_dir(self) -> Path:
        return self.data_dir / "test_images"

    @property
    def sample_submission_file(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    @property
    def metadata_file(self) -> Path:
        return self.checkpoint_dir / "metadata.json"


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    image_size: int = 384
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
    amp_dtype: str = "bfloat16"
    top_k: int = 86
    seed: int = 2026
    id_column: str = "id"
    label_column: str = "label"

    def validate(self) -> None:
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError("image_size and batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


def load_config(path: str | Path) -> InferenceConfig:
    """Load and validate the final JSON configuration."""

    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = json.load(file)
    allowed = {field.name for field in fields(InferenceConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    config = InferenceConfig(**payload)
    config.validate()
    return config
