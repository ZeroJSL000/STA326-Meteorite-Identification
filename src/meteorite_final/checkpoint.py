"""Checkpoint loading for the final SwinV2 ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def load_checkpoint_files(metadata_file: Path) -> tuple[dict[str, Any], list[Path]]:
    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    checkpoint_files = [
        metadata_file.parent / name for name in metadata["checkpoint_files"]
    ]
    missing = [str(path) for path in checkpoint_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing final fold checkpoints:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    return metadata, checkpoint_files


def _state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in (
        "state_dict_ema",
        "state_dict",
        "model",
        "model_state_dict",
        "ema_state_dict",
    ):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def load_weights(model: nn.Module, path: Path) -> int:
    source = _state_dict(torch.load(path, map_location="cpu", weights_only=False))
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and value.shape == target[key].shape
    }
    if len(compatible) != len(target):
        missing = sorted(set(target) - set(compatible))
        raise RuntimeError(f"{path.name} is incompatible; missing keys: {missing[:10]}")
    model.load_state_dict(compatible, strict=True)
    return len(compatible)
