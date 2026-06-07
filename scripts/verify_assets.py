#!/usr/bin/env python3
"""Validate final dataset and checkpoint layout before inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "weights" / "checkpoints" / "final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train = pd.read_csv(DATA / "train_labels.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    train_images = list((DATA / "train_images").iterdir())
    test_images = list((DATA / "test_images").iterdir())
    assert len(train_images) == len(train), (len(train_images), len(train))
    assert len(test_images) == len(sample), (len(test_images), len(sample))
    assert {str(value) for value in train["id"]} == {path.name for path in train_images}
    assert {str(value) for value in sample["id"]} == {path.name for path in test_images}

    metadata_file = CHECKPOINTS / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    missing = [
        name
        for name in metadata["checkpoint_files"]
        if not (CHECKPOINTS / name).is_file()
    ]
    print(f"train_rows={len(train)}")
    print(f"test_rows={len(sample)}")
    print(f"checkpoint_files={len(metadata['checkpoint_files'])}")
    print(f"missing_checkpoints={len(missing)}")
    if missing:
        print("Missing checkpoints:")
        for name in missing:
            print(f"- {name}")
    else:
        for name in metadata["checkpoint_files"]:
            print(f"{sha256(CHECKPOINTS / name)}  {name}")


if __name__ == "__main__":
    main()
