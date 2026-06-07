#!/usr/bin/env python3
"""Reproduce the pure-model SwinV2 fold ensemble and Top-K submission.

The checkpoints were trained with pseudo-labelled samples. At inference time,
this script does not read external/API labels or apply guarded fusion.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.builder import build_model  # noqa: E402
from utils.checkpoint import load_weights_flexible  # noqa: E402
from utils.config import load_config, resolve_image_dir  # noqa: E402
from utils.dataset import BinaryImageDataset, build_valid_transforms  # noqa: E402
from utils.engine import predict_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure-model SwinV2 inference and create a Top-K submission."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/swinv2_base_384_minimal_pseudo0818_no_guarded.yaml",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "weights/checkpoints/swinv2_base_384_minimal_pseudo0818",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp/swinv2_base_384_minimal_pseudo0818_no_guarded",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=86)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loader(
    sample_submission: pd.DataFrame,
    image_dir: Path,
    image_size: int,
    id_col: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = BinaryImageDataset(
        frame=sample_submission,
        image_dir=image_dir,
        transform=build_valid_transforms(image_size),
        id_col=id_col,
        label_col=None,
    )
    kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = load_config(args.config)
    if args.batch_size is not None:
        config.inference.batch_size = args.batch_size
    if config.inference.tta_enabled:
        raise ValueError("This reproducibility entry point requires TTA to be disabled.")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    checkpoint_files = [checkpoint_dir / name for name in metadata["checkpoint_files"]]
    missing = [str(path) for path in checkpoint_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fold checkpoints: {missing}")

    config.model.name = str(metadata["model_name"])
    config.data.image_size = int(metadata["image_size"])
    sample_submission = pd.read_csv(config.paths.sample_submission_csv)
    if not 0 <= args.top_k <= len(sample_submission):
        raise ValueError(f"--top-k must be between 0 and {len(sample_submission)}")

    first_id = str(sample_submission.iloc[0][config.data.id_col])
    image_dir = resolve_image_dir(config.paths.test_image_dir, first_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(
        sample_submission,
        image_dir,
        config.data.image_size,
        config.data.id_col,
        config.inference.batch_size,
        config.data.num_workers,
        device,
    )

    probability_sum = np.zeros(len(sample_submission), dtype=np.float64)
    print(
        f"device={device}; folds={len(checkpoint_files)}; "
        f"batch_size={config.inference.batch_size}; top_k={args.top_k}"
    )
    for checkpoint_path in checkpoint_files:
        model = build_model(config.model).to(device)
        loaded_count, skipped = load_weights_flexible(model, checkpoint_path)
        print(
            f"loaded={checkpoint_path.name}; parameters={loaded_count}; "
            f"skipped={len(skipped)}"
        )
        probability_sum += predict_probabilities(
            model,
            loader,
            device,
            config.train.amp,
            description=checkpoint_path.name,
            inference_config=config.inference,
            amp_dtype=config.train.amp_dtype,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = probability_sum / len(checkpoint_files)
    probability_frame = pd.DataFrame(
        {
            config.data.id_col: sample_submission[config.data.id_col],
            "probability": probabilities,
        }
    )
    ranked = probability_frame.sort_values(
        ["probability", config.data.id_col],
        ascending=[False, True],
        kind="mergesort",
    )
    positive_ids = set(ranked.head(args.top_k)[config.data.id_col])
    submission = sample_submission[[config.data.id_col]].copy()
    submission[config.data.label_col] = (
        submission[config.data.id_col].isin(positive_ids).astype(int)
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = output_dir / "submission_probabilities.csv"
    submission_path = output_dir / "submission_top86_no_guarded.csv"
    manifest_path = output_dir / "reproduction_manifest.json"
    probability_frame.to_csv(probability_path, index=False)
    submission.to_csv(submission_path, index=False)

    threshold = float(ranked.iloc[args.top_k - 1]["probability"]) if args.top_k else None
    manifest = {
        "strategy": "pure_model_fold_average_then_top_k",
        "guarded_fusion": False,
        "external_labels_used_at_inference": False,
        "checkpoints_trained_with_pseudo_labels": bool(
            metadata.get("pseudo_label_enabled", False)
        ),
        "checkpoint_experiment_id": metadata.get("experiment_id"),
        "checkpoint_files": [path.name for path in checkpoint_files],
        "model_name": config.model.name,
        "image_size": config.data.image_size,
        "batch_size": config.inference.batch_size,
        "amp_dtype": config.train.amp_dtype,
        "seed": args.seed,
        "top_k": args.top_k,
        "top_k_threshold": threshold,
        "test_rows": len(submission),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=True, indent=2)
        file.write("\n")

    print(f"probabilities={probability_path}")
    print(f"submission={submission_path}")
    print(f"manifest={manifest_path}")
    print(f"positive_count={int(submission[config.data.label_col].sum())}")
    print(f"top_k_threshold={threshold}")


if __name__ == "__main__":
    main()
