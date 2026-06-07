"""Pure-model five-fold inference without guarded fusion."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint_files, load_weights
from .config import InferenceConfig, ProjectPaths
from .data import TestImageDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _autocast(device: torch.device, config: InferenceConfig):
    if not config.amp or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: InferenceConfig,
    description: str,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with _autocast(device, config):
            logits = model(images).flatten()
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(predictions)


def run_final_inference(
    config: InferenceConfig, paths: ProjectPaths
) -> dict[str, object]:
    seed_everything(config.seed)
    metadata, checkpoint_files = load_checkpoint_files(paths.metadata_file)
    model_name = str(metadata.get("model_name", config.model_name))
    image_size = int(metadata.get("image_size", config.image_size))

    sample = pd.read_csv(paths.sample_submission_file)
    if config.top_k > len(sample):
        raise ValueError(f"top_k={config.top_k} exceeds test rows={len(sample)}")
    dataset = TestImageDataset(
        sample, paths.test_image_dir, image_size, config.id_column
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )

    probability_sum = np.zeros(len(sample), dtype=np.float64)
    print(
        f"device={device}; folds={len(checkpoint_files)}; "
        f"batch_size={config.batch_size}; top_k={config.top_k}"
    )
    for checkpoint_file in checkpoint_files:
        model = timm.create_model(model_name, pretrained=False, num_classes=1).to(
            device
        )
        loaded = load_weights(model, checkpoint_file)
        print(f"loaded={checkpoint_file.name}; parameters={loaded}")
        probability_sum += _predict(model, loader, device, config, checkpoint_file.name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = probability_sum / len(checkpoint_files)
    probability_frame = pd.DataFrame(
        {config.id_column: sample[config.id_column], "probability": probabilities}
    )
    ranked = probability_frame.sort_values(
        ["probability", config.id_column],
        ascending=[False, True],
        kind="mergesort",
    )
    positive_ids = set(ranked.head(config.top_k)[config.id_column])
    submission = sample[[config.id_column]].copy()
    submission[config.label_column] = (
        submission[config.id_column].isin(positive_ids).astype(int)
    )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    probability_file = paths.output_dir / "submission_probabilities.csv"
    submission_file = paths.output_dir / "submission_top86_no_guarded.csv"
    manifest_file = paths.output_dir / "manifest.json"
    probability_frame.to_csv(probability_file, index=False)
    submission.to_csv(submission_file, index=False)
    threshold = (
        float(ranked.iloc[config.top_k - 1]["probability"]) if config.top_k else None
    )
    manifest: dict[str, object] = {
        "strategy": "pure_model_fold_average_then_top_k",
        "guarded_fusion": False,
        "external_labels_used_at_inference": False,
        "checkpoints_trained_with_pseudo_labels": bool(
            metadata.get("pseudo_label_enabled", False)
        ),
        "checkpoint_files": [path.name for path in checkpoint_files],
        "model_name": model_name,
        "image_size": image_size,
        "batch_size": config.batch_size,
        "amp_dtype": config.amp_dtype,
        "seed": config.seed,
        "top_k": config.top_k,
        "top_k_threshold": threshold,
        "test_rows": len(sample),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"submission={submission_file}")
    print(f"positive_count={int(submission[config.label_column].sum())}")
    print(f"top_k_threshold={threshold}")
    return manifest
