"""Command-line interface for final inference."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import ProjectPaths, load_config, project_path
from .inference import run_final_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final SwinV2 Top-86 inference without guarded fusion."
    )
    parser.add_argument("--config", default="configs/final.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="weights/checkpoints/final")
    parser.add_argument("--output-dir", default="results/final")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        key: value
        for key, value in {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }.items()
        if value is not None
    }
    if overrides:
        config = replace(config, **overrides)
        config.validate()
    paths = ProjectPaths(
        data_dir=project_path(args.data_dir),
        checkpoint_dir=project_path(args.checkpoint_dir),
        output_dir=project_path(args.output_dir),
        config_file=project_path(args.config),
    )
    run_final_inference(config, paths)


if __name__ == "__main__":
    main()
