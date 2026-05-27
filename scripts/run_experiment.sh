#!/usr/bin/env bash
set -euo pipefail

# 可通过环境变量覆盖训练批量大小或 epochs 进行调试。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-convnextv2_base_384_gem_rollback}"
mkdir -p weights outputs docs

uv run python src/train.py
uv run python src/predict.py --experiment-name "${EXPERIMENT_NAME}"
