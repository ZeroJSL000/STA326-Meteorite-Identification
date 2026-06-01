#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <path_to_yaml_config>" >&2
  echo "Example: bash $0 configs/convnextv2_masked_baseline.yaml" >&2
  exit 1
fi

CONFIG_PATH="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[1/3] Preparing cascade training dataset"
bash scripts/pre.sh

echo "[2/3] Preparing cascade test dataset"
bash scripts/pre_test.sh

echo "[3/3] Training and generating submission"
bash scripts/run_experiment.sh "${CONFIG_PATH}"
