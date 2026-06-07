#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Input and output paths.
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-data/train_images}"
TRAIN_CSV="${TRAIN_CSV:-data/train_labels.csv}"
CASCADE_RESULT_DIR="${CASCADE_RESULT_DIR:-preData/cascade}"
CASCADE_DATASET_DIR="${CASCADE_DATASET_DIR:-preData/cascade_dataset}"
CASCADE_EXCLUDE_FILE="${CASCADE_EXCLUDE_FILE:-scripts/pre_exclude.txt}"

# YOLO-World cascade tuning parameters.
export CASCADE_YOLO_PREDICTION_CONFIDENCE="${CASCADE_YOLO_PREDICTION_CONFIDENCE:-0.02}"
export CASCADE_YOLO_MIN_BOX_CONFIDENCE="${CASCADE_YOLO_MIN_BOX_CONFIDENCE:-0.015}"
export CASCADE_YOLO_MAX_ASPECT_RATIO="${CASCADE_YOLO_MAX_ASPECT_RATIO:-5.0}"
export CASCADE_RMBG_SECONDARY_YOLO="${CASCADE_RMBG_SECONDARY_YOLO:-1}"
export CASCADE_EXCLUDE_FILE
export CASCADE_SKIP_EXISTING="${CASCADE_SKIP_EXISTING:-1}"

# Set any step to 0 when reusing existing preprocessing outputs.
RUN_MASKING="${RUN_MASKING:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
RUN_BASELINE_ASSEMBLY="${RUN_BASELINE_ASSEMBLY:-1}"

ABS_TRAIN_CSV="$(realpath "${TRAIN_CSV}")"

if [[ "${RUN_MASKING}" == "1" ]]; then
  echo "[1/3] Running cascade masking"
  uv run python src/preedit.py \
    --task mask_cascade \
    --data_dir "${TRAIN_IMAGE_DIR}"
fi

if [[ "${RUN_EVALUATION}" == "1" ]]; then
  echo "[2/3] Evaluating cascade masking"
  uv run python src/evaluate.py \
    --csv_path "${ABS_TRAIN_CSV}" \
    --result_dir "${CASCADE_RESULT_DIR}"
fi

if [[ "${RUN_BASELINE_ASSEMBLY}" == "1" ]]; then
  echo "[3/3] Assembling baseline dataset"
  uv run python src/prepare_baseline.py \
    --orig_csv "${TRAIN_CSV}" \
    --masked_dir "${CASCADE_RESULT_DIR}/images" \
    --out_root "${CASCADE_DATASET_DIR}" \
    --exclude_file "${CASCADE_EXCLUDE_FILE}"
fi
