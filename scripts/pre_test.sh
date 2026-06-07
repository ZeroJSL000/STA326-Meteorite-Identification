#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Input and output paths.
TEST_IMAGE_DIR="${TEST_IMAGE_DIR:-data/test_images}"
SAMPLE_SUBMISSION_CSV="${SAMPLE_SUBMISSION_CSV:-data/sample_submission.csv}"
CASCADE_TEST_RESULT_DIR="${CASCADE_TEST_RESULT_DIR:-preData/cascade_test}"
CASCADE_DATASET_DIR="${CASCADE_DATASET_DIR:-preData/cascade_dataset}"

# Keep test preprocessing aligned with the training cascade.
export CASCADE_YOLO_PREDICTION_CONFIDENCE="${CASCADE_YOLO_PREDICTION_CONFIDENCE:-0.02}"
export CASCADE_YOLO_MIN_BOX_CONFIDENCE="${CASCADE_YOLO_MIN_BOX_CONFIDENCE:-0.015}"
export CASCADE_YOLO_MAX_ASPECT_RATIO="${CASCADE_YOLO_MAX_ASPECT_RATIO:-5.0}"
export CASCADE_RMBG_SECONDARY_YOLO="${CASCADE_RMBG_SECONDARY_YOLO:-1}"
export CASCADE_SKIP_EXISTING="${CASCADE_SKIP_EXISTING:-1}"
export CASCADE_EXCLUDE_FILE=""

RUN_TEST_MASKING="${RUN_TEST_MASKING:-1}"
RUN_TEST_ASSEMBLY="${RUN_TEST_ASSEMBLY:-1}"

if [[ "${RUN_TEST_MASKING}" == "1" ]]; then
  echo "[1/2] Running cascade masking for test images"
  uv run python src/preedit.py \
    --task mask_cascade \
    --data_dir "${TEST_IMAGE_DIR}" \
    --output_dir "${CASCADE_TEST_RESULT_DIR}"
fi

if [[ "${RUN_TEST_ASSEMBLY}" == "1" ]]; then
  echo "[2/2] Assembling complete masked test dataset"
  uv run python src/prepare_masked_test.py \
    --sample_submission_csv "${SAMPLE_SUBMISSION_CSV}" \
    --source_dir "${TEST_IMAGE_DIR}" \
    --result_dir "${CASCADE_TEST_RESULT_DIR}" \
    --output_dir "${CASCADE_DATASET_DIR}/test_images"
fi

echo "Visual comparisons: ${CASCADE_TEST_RESULT_DIR}/visualizations"
