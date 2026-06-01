#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "❌ 错误: 未提供配置文件。" >&2
  echo "💡 用法: bash $0 <path_to_yaml_config>" >&2
  echo "📝 示例: bash $0 configs/convnextv2_masked_baseline.yaml" >&2
  exit 1
fi

CONFIG_PATH="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "❌ 错误: 找不到配置文件 -> ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p weights outputs docs

echo "🚀 [1/2] 开始训练 (Training) | Config: ${CONFIG_PATH}"
uv run python src/train.py --config "${CONFIG_PATH}"

echo "🚀 [2/2] 开始推理 (Prediction) | Config: ${CONFIG_PATH}"
uv run python src/predict.py --config "${CONFIG_PATH}"

echo "✅ 实验运行完毕！"