# 图像二分类 ConvNeXtV2 Baseline

当前管线严格复现历史最佳方向：`convnextv2_base.fcmae_ft_in22k_in1k`、
`384 x 384`、`GeM(p=3.0)` 池化、单 logit 分类头、动态加权
`BCEWithLogitsLoss`、`AdamW` 与 `CosineAnnealingLR`。验证和测试图像
只执行保持比例的缩放及零填充，不会将石头主体强行拉伸。

## 环境安装

项目使用 Python `3.14+` 与 `uv`：

```bash
uv add torch torchvision timm pandas numpy scikit-learn albumentations tqdm pillow
```

## 下载离线预训练权重

训练代码仅从 `weights/` 读取初始化权重。首次在联网环境执行：

```bash
mkdir -p weights
uv run python - <<'PY'
from pathlib import Path

import timm
import torch

model_name = "convnextv2_base.fcmae_ft_in22k_in1k"
output_path = Path("weights") / f"{model_name}.pth"
model = timm.create_model(model_name, pretrained=True)
torch.save(model.state_dict(), output_path)
print(f"saved pretrained weights to: {output_path}")
PY
```

## 预处理与增强

验证和推理使用固定流程：

```text
LongestMaxSize(max_size=384)
PadIfNeeded(min_height=384, min_width=384, border_mode=BORDER_CONSTANT, fill=0)
Normalize()
```

训练在该 aspect-safe 基础上加入 `ColorJitter`、`RandomGamma`、
`HueSaturationValue`、`CLAHE`、`ShiftScaleRotate`、轻度模糊/噪声/JPEG
压缩与强化 `CoarseDropout(max_holes=8, max_size=64)`。

## 执行实验

```bash
chmod +x scripts/run_experiment.sh
./scripts/run_experiment.sh
```

当前实验从 LB `0.73631` 的高分基线回退并叠加 GeM/遮挡优化，默认采用：

```text
pos_weight = negative_count / positive_count
pooling = GeM(p_init=3.0)
OOF threshold search range = [0.10, 0.90], step=0.01
```

可通过环境变量调整 GeM 初值，例如
`GEM_P=3.5 ./scripts/run_experiment.sh`。
默认输出位于 `outputs/convnextv2_base_384_gem_rollback/`，模型
checkpoint 位于 `weights/checkpoints/convnextv2_base_384_gem_rollback/`。
每次完整训练结束后，
`src/train.py` 会自动向 `docs/experiment_log.md` 追加配置、各折 F1 和
OOF 阈值；提交线上分数后，在对应实验条目补充 LB Score 与结论。
