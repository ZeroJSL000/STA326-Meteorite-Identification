# 图像二分类 ConvNeXtV2 Baseline

当前管线严格复现历史最佳方向：`convnextv2_base.fcmae_ft_in22k_in1k`、
`384 x 384`、默认 GAP 池化、单 logit 分类头、动态加权
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
压缩，以及由 YAML 控制的 `CoarseDropout`。

## 执行实验

```bash
chmod +x scripts/run_experiment.sh
./scripts/run_experiment.sh
```

当前实验升级为 YAML 配置驱动，默认读取 `configs/convnextv2_base_384_ema_smooth.yaml`：

```text
pos_weight = disabled (plain BCE)
label_smoothing = 0.05
EMA = enabled(decay=0.999)
pooling = GAP (timm default)
head_lr = 5e-4
backbone_lr = 5e-5
CoarseDropout = tuned(max_holes=4, max_size=48)
OOF threshold search range = [0.10, 0.90], step=0.01
```

推荐通过复制并修改 `configs/*.yaml` 管理新实验，例如：
`bash scripts/run_experiment.sh configs/convnextv2_base_384_ema_smooth.yaml`。
默认输出位于 `outputs/convnextv2_base_384_ema_smooth/`，模型
checkpoint 位于 `weights/checkpoints/convnextv2_base_384_ema_smooth/`。
每次完整训练结束后，
`src/train.py` 会自动向 `docs/experiment_log.md` 追加配置、各折 F1 和
OOF 阈值；提交线上分数后，在对应实验条目补充 LB Score 与结论。


## SwinV2 minimal + 0.81818 强基线融合

新增配置：`configs/swinv2_base_384_minimal_external_081818.yaml`。它使用本地
timm 1.0.27 可用的较新 SwinV2：
`swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`。训练阶段设置
`augmentation.mode: minimal`，只保留模型训练必须的 resize/pad、Normalize 和
ToTensor，不再做颜色、几何、压缩、CLAHE 或 CoarseDropout 增强。

首次使用 SwinV2 前需要准备离线权重：

```bash
uv run python - <<'PY'
from pathlib import Path

import timm
import torch

model_name = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
output_path = Path("weights") / f"{model_name}.pth"
model = timm.create_model(model_name, pretrained=True)
torch.save(model.state_dict(), output_path)
print(f"saved pretrained weights to: {output_path}")
PY
```

运行：

```bash
bash scripts/run_experiment.sh configs/swinv2_base_384_minimal_external_081818.yaml
```

推理会读取项目根目录的 `0.81818.csv`。默认策略为 `guarded`：以该 CSV
作为 0.81818 F1 的强基线，只在 SwinV2 与它不一致且 SwinV2 概率距离 OOF
阈值至少 `external_override_margin` 时替换标签。输出包括：

```text
submission_model_only.csv          # SwinV2 单模型结果
submission_external_baseline.csv   # 对齐后的 0.81818 强基线
submission_external_guarded.csv    # guarded 融合候选
external_disagreements.csv         # 两者不一致样本，按模型置信度排序
submission.csv                     # 当前配置选定的最终提交
```
