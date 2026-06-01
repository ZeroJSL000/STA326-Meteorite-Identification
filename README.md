# Cascade Masked Image Classification Pipeline

## 项目概述

本项目用于陨石 vs 非陨石二分类。完整流程包含级联掩码预处理、单模型训练、异构集成、co-teaching 伪标签迭代，以及 Temperature Scaling 与 Bayesian Label Shift Correction 校准。

所有预训练权重均由用户手动放入 `weights/`。源码不会自动下载权重。

## 目录结构

```text
configs/                  # YAML 实验配置
scripts/                  # 预处理、训练和完整流水线脚本
src/                      # 训练、推理、集成、伪标签和校准源码
data/                     # 原始竞赛数据，本地保留，不上传 Git
preData/                  # 级联掩码后的数据集，本地生成，不上传 Git
weights/                  # 手动放置的预训练权重与训练 checkpoint，不上传 Git
outputs/                  # 本地实验输出，不上传 Git
docs/                     # 实验记录
```

核心数据结构：

```text
data/
├── train_images/
├── train_labels.csv
├── test_images/
└── sample_submission.csv

preData/cascade_dataset/
├── train_images/
├── train_labels.csv
├── test_images/
└── pseudo_labels.csv       # Phase 3 生成后存在
```

## 环境准备

项目使用 Python `3.14+` 与 `uv`：

```bash
uv sync
```

将需要的离线权重手动放入 `weights/`。不要在源码中加入自动下载逻辑。

## 快速开始

### 1. 数据预处理

```bash
bash scripts/pre.sh        # 训练集
bash scripts/pre_test.sh   # 测试集
```

预处理结果写入 `preData/cascade_dataset/`。也可以从预处理到基线提交一次执行：

```bash
bash scripts/run_pipeline.sh configs/convnextv2_masked_baseline.yaml
```

### 2. 单模型训练（Phase 1）

ConvNeXtV2 基线：

```bash
bash scripts/run_experiment.sh configs/convnextv2_masked_baseline.yaml
```

CSWin 基线：

```bash
bash scripts/run_experiment.sh configs/cswin_masked_baseline.yaml
```

`run_experiment.sh` 会依次调用 `src/train.py` 和 `src/predict.py`。Focal Loss、Soft-Masking、RandAugment、MixUp、CutMix、多尺度训练、hard sample mining 和 TTA 均由 YAML 开关控制；关闭开关时保留原始 BCE、单尺度、无 TTA 流程。

### 3. 异构模型集成（Phase 2）

`src/ensemble.py` 会输出 weighted blend、Logistic Regression stacking 和 rank averaging 三种结果：

```bash
uv run python src/ensemble.py \
  --dirs outputs/convnextv2_masked_baseline outputs/cswin_masked_baseline \
  --output-dir outputs/ensemble
```

如已训练更多模型，可继续在 `--dirs` 后追加实验目录。每个输入目录必须包含：

```text
oof_predictions.csv
submission_probabilities.csv
```

### 4. 伪标签迭代（Phase 3）

根据两个强模型的高置信共识生成伪标签：

```bash
uv run python src/pseudo_label.py \
  --config configs/convnextv2_phase3_pseudo.yaml
```

重新训练带伪标签的模型：

```bash
bash scripts/run_experiment.sh configs/convnextv2_phase3_pseudo.yaml
```

重新集成：

```bash
uv run python src/ensemble.py \
  --dirs outputs/convnextv2_phase3_pseudo outputs/cswin_masked_baseline \
  --output-dir outputs/ensemble_phase3
```

### 5. 概率校准与提交（Phase 4）

Phase 4 直接复用已有 OOF 和测试概率，无需重新训练：

```bash
uv run python src/calibrate.py \
  --config configs/ensemble_phase4_calibrated.yaml
```

当前配置读取 `outputs/ensemble_phase3/stacking_oof_predictions.csv` 和 `outputs/ensemble_phase3/stacking.csv`，输出到 `outputs/ensemble_phase4_calibrated/`：

- `submission.csv`：当前 `auto` 策略选择的 OOF F1-optimal 阈值版。
- `prior_aligned_submission.csv`：使用合法单一全局阈值对齐测试集先验的对照版，推荐提交。

## 配置说明（YAML Schema）

关键 YAML 字段：

| 字段 | 说明 |
| --- | --- |
| `experiment_name` | 实验名称，同时决定 `outputs/<exp>/` 与 checkpoint 子目录 |
| `paths.dataset_dir` | 级联掩码数据集根目录 |
| `model.backbone` | timm 模型名称或仓库内注册模型名称 |
| `model.pretrained_file` | `weights/` 下的离线预训练权重文件 |
| `train.loss_type` | 损失函数：`bce` 或 `focal` |
| `train.focal_gamma`, `train.focal_alpha` | Focal Loss 参数 |
| `train.hard_mining_enabled` | 是否按历史 OOF loss 启用 hard sample mining |
| `augmentation.randaugment.enabled` | 是否启用 RandAugment |
| `augmentation.mixup_enabled`, `augmentation.cutmix_enabled` | 是否启用互斥的 MixUp / CutMix |
| `augmentation.multiscale_train_sizes` | 每个 epoch 可随机采样的训练分辨率 |
| `preprocessing.soft_masking.enabled` | 是否融合原图与掩码图 |
| `test.tta_enabled`, `test.tta_scales` | 是否启用 TTA 及其尺度 |
| `pseudo_labeling.enabled` | 是否合并高置信伪标签进行重训练 |
| `pseudo_labeling.co_teaching_enabled` | 是否要求双模型共识筛选伪标签 |
| `calibration.enabled` | 是否启用 Phase 4 校准；关闭时跳过校准逻辑 |
| `calibration.prior_pos_rate` | 测试集先验正类率，当前为 `0.438` |
| `calibration.use_bayesian_correction` | 是否启用 Bayesian Label Shift Correction |
| `calibration.threshold_strategy` | `auto`、`f1_optimal` 或 `prior_aligned` |

## 模型权重清单

以下文件由用户手动准备。代码不会联网下载权重：

| 模型 | 本地文件或配置方式 | 用途与来源说明 |
| --- | --- | --- |
| ConvNeXtV2 | `weights/convnextv2_base.fcmae_ft_in22k_in1k.pth` | 当前主力分类器；使用 timm 兼容的官方预训练权重 |
| CSWin | `weights/cswin_base_384.pth` | 当前异构分类器；使用上游 CSWin 预训练权重 |
| BEiT | `weights/beit_base_patch16_384.pth` | 可选 timm 分类器；使用 BEiT 预训练权重 |
| EVA-02 | 当前未放置固定文件 | 可选扩展；手动准备 timm 兼容权重并在 YAML 中配置 |
| MaxViT | 当前未放置固定文件 | 可选扩展；手动准备 timm 兼容权重并在 YAML 中配置 |
| YOLO-World | `weights/yolov8l-worldv2.pt` | 级联预处理主体检测 |
| U2-Net | `weights/u2net.onnx` | 级联预处理背景移除 |

训练生成的折模型保存在 `weights/checkpoints/<experiment_name>/`。`weights/` 当前体积较大，仅在本地保留，不上传 Git。

## 输出文件说明

单模型输出位于 `outputs/<experiment_name>/`：

| 文件 | 说明 |
| --- | --- |
| `oof_predictions.csv` | OOF 标签与概率，用于阈值搜索和集成 |
| `submission_probabilities.csv` | 测试集概率 |
| `submission.csv` | 单模型或校准后的 0/1 提交 |
| `temperature.json` | Phase 4 的温度参数与校准前后 NLL |
| `optimal_threshold.json` | F1 阈值、先验阈值、最终策略和正类率记录 |

集成阶段还会生成 `weighted_blend.csv`、`stacking.csv`、`rank_avg.csv` 与对应 OOF 文件。

## 提交策略建议

**推荐提交文件**：`outputs/<exp>/prior_aligned_submission.csv`，其正类数量已根据先验知识（80~90/194）通过合法的全局阈值校准对齐。

当前 Phase 4 实验的推荐文件为：

```text
outputs/ensemble_phase4_calibrated/prior_aligned_submission.csv
```

其次可提交 `outputs/ensemble_phase4_calibrated/submission.csv` 作为 OOF F1-optimal 对照。

## 注意事项

- 严禁使用 Top-K 截断、`argsort()[-K:]` 或类似提交后处理。
- 测试集先验正类率配置为 `0.438`，即 `85 / 194` 附近，不是百分数。
- `data/`、`preData/`、`weights/` 和 `outputs/` 均为本地数据或流程生成物，不上传 Git。
- 使用新模型前，先手动放置离线权重，再在 YAML 中填写 `model.pretrained_file`。
