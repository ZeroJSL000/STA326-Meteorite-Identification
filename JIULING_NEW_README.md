# Jiuling_new Experiment README

本分支记录本轮为冲击 `0.81818.csv` 强基线而做的全部改动与实验输出。核心目标是：不再做额外图片预处理，采用当前环境可用的较新 SwinV2 版本，并把 `0.81818.csv` 从单纯的后处理参考推进到训练机制中。

## 关键结论

- `0.81818.csv` 已经引入训练：在 `swinv2_base_384_minimal_pseudo0818` 中，它作为 test set 伪标签加入每个 fold 的训练集，伪标签样本权重为 `0.40`。
- 验证集仍只使用真实训练 fold 的 holdout，不把伪标签放入 validation，因此 OOF 不会直接被 `0.81818.csv` 污染。
- 已加入测试集类别数量先验：最终提交强制校准为 `86` 个正类、`108` 个负类。
- 最终推荐提交文件是 `outputs/swinv2_base_384_minimal_pseudo0818/submission.csv`。
- OOF 不是 test set 分数。本轮最终是否超过 `0.81818` 需要线上或外部 test set 评估确认。

## 代码机制改动

### Minimal Augmentation

`augmentation.mode: minimal` 会让训练图像只经过以下必要处理：

```text
LongestMaxSize(image_size)
PadIfNeeded(image_size, image_size)
Normalize()
ToTensorV2()
```

也就是说训练阶段不再使用颜色扰动、几何扰动、压缩退化、CLAHE、模糊、噪声或 CoarseDropout。

相关文件：

- `src/utils/config.py`
- `src/utils/dataset.py`
- `src/train.py`

### 外部强基线融合

推理阶段新增外部 CSV 接入机制，支持：

- `external_strategy: none`
- `external_strategy: blend`
- `external_strategy: guarded`

本轮使用 `guarded`：以 `0.81818.csv` 为强基线，只在 SwinV2 与其不一致且模型概率距离 OOF 阈值足够远时替换标签。

相关输出：

- `submission_model_only.csv`
- `submission_external_baseline.csv`
- `submission_external_guarded.csv`
- `external_disagreements.csv`
- `submission.csv`

相关文件：

- `src/predict.py`
- `src/utils/config.py`

### 伪标签训练

新增 `pseudo_label` 配置块：

```yaml
pseudo_label:
  enabled: true
  csv: "0.81818.csv"
  weight: 0.40
```

训练时会把 `0.81818.csv` 中的 test image 标签作为低权重伪标签样本加入训练集。真实训练样本权重为 `1.0`，伪标签样本权重为 `0.40`。

相关文件：

- `src/train.py`
- `src/utils/config.py`
- `src/utils/dataset.py`
- `src/utils/engine.py`

### 正负类数量校准

推理阶段新增 `target_positive_count`：

```yaml
inference:
  target_positive_count: 86
```

最终会按照模型概率排序，取概率最高的 `86` 个样本为正类，其余 `108` 个为负类。该机制会生成 `submission_count_calibrated.csv`，并覆盖最终 `submission.csv`。

相关文件：

- `src/predict.py`
- `src/utils/config.py`

## 实验 1: SwinV2 Minimal + 0.81818 Guarded Fusion

配置文件：

```text
configs/swinv2_base_384_minimal_external_081818.yaml
```

模型：

```text
swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
```

训练设置：

- image size: `384`
- augmentation: `minimal`
- loss: `BCEWithLogitsLoss`
- label smoothing: `0.05`
- EMA: `enabled, decay=0.999`
- batch size: `4`
- accumulation steps: `8`
- epochs: `15`
- `0.81818.csv`: 只在推理阶段用于 guarded fusion

结果：

```text
OOF F1: 0.954545
OOF threshold: 0.46
fold 0 best F1: 0.961194
fold 1 best F1: 0.951289
fold 2 best F1: 0.955224
fold 3 best F1: 0.964706
fold 4 best F1: 0.960000
```

预测分布：

```text
model-only: 131 positives, 63 negatives
0.81818.csv: 90 positives, 104 negatives
guarded final: 110 positives, 84 negatives
guarded final vs 0.81818.csv differences: 20
model-only vs 0.81818.csv differences: 75
```

输出目录：

```text
outputs/swinv2_base_384_minimal_external_081818/
```

## 实验 2: SwinV2 Minimal + 0.81818 Pseudo Label + 86/108 Count Prior

配置文件：

```text
configs/swinv2_base_384_minimal_pseudo0818.yaml
```

模型：

```text
swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
```

训练设置：

- image size: `384`
- augmentation: `minimal`
- loss: `BCEWithLogitsLoss`
- label smoothing: `0.05`
- EMA: `enabled, decay=0.999`
- batch size: `4`
- accumulation steps: `8`
- epochs: `8`
- pseudo label CSV: `0.81818.csv`
- pseudo label weight: `0.40`
- final target positive count: `86`
- final target negative count: `108`

结果：

```text
OOF F1: 0.938084
OOF threshold: 0.44
fold 0 best F1: 0.949853
fold 1 best F1: 0.944928
fold 2 best F1: 0.941860
fold 3 best F1: 0.952941
fold 4 best F1: 0.925816
```

预测分布：

```text
final submission: 86 positives, 108 negatives
count calibrated: 86 positives, 108 negatives
model-only: 106 positives, 88 negatives
0.81818.csv: 90 positives, 104 negatives
guarded fusion before count calibration: 91 positives, 103 negatives
```

差异统计：

```text
final vs 0.81818.csv differences: 26
final vs guarded differences: 25
final vs model-only differences: 20
model-only vs 0.81818.csv differences: 34
guarded vs 0.81818.csv differences: 1
```

最终推荐提交文件：

```text
outputs/swinv2_base_384_minimal_pseudo0818/submission.csv
```

输出目录：

```text
outputs/swinv2_base_384_minimal_pseudo0818/
```

## 复现命令

准备 SwinV2 离线权重：

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

运行 guarded fusion 实验：

```bash
bash scripts/run_experiment.sh configs/swinv2_base_384_minimal_external_081818.yaml
```

运行伪标签 + 数量先验实验：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/run_experiment.sh configs/swinv2_base_384_minimal_pseudo0818.yaml
```

## 后续建议

当前伪标签实验只跑了 `8` epoch，OOF 低于 15 epoch 的非伪标签 SwinV2 minimal 实验。若线上反馈接近或优于 `0.81818`，建议再跑一个 15 epoch 的同配置伪标签版本，并对 `pseudo_label.weight` 做 `0.20 / 0.40 / 0.60` 小网格搜索。
