# Experiment Log

## HIST-001-swin_base_384_ema

- 时间: 已完成实验，线上反馈记录于 2026-05-26
- 实验名: `swin_base_384_ema`
- Backbone: `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
- 图像尺寸: `384 x 384`
- Loss: `Focal Loss + label smoothing`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingWarmRestarts`
- 数据增强: RandomResizedCrop + flips/rotate + color/noise/dropout + Mixup/CutMix；推理包含翻转 TTA

| Fold | Best Valid F1 | Best Threshold |
| ---: | ---: | ---: |
| 0 | 0.996289 | - |
| 1 | 0.991659 | - |
| 2 | 0.994413 | - |
| 3 | 0.996283 | - |
| 4 | 0.988889 | - |

- OOF F1: `0.992386`
- OOF 阈值: `0.520100`
- Kaggle LB Score: `0.56375`
- 后续推断改进方向: OOF 与 LB 严重背离，停止会改变主体形态的普通 Resize/RandomResizedCrop 路径；回到 `convnextv2_base.fcmae_ft_in22k_in1k` 的 aspect-safe Baseline 严格复现。

## HIST-REF-002-best-baseline

- 时间: 历史参考，线上反馈记录于 2026-05-26
- 实验名: `historical_best_baseline`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: Aspect-safe resize/pad 基础上的颜色、局部亮度、几何、轻度退化与遮挡增强
- 各折验证 F1: `历史记录未提供`
- OOF 阈值: `历史记录未提供`
- Kaggle LB Score: `0.70748`
- 后续推断改进方向: 首先严格复现该配置，并用自动实验日志记录新 OOF/LB 对齐情况。

## EXP-20260527-023149-convnextv2_base_384_baseline

- 时间: 2026-05-27T02:31:49+08:00
- 实验名: `convnextv2_base_384_baseline`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout; valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.994434 |
| 1 | 2148 | 1930 | 0.898510 | 0.991659 |
| 2 | 2148 | 1930 | 0.898510 | 0.993513 |
| 3 | 2148 | 1931 | 0.898976 | 0.995349 |
| 4 | 2148 | 1931 | 0.898976 | 0.989805 |

- OOF F1: `0.991301`
- OOF 最优阈值: `0.120000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_baseline/optimal_threshold.json`
- Kaggle LB Score: `0.73631`
- 线上预测分布: `115` 个正例、`79` 个负例（从 `submission.csv` 核验，共 `194` 个样本）
- 分析: 线上输出明显偏向正类，存在矫枉过正（Over-correction）。训练折原始 `pos_weight=negative/positive` 约为 `0.899`，实际小于 `1`；此外，全局 OOF 阈值低至 `0.12`，也可能放大了测试集正例输出。下一轮以可记录的阻尼系数进一步降低正类损失权重，观察精确率与 LB 变化。
- 推断改进方向: 引入 `pos_weight` 阻尼（Dampening），使用 `effective_pos_weight = raw_pos_weight * 0.5` 削弱当前正类输出倾向；继续只基于 OOF 搜索阈值，不硬编码提交集正类数量。

## EXP-20260527-133631-convnextv2_base_384_posweight_damped

- 时间: 2026-05-27T13:36:31+08:00
- 实验名: `convnextv2_base_384_posweight_damped`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=raw_pos_weight*dampening)`
- pos_weight 阻尼系数: `0.500000`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout; valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Raw Weight | Effective Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.449255 | 0.996283 |
| 1 | 2148 | 1930 | 0.898510 | 0.449255 | 0.992565 |
| 2 | 2148 | 1930 | 0.898510 | 0.449255 | 0.993513 |
| 3 | 2148 | 1931 | 0.898976 | 0.449488 | 0.996283 |
| 4 | 2148 | 1931 | 0.898976 | 0.449488 | 0.987996 |

- OOF F1: `0.992389`
- OOF 最优阈值: `0.510000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_posweight_damped/optimal_threshold.json`
- Kaggle LB Score: `0.71895`
- 线上预测分布: `67` 个正例、`127` 个负例（从 `submission.csv` 核验，共 `194` 个样本）
- 分析: 训练集实际接近平衡且正类略多，静态 `pos_weight` 再叠加固定 `0.5` 阻尼将有效正类权重压至约 `0.449`，导致明显少报正类和线上性能回落。需要校正的是，本轮 OOF 阈值为 `0.51`；低至 `0.12` 的脆弱阈值发生在前一轮 baseline。两轮合并说明，通过试错操控静态类别权重或依赖极端阈值都缺乏稳健泛化依据。
- 推断改进方向: 彻底移除静态 `pos_weight` 计算，引入带 `0.05` Label Smoothing 的 `FocalLoss(gamma=2.0)`，动态降低易分样本贡献；同时将 OOF F1 阈值搜索限制在 `[0.25, 0.75]`，继续避免硬编码测试集正类数量。
## EXP-20260527-163053-convnextv2_base_384_focalloss_smoothed

- 时间: 2026-05-27T16:30:53+08:00
- 实验名: `convnextv2_base_384_focalloss_smoothed`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- 图像尺寸: `384 x 384`
- Loss: `FocalLoss(gamma=2.00, label_smoothing=0.05)`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + Affine + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout; valid/test: aspect-safe resize-pad + Normalize only

| Fold | Best Valid F1 | Best Threshold |
| ---: | ---: | ---: |
| 0 | 0.996283 | 0.740000 |
| 1 | 0.994403 | 0.500000 |
| 2 | 0.995366 | 0.710000 |
| 3 | 0.993464 | 0.690000 |
| 4 | 0.988889 | 0.500000 |

- OOF F1: `0.992929`
- OOF 最优阈值: `0.710000`
- 阈值搜索: `0.25..0.75, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_focalloss_smoothed/optimal_threshold.json`
- Kaggle LB Score: `~0.6`（用户反馈的近似值）
- 线上预测分布: `50` 个正例、`144` 个负例（从 `submission.csv` 核验，共 `194` 个样本）
- 分析: Focal Loss 实验严重失败；在当前极度嘈杂的数据集上，动态强调难分样本可能过度关注离群点和背景噪声，导致概率分布崩溃并大幅少报正类。
- 推断改进方向: 放弃 Focal Loss 与 Label Smoothing，确立 `EXP-20260527-023149-convnextv2_base_384_baseline`（LB `0.73631`）为回退点，恢复原始动态 `pos_weight=negative_count/positive_count` 与宽松 OOF 阈值搜索；仅叠加 GeM 池化和强遮挡增强以抵抗标尺、手指等局部干扰。
## EXP-20260527-200410-convnextv2_base_384_gem_rollback

- 时间: 2026-05-27T20:04:10+08:00
- 实验名: `convnextv2_base_384_gem_rollback`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GeM(p_init=3.00)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + Strong CoarseDropout(max_holes=8, max_size=64); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.996283 |
| 1 | 2148 | 1930 | 0.898510 | 0.991612 |
| 2 | 2148 | 1930 | 0.898510 | 0.994424 |
| 3 | 2148 | 1931 | 0.898976 | 0.991674 |
| 4 | 2148 | 1931 | 0.898976 | 0.990775 |

- OOF F1: `0.991453`
- OOF 最优阈值: `0.820000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_gem_rollback/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
