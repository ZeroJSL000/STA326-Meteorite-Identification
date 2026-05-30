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
- 线上预测分布: `47` 个正例（用户反馈）
- 分析: `GeM + Strong CoarseDropout` 使正例输出显著减少；激进的大块黑色遮挡可能破坏了 GeM 聚焦局部高响应区域的前提，导致概率分布向负类塌缩。
- 后续推断改进方向: 开启消融实验 Phase 1，仅保留 `GeM(p=3.0)`，移除强遮挡以隔离评估池化层本身的贡献。

## Planned Experiment: convnextv2_base_384_gem_only

- 状态: `待训练`
- 目标: 消融实验 Phase 1，测试纯 `GeM Pooling` 的线上效果。
- 操作: 保留 `GeM(p=3.0)`、动态 `pos_weight=negative_count/positive_count` 的 `BCEWithLogitsLoss` 与宽松 OOF 阈值搜索；训练增强中移除 `CoarseDropout`，验证上一轮分布塌缩是否由强遮挡引起。
## EXP-20260528-010032-convnextv2_base_384_gem_only

- 时间: 2026-05-28T01:00:32+08:00
- 实验名: `convnextv2_base_384_gem_only`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GeM(p_init=3.00)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression; no CoarseDropout (GeM-only ablation); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.996283 |
| 1 | 2148 | 1930 | 0.898510 | 0.988889 |
| 2 | 2148 | 1930 | 0.898510 | 0.991674 |
| 3 | 2148 | 1931 | 0.898976 | 0.996276 |
| 4 | 2148 | 1931 | 0.898976 | 0.988018 |

- OOF F1: `0.990894`
- OOF 最优阈值: `0.530000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_gem_only/optimal_threshold.json`
- Kaggle LB Score: `0.67073`
- 线上预测分布: `73` 个正例（用户反馈）
- 分析: 纯 `GeM` 配置仍显著低于 LB `0.73631` 的 GAP 基线，说明 GeM 在当前脏数据集上容易聚焦标尺、高亮边缘等高对比局部噪声，放大局部伪特征并导致泛化崩溃。
- 后续推断改进方向: 彻底废弃 GeM，回退到 timm 默认 GAP 架构；恢复强遮挡抑制局部记忆，同时引入 LLRD，让分类头使用基础学习率、ConvNeXtV2 预训练骨干使用 `0.1x` 学习率以保护预训练表征。

## Planned Experiment: convnextv2_base_384_gap_llrd

- 状态: `待训练`
- 目标: 从优化器动力学入手，测试 `GAP + Strong CoarseDropout + LLRD` 能否在保留 LB `0.73631` 基线优势的同时降低随机分类头对预训练骨干的扰动。
- 操作: 删除 `GeM`，使用 timm 默认 GAP；恢复 `CoarseDropout(max_holes=8, max_size=64)`；`AdamW` 使用参数组，`head` 学习率为基础 LR，backbone 学习率为 `0.1 * LR`；继续保持动态 `pos_weight` 与 `[0.10, 0.90]` OOF 阈值搜索。
## EXP-20260528-125220-convnextv2_base_384_gap_llrd

- 时间: 2026-05-28T12:52:20+08:00
- 实验名: `convnextv2_base_384_gap_llrd`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=3.00e-05`, `backbone_lr=3.00e-06`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + Strong CoarseDropout(max_holes=8, max_size=64); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.995357 |
| 1 | 2148 | 1930 | 0.898510 | 0.984127 |
| 2 | 2148 | 1930 | 0.898510 | 0.993525 |
| 3 | 2148 | 1931 | 0.898976 | 0.995366 |
| 4 | 2148 | 1931 | 0.898976 | 0.990741 |

- OOF F1: `0.991272`
- OOF 最优阈值: `0.580000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_gap_llrd/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充

## Planned Experiment: convnextv2_base_384_llrd_tuned

- 状态: `待训练`
- 架构升级: 重构为 YAML 配置驱动，实验配置移入 `configs/` 目录，训练与推理通过 `--config` 读取同一份配置，提升复现能力并减少硬编码超参漂移。
- 超参调优: 将 `base_lr` 提升至 `5e-4` 以充分训练随机初始化分类头，backbone 通过 LLRD 使用 `5e-5`；将 `CoarseDropout` 从强遮挡削弱到 `max_holes=4, max_height=48, max_width=48`，缓解模型过度保守（上一轮 67 正例）的问题。
- 操作: 保持 GAP、动态 `pos_weight=negative_count/positive_count` 与 `[0.10, 0.90]` OOF 阈值搜索；新配置文件为 `configs/convnextv2_base_384_llrd_tuned.yaml`。
## EXP-20260528-154431-convnextv2_base_384_llrd_tuned

- 时间: 2026-05-28T15:44:31+08:00
- 实验名: `convnextv2_base_384_llrd_tuned`
- Config: `/root/project/stage2/configs/convnextv2_base_384_llrd_tuned.yaml`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss(pos_weight=fold_negative/fold_positive)`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=5.00e-04`, `backbone_lr=5.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=4, max_height=48, max_width=48); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | pos_weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | 0.995357 |
| 1 | 2148 | 1930 | 0.898510 | 0.992565 |
| 2 | 2148 | 1930 | 0.898510 | 0.991690 |
| 3 | 2148 | 1931 | 0.898976 | 0.995349 |
| 4 | 2148 | 1931 | 0.898976 | 0.987974 |

- OOF F1: `0.992199`
- OOF 最优阈值: `0.900000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_llrd_tuned/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 线上预测分布: `40` 个正例（用户反馈）
- 分析: `base_lr=5e-4` 后分类头学习能力恢复，`pos_weight=negative/positive≈0.898` 的负面压制作用被完整激活；模型快速拟合到偏向负类的 loss landscape，导致正类数量跌至谷底。
- 后续推断改进方向: 彻底废除动态 `pos_weight` 补偿，回归纯净 `BCEWithLogitsLoss()`；训练集正负已接近平衡，不需要人为压低正类损失。

## Planned Experiment: convnextv2_base_384_pure_bce

- 状态: `待训练`
- 纠偏方案: 使用 `pos_weight_mode: none`，构建无任何 `pos_weight` 参数的 plain `BCEWithLogitsLoss()`。
- 保持不变: GAP、LLRD (`head_lr=5e-4`, `backbone_lr=5e-5`)、轻度 `CoarseDropout(max_holes=4, max_height=48, max_width=48)` 与 `[0.10, 0.90]` OOF 阈值搜索。
- 启动配置: `configs/convnextv2_base_384_pure_bce.yaml`
## EXP-20260528-174739-convnextv2_base_384_pure_bce

- 时间: 2026-05-28T17:47:39+08:00
- 实验名: `convnextv2_base_384_pure_bce`
- Config: `/root/project/stage2/configs/convnextv2_base_384_pure_bce.yaml`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=5.00e-04`, `backbone_lr=5.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=4, max_height=48, max_width=48); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.995349 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.990689 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.994444 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.995340 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.990706 |

- OOF F1: `0.991822`
- OOF 最优阈值: `0.470000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_pure_bce/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充

## Planned Experiment: convnextv2_base_384_ema_smooth

- 状态: `待训练`
- 上轮 pure BCE 反馈: 预测正例回升至 `66`，说明卸除错误 `pos_weight` 和修复 `base_lr=5e-4` 有效；但 plain BCE 概率仍偏过度自信，OOF 阈值搜索不够平滑，尚未进入 `80-90` 正例的理想先验区间。
- 配置管理纪律: `configs/` 是实验快照目录，后续严禁重命名、覆盖或删除历史 YAML；本轮保留 `pure_bce`，并补回 `llrd_tuned` 历史快照，只新增 `configs/convnextv2_base_384_ema_smooth.yaml`。
- 优化方向: 引入 `EMA(decay=0.999)` 与 `label_smoothing=0.05`，在保持 GAP、LLRD、plain BCE 无 `pos_weight` 和轻度遮挡的前提下平滑概率分布，提高 OOF 阈值搜索稳定性。
## EXP-20260528-215017-convnextv2_base_384_ema_smooth

- 时间: 2026-05-28T21:50:17+08:00
- 实验名: `convnextv2_base_384_ema_smooth`
- Config: `/root/project/stage2/configs/convnextv2_base_384_ema_smooth.yaml`
- Backbone: `convnextv2_base.fcmae_ft_in22k_in1k`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Label Smoothing: `0.0500`
- EMA: `enabled=True, decay=0.999000`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=5.00e-04`, `backbone_lr=5.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=4, max_height=48, max_width=48); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.995357 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.989805 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.995366 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.995340 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.989767 |

- OOF F1: `0.991431`
- OOF 最优阈值: `0.620000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/convnextv2_base_384_ema_smooth/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
## EXP-PLANNED-cswin_base_384_ema_smooth
- **时间**: 2026-05-28
- **配置文件**: configs/cswin_base_384_ema_smooth.yaml
- **架构替换**: 拒绝 Top-K 作弊。在本地完全集成 CSWin-B 模型结构，不依赖 timm 自动注册，离线硬加载 `weights/cswin_base_384.pth`。利用十字交叉注意力捕捉宏观物理轮廓。
- **阈值校准**: 针对泄露导致的分数虚高，将验证集阈值搜索严格约束在理论中心点 `[0.40, 0.60]` 附近。
- **训练策略**: plain BCE + Label Smoothing 0.05 + EMA 0.999 + LLRD。
## EXP-20260529-004913-cswin_base_384_ema_smooth

- 时间: 2026-05-29T00:49:13+08:00
- 实验名: `cswin_base_384_ema_smooth`
- Config: `/root/project/stage2/configs/cswin_base_384_ema_smooth.yaml`
- Backbone: `cswin_base_384`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Label Smoothing: `0.0500`
- EMA: `enabled=True, decay=0.999000`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=5.00e-04`, `backbone_lr=5.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=4, max_height=48, max_width=48); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.995357 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.986916 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.990689 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.993476 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.982456 |

- OOF F1: `0.988467`
- OOF 最优阈值: `0.520000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/cswin_base_384_ema_smooth/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
## EXP-20260529-174941-cswin_base_384_ema_smooth_tuned

- 时间: 2026-05-29T17:49:41+08:00
- 实验名: `cswin_base_384_ema_smooth_tuned`
- Config: `/root/project/stage2/configs/cswin_base_384_ema_smooth_tuned.yaml`
- Backbone: `cswin_base_384`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Label Smoothing: `0.0500`
- EMA: `enabled=True, decay=0.999000`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=1.00e-04`, `backbone_lr=1.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=8, max_height=64, max_width=64); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.992551 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.985915 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.990724 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.990706 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.981584 |

- OOF F1: `0.987719`
- OOF 最优阈值: `0.560000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/cswin_base_384_ema_smooth_tuned/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
## EXP-20260529-193616-beit_base_patch16_384_baseline

- 时间: 2026-05-29T19:36:16+08:00
- 实验名: `beit_base_patch16_384_baseline`
- Config: `/root/project/stage2/configs/beit_base_384.yaml`
- Backbone: `beit_base_patch16_384`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Label Smoothing: `0.0500`
- EMA: `enabled=True, decay=0.999000`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=1.00e-04`, `backbone_lr=1.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=2, max_height=32, max_width=32); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.798238 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.812721 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.812183 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.820779 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.815074 |

- OOF F1: `0.808912`
- OOF 最优阈值: `0.410000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/beit_base_patch16_384_baseline/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
## EXP-20260529-230332-beit_base_patch16_384_baseline

- 时间: 2026-05-29T23:03:32+08:00
- 实验名: `beit_base_patch16_384_baseline`
- Config: `/root/project/stage2/configs/beit_base_384.yaml`
- Backbone: `beit_base_patch16_384`
- Pooling: `GAP (timm default)`
- 图像尺寸: `384 x 384`
- Loss: `BCEWithLogitsLoss()`
- pos_weight_mode: `none`
- Label Smoothing: `0.0500`
- EMA: `enabled=True, decay=0.999000`
- Optimizer: `AdamW` with LLRD
- LLRD: `head_lr=1.00e-04`, `backbone_lr=1.00e-05`
- Scheduler: `CosineAnnealingLR`
- 数据增强: LongestMaxSize(384) + zero PadIfNeeded(384) + ShiftScaleRotate + ColorJitter + RandomGamma + HueSaturationValue + CLAHE + Blur/GaussNoise/ImageCompression + CoarseDropout(max_holes=2, max_height=32, max_width=32); valid/test: aspect-safe resize-pad + Normalize only

| Fold | Train Positive | Train Negative | Class Ratio | Loss Weight | Best Valid F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2148 | 1930 | 0.898510 | plain_bce | 0.986940 |
| 1 | 2148 | 1930 | 0.898510 | plain_bce | 0.975610 |
| 2 | 2148 | 1930 | 0.898510 | plain_bce | 0.985102 |
| 3 | 2148 | 1931 | 0.898976 | plain_bce | 0.986965 |
| 4 | 2148 | 1931 | 0.898976 | plain_bce | 0.976526 |

- OOF F1: `0.981315`
- OOF 最优阈值: `0.450000`
- 阈值搜索: `0.10..0.90, step=0.01, maximize F1`
- 阈值文件: `/root/project/stage2/outputs/beit_base_patch16_384_baseline/optimal_threshold.json`
- Kaggle LB Score: `待补充`
- 后续推断改进方向: 待补充
