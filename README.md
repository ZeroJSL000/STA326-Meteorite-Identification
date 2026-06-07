# STA326 Meteorite Identification - Final

这是清洗后的最终发布版。它只保留一个明确、可复现的任务：

> 使用 5 折 SwinV2 checkpoint 做纯模型概率平均，再选择概率最高的 86 张测试图作为陨石。

推理阶段不读取外部/API 标签，不执行 Guarded Fusion，也不包含旧实验分支的后处理逻辑。

## 项目结构

```text
configs/final.json                 # 唯一运行配置
data/                              # 发布所需训练集与测试集
src/meteorite_final/               # 最终推理包
scripts/run_final.sh               # 一键运行入口
scripts/verify_assets.py           # 数据与 checkpoint 检查
weights/checkpoints/final/         # 本地放置 5 折 checkpoint
results/reference/                 # 已验证运行的参考清单
```

所有代码路径都相对于仓库根目录解析，也可以通过 CLI 参数覆盖，不依赖
`/Work/project/...` 等机器路径。

## 数据集

仓库中的 `data/` 包含最终流程需要的数据：

- `train_images/`：1472 张训练图
- `train_labels.csv`：训练标签
- `test_images/`：194 张最终测试图
- `sample_submission.csv`：提交模板

数据集总计约 480 MiB。请确保数据的使用和再分发符合原始任务许可。

## Checkpoint

最终模型由以下文件组成：

```text
weights/checkpoints/final/
├── metadata.json
├── fold_0_best.pth
├── fold_1_best.pth
├── fold_2_best.pth
├── fold_3_best.pth
└── fold_4_best.pth
```

每个 checkpoint 约 332 MiB，超过 GitHub 普通 Git 的单文件 100 MiB 限制，因此不直接
提交到代码仓库。将 checkpoint 放入上述目录后运行：

```bash
uv run python scripts/verify_assets.py
```

## 环境

需要 Python 3.11+、PyTorch 和可选 CUDA GPU：

```bash
uv sync
```

## 运行最终推理

```bash
bash scripts/run_final.sh
```

可覆盖运行路径和资源参数：

```bash
bash scripts/run_final.sh \
  --data-dir data \
  --checkpoint-dir weights/checkpoints/final \
  --output-dir results/final \
  --batch-size 32 \
  --num-workers 8
```

输出：

```text
results/final/
├── submission_probabilities.csv
├── submission_top86_no_guarded.csv
└── manifest.json
```

已验证参考结果：

- 测试样本：194
- 正类数量：86
- Top-86 截止概率：`0.5249254107475281`
- Guarded Fusion：关闭
- 推理外部标签：未使用

## 路径管理

默认路径集中定义于 `src/meteorite_final/config.py`：

- 数据：`data/`
- checkpoint：`weights/checkpoints/final/`
- 输出：`results/final/`
- 配置：`configs/final.json`

CLI 参数只覆盖本次运行，不修改源码或配置文件。
