# STA326 Meteorite Identification

This repository contains our STA326 meteorite image classification pipeline. The task is formulated as binary image classification: given a test image, predict whether it belongs to the meteorite class.

The final reproducible inference pipeline uses a 5-fold SwinV2 ensemble. For each test image, the five fold checkpoints produce probabilities that are averaged, then the final submission is calibrated by selecting the top 86 images with the highest predicted probabilities as positive.

## Repository Structure

```text
configs/                         Experiment YAML files
scripts/run_prediction.sh        Inference entry point
scripts/run_experiment.sh        Training + inference entry point
src/
├── train.py                     K-fold training
├── predict.py                   5-fold ensemble inference
├── models/                      Model construction
└── utils/                       Config, dataset, metrics, checkpoint helpers
```

Large image data and checkpoint files are intentionally not tracked by Git. 

After preparing the competition data and model checkpoints, the working tree should also contain:

```text
data/
├── train_labels.csv
├── sample_submission.csv
├── train_images/train_images/
└── test_images/test_images/

weights/
└── checkpoints/
    ├── swinv2_base_384_minimal_pseudo/
    │   ├── fold_0_best.pth
    │   ├── fold_1_best.pth
    │   ├── fold_2_best.pth
    │   ├── fold_3_best.pth
    │   ├── fold_4_best.pth
    │   └── metadata.json
    └── swinv2_base_384_minimal_external/
        ├── fold_0_best.pth
        ├── fold_1_best.pth
        ├── fold_2_best.pth
        ├── fold_3_best.pth
        ├── fold_4_best.pth
        └── metadata.json
```


## Environment

The project uses `uv` for dependency management. From the repository root:

```bash
uv sync --frozen
```

The project metadata currently requires Python 3.14 or newer.

## Data Preparation

Place the competition data under `data/`:

```text
data/train_labels.csv
data/sample_submission.csv
data/train_images/train_images/<image files>
data/test_images/test_images/<image files>
```

## Checkpoint Preparation

The trained checkpoints are large and are provided with the project report. After obtaining the checkpoint archives, place the files as follows:

```bash
mkdir -p weights/checkpoints/swinv2_base_384_minimal_pseudo
unzip -o swinv2_base_384_minimal_pseudo.zip \
  -d weights/checkpoints/swinv2_base_384_minimal_pseudo
```

The directory used by this code should be:

```text
weights/checkpoints/swinv2_base_384_minimal_pseudo/
```

Each checkpoint directory must contain `fold_0_best.pth` through `fold_4_best.pth` and `metadata.json`.

## Reproduce Final Inference

The default inference command reproduces the final SwinV2 pseudo-label checkpoint submission:

```bash
bash scripts/run_prediction.sh
```

This is equivalent to:

```bash
uv run python src/predict.py \
  --config configs/swinv2_base_384_minimal_pseudo.yaml
```

The output files are written to:

```text
outputs/swinv2_base_384_minimal_pseudo/
├── submission_model_only.csv
├── submission_probabilities.csv
├── submission_count_calibrated.csv
└── submission.csv
```

`submission_model_only.csv` is the raw 5-fold ensemble prediction using the OOF threshold stored in checkpoint metadata. `submission.csv` is the final Top-86-calibrated submission.

## Training

To rerun training and then inference with the default final configuration:

```bash
bash scripts/run_experiment.sh configs/swinv2_base_384_minimal_pseudo.yaml
```

Training uses:

- Backbone: `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- Image size: `384 x 384`
- 5-fold cross validation
- EMA checkpoint saving
- Minimal aspect-safe preprocessing
- Optional pseudo-label training from `data/pseudo_labels/api_prediction.csv`

To train with pretrained initialization, place the offline timm backbone state dict at:

```text
weights/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k.pth
```

By default, the training script loads pretrained weights from this local path instead of downloading them at runtime. To train without pretrained initialization, pass `--no-pretrained` directly to `src/train.py`.

## Key Configurations

- `configs/swinv2_base_384_minimal_pseudo.yaml`: final default configuration
- `configs/swinv2_base_384_minimal_external.yaml`: auxiliary SwinV2 configuration
- `configs/convnextv2_base_384_data.yaml` and `configs/swin_base_384_data.yaml`: earlier baseline configurations

