"""集中维护训练、验证和推理配置，并支持 YAML 实验覆盖。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as error:
    raise ImportError("缺少 PyYAML，请先执行 `uv add pyyaml`。") from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class PathConfig:
    """项目输入、输出及实验日志路径。"""

    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    )
    weights_dir: Path = field(
        default_factory=lambda: Path(os.getenv("WEIGHTS_DIR", PROJECT_ROOT / "weights"))
    )
    outputs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
    )
    docs_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DOCS_DIR", PROJECT_ROOT / "docs"))
    )
    train_image_dir: Path = field(init=False)
    test_image_dir: Path = field(init=False)
    original_image_dir: Path = field(init=False)
    train_csv: Path = field(init=False)
    pseudo_labels_csv: Path = field(init=False)
    sample_submission_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.train_image_dir = Path(
            os.getenv("TRAIN_IMAGE_DIR", self.data_dir / "train_images")
        )
        self.test_image_dir = Path(
            os.getenv("TEST_IMAGE_DIR", self.data_dir / "test_images")
        )
        self.original_image_dir = Path(
            os.getenv("ORIGINAL_IMAGE_DIR", self.data_dir / "train_images")
        )
        self.train_csv = Path(
            os.getenv("TRAIN_CSV", self.data_dir / "train_labels.csv")
        )
        self.pseudo_labels_csv = Path(
            os.getenv("PSEUDO_LABELS_CSV", self.data_dir / "pseudo_labels.csv")
        )
        self.sample_submission_csv = Path(
            os.getenv(
                "SAMPLE_SUBMISSION_CSV",
                self.data_dir / "sample_submission.csv",
            )
        )


@dataclass
class DataConfig:
    """字段名、输入分辨率和 DataLoader 参数。"""

    id_col: str = "id"
    label_col: str = "label"
    image_size: int = 384
    num_workers: int = field(default_factory=lambda: _env_int("NUM_WORKERS", 8))


@dataclass
class ModelConfig:
    """timm GAP 单 logit 分类器配置。"""

    name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        )
    )
    pooling: str = "gap"
    drop_rate: float = field(default_factory=lambda: _env_float("DROP_RATE", 0.0))
    drop_path_rate: float = field(
        default_factory=lambda: _env_float("DROP_PATH_RATE", 0.0)
    )
    use_pretrained: bool = field(
        default_factory=lambda: _env_bool("USE_PRETRAINED", True)
    )
    pretrained_file: str = field(
        default_factory=lambda: os.getenv(
            "PRETRAINED_FILE",
            "convnextv2_base.fcmae_ft_in22k_in1k.pth",
        )
    )
    patch_size: int = 32


@dataclass
class AugmentConfig:
    """训练图像增强参数；验证与推理始终保持形态。"""

    color_jitter: float = 0.45
    clahe_probability: float = 0.6
    geometry_probability: float = 0.5
    degradation_probability: float = 0.35
    coarse_dropout_enabled: bool = True
    coarse_dropout_probability: float = 0.45
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_height: int = 64
    coarse_dropout_max_width: int = 64
    tta_names: tuple[str, ...] = ("identity",)
    randaugment_enabled: bool = False
    randaugment_num_ops: int = 9
    randaugment_magnitude: int = 10
    mixup_enabled: bool = False
    mixup_probability: float = 0.5
    mixup_alpha: float = 0.4
    cutmix_enabled: bool = False
    cutmix_probability: float = 0.5
    cutmix_alpha: float = 1.0
    multiscale_train_enabled: bool = False
    multiscale_train_sizes: tuple[int, ...] = (320, 352, 384, 416, 448)


@dataclass
class PreprocessingConfig:
    """可选的掩码图预处理策略。"""

    soft_masking_enabled: bool = False
    soft_masking_alpha: float = 0.7


@dataclass
class TrainConfig:
    """动态加权 BCE + LLRD AdamW + CosineAnnealingLR 的训练策略。"""

    seed: int = field(default_factory=lambda: _env_int("SEED", 2026))
    folds: int = field(default_factory=lambda: _env_int("N_FOLDS", 5))
    fold_limit: int | None = None
    epochs: int = field(default_factory=lambda: _env_int("EPOCHS", 24))
    batch_size: int = field(default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 8))
    valid_batch_size: int = field(
        default_factory=lambda: _env_int("VALID_BATCH_SIZE", 16)
    )
    accumulation_steps: int = field(
        default_factory=lambda: _env_int("GRAD_ACCUMULATION_STEPS", 2)
    )
    learning_rate: float = field(
        default_factory=lambda: _env_float("LEARNING_RATE", 3.0e-5)
    )
    backbone_lr_factor: float = field(
        default_factory=lambda: _env_float("BACKBONE_LR_FACTOR", 0.1)
    )
    min_learning_rate: float = field(
        default_factory=lambda: _env_float("MIN_LEARNING_RATE", 2.0e-7)
    )
    weight_decay: float = field(
        default_factory=lambda: _env_float("WEIGHT_DECAY", 0.05)
    )
    patience: int = field(default_factory=lambda: _env_int("PATIENCE", 7))
    max_grad_norm: float = 1.0
    amp: bool = field(default_factory=lambda: _env_bool("AMP", True))
    amp_dtype: str = field(default_factory=lambda: os.getenv("AMP_DTYPE", "float16"))
    pos_weight_mode: str = "none"
    label_smoothing: float = 0.0
    ema_enabled: bool = False
    ema_decay: float = 0.999
    loss_name: str = "BCEWithLogitsLoss"
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    use_logit_adjustment: bool = False
    train_pos_prior: float = 0.5
    hard_mining_enabled: bool = False
    hard_mining_topk: float = 0.1
    hard_mining_boost: float = 3.0
    hard_mining_oof_path: Path | None = None
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"


@dataclass
class InferenceConfig:
    """折模型平均推理配置。"""

    batch_size: int = field(default_factory=lambda: _env_int("INFER_BATCH_SIZE", 24))
    threshold: float = field(
        default_factory=lambda: _env_float("FALLBACK_THRESHOLD", 0.5)
    )
    tta_enabled: bool = False
    tta_scales: tuple[int, ...] = (384, 448, 512)
    tta_flips: tuple[str, ...] = ("none", "horizontal")

@dataclass
class PseudoLabelConfig:
    """Co-teaching 伪标签筛选与重训练参数。"""

    enabled: bool = False
    high_confidence_threshold: float = 0.95
    low_confidence_threshold: float = 0.05
    co_teaching_enabled: bool = True
    co_teaching_agreement_threshold: float = 0.90
    co_teaching_model_dirs: tuple[str, ...] = ("outputs/convnextv2", "outputs/eva02")
    pseudo_sample_weight: float = 0.5
    pseudo_lr_factor: float = 0.33
    pseudo_epoch_factor: float = 0.5
    init_checkpoint_dir: Path | None = None


@dataclass
class EnsembleConfig:
    """校准阶段使用的集成 OOF 与测试概率路径。"""

    oof_predictions: Path | None = None
    test_probabilities: Path | None = None


@dataclass
class CalibrationConfig:
    """温度缩放、label-shift 修正与全局阈值策略。"""

    enabled: bool = False
    temperature_scaling: bool = False
    temperature_min: float = 0.1
    temperature_max: float = 5.0
    prior_pos_rate: float = 0.438
    use_bayesian_correction: bool = False
    train_pos_prior: float = 0.5
    threshold_strategy: str = "auto"
    f1_threshold_step: float = 0.001
    threshold_iterations: int = 64
    probability_epsilon: float = 1.0e-6
    threshold_tie_breaker: float = 0.5



@dataclass
class ExperimentConfig:
    """一次实验的完整配置。"""

    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "EXPERIMENT_NAME",
            "convnextv2_base_384_ema_smooth",
        )
    )
    config_path: Path | None = None
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pseudo_labeling: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    @property
    def experiment_output_dir(self) -> Path:
        return self.paths.outputs_dir / self.experiment_name

    @property
    def checkpoint_dir(self) -> Path:
        return self.paths.weights_dir / "checkpoints" / self.experiment_name

    @property
    def pretrained_path(self) -> Path:
        return self.paths.weights_dir / self.model.pretrained_file

    @property
    def experiment_log_path(self) -> Path:
        return self.paths.docs_dir / "experiment_log.md"

    def create_output_dirs(self) -> None:
        self.paths.weights_dir.mkdir(parents=True, exist_ok=True)
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.docs_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的 JSON 结构。"""

        def normalize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items()}
            return value

        return normalize(asdict(self))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 YAML 配置文件: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML 顶层必须是 mapping: {path}")
    return payload


def _apply_yaml(config: ExperimentConfig, payload: dict[str, Any], path: Path) -> None:
    """将 YAML 字段映射到 dataclass 配置对象。"""

    config.config_path = path
    if "experiment_name" in payload:
        config.experiment_name = str(payload["experiment_name"])

    model = payload.get("model", {}) or {}
    if not isinstance(model, dict):
        raise TypeError("model 配置必须是 mapping。")
    if "backbone" in model:
        config.model.name = str(model["backbone"])
    if "name" in model:
        config.model.name = str(model["name"])
    if "image_size" in model:
        config.data.image_size = int(model["image_size"])
    if "pooling" in model:
        config.model.pooling = str(model["pooling"]).lower()
    if config.model.pooling != "gap":
        raise ValueError("当前稳定管线仅支持 pooling: gap。")
    if "pretrained_file" in model:
        config.model.pretrained_file = str(model["pretrained_file"])
    elif config.model.name == "cswin_base_384":
        config.model.pretrained_file = "cswin_base_384.pth"
    if "patch_size" in model:
        config.model.patch_size = int(model["patch_size"])
    if "drop_rate" in model:
        config.model.drop_rate = float(model["drop_rate"])
    if "drop_path_rate" in model:
        config.model.drop_path_rate = float(model["drop_path_rate"])

    train = payload.get("train", {}) or {}
    if not isinstance(train, dict):
        raise TypeError("train 配置必须是 mapping。")
    if "batch_size" in train:
        config.train.batch_size = int(train["batch_size"])
    if "valid_batch_size" in train:
        config.train.valid_batch_size = int(train["valid_batch_size"])
    if "epochs" in train:
        config.train.epochs = int(train["epochs"])
    if "base_lr" in train:
        config.train.learning_rate = float(train["base_lr"])
    if "learning_rate" in train:
        config.train.learning_rate = float(train["learning_rate"])
    if "backbone_lr_factor" in train:
        config.train.backbone_lr_factor = float(train["backbone_lr_factor"])
    if "pos_weight_mode" in train:
        config.train.pos_weight_mode = str(train["pos_weight_mode"]).lower()
    if "label_smoothing" in train:
        config.train.label_smoothing = float(train["label_smoothing"])
    if "amp" in train:
        config.train.amp = bool(train["amp"])
    if "amp_dtype" in train:
        config.train.amp_dtype = str(train["amp_dtype"]).lower()
    if config.train.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("train.amp_dtype 仅支持 float16 或 bfloat16。")
    if not 0.0 <= config.train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing 必须在 [0, 1) 范围内。")
    if "loss_type" in train:
        config.train.loss_type = str(train["loss_type"]).lower()
    if "focal_gamma" in train:
        config.train.focal_gamma = float(train["focal_gamma"])
    if "focal_alpha" in train:
        config.train.focal_alpha = float(train["focal_alpha"])
    if "use_logit_adjustment" in train:
        config.train.use_logit_adjustment = bool(train["use_logit_adjustment"])
    if "train_pos_prior" in train:
        config.train.train_pos_prior = float(train["train_pos_prior"])
    if config.train.loss_type not in {"bce", "focal"}:
        raise ValueError("train.loss_type 仅支持 bce 或 focal。")
    if config.train.focal_gamma < 0.0:
        raise ValueError("train.focal_gamma 必须大于等于 0。")
    if not 0.0 <= config.train.focal_alpha <= 1.0:
        raise ValueError("train.focal_alpha 必须在 [0, 1] 范围内。")
    if not 0.0 < config.train.train_pos_prior < 1.0:
        raise ValueError("train.train_pos_prior 必须在 (0, 1) 范围内。")
    if "hard_mining_enabled" in train:
        config.train.hard_mining_enabled = bool(train["hard_mining_enabled"])
    if "hard_mining_topk" in train:
        config.train.hard_mining_topk = float(train["hard_mining_topk"])
    if "hard_mining_boost" in train:
        config.train.hard_mining_boost = float(train["hard_mining_boost"])
    if "hard_mining_oof_path" in train:
        config.train.hard_mining_oof_path = _as_path(train["hard_mining_oof_path"])
    if not 0.0 < config.train.hard_mining_topk <= 0.5:
        raise ValueError("train.hard_mining_topk 必须在 (0, 0.5] 范围内。")
    if config.train.hard_mining_boost <= 0.0:
        raise ValueError("train.hard_mining_boost 必须大于 0。")
    if config.train.pos_weight_mode not in {"dynamic", "none"}:
        raise ValueError("train.pos_weight_mode 仅支持 dynamic 或 none。")
    for key, attr, caster in (
        ("folds", "folds", int),
        ("seed", "seed", int),
        ("accumulation_steps", "accumulation_steps", int),
        ("weight_decay", "weight_decay", float),
        ("min_lr", "min_learning_rate", float),
        ("min_learning_rate", "min_learning_rate", float),
        ("patience", "patience", int),
        ("max_grad_norm", "max_grad_norm", float),
    ):
        if key in train:
            setattr(config.train, attr, caster(train[key]))
    if "fold_limit" in train:
        config.train.fold_limit = int(train["fold_limit"])

    ema = payload.get("ema", {}) or train.get("ema", {}) or {}
    if not isinstance(ema, dict):
        raise TypeError("ema 配置必须是 mapping。")
    if "enabled" in ema:
        config.train.ema_enabled = bool(ema["enabled"])
    if "decay" in ema:
        config.train.ema_decay = float(ema["decay"])
    if not 0.0 < config.train.ema_decay < 1.0:
        raise ValueError("ema.decay 必须在 (0, 1) 范围内。")

    augmentation = payload.get("augmentation", {}) or {}
    if not isinstance(augmentation, dict):
        raise TypeError("augmentation 配置必须是 mapping。")
    coarse = augmentation.get("coarse_dropout", {}) or {}
    if not isinstance(coarse, dict):
        raise TypeError("augmentation.coarse_dropout 必须是 mapping。")
    if "enabled" in coarse:
        config.augment.coarse_dropout_enabled = bool(coarse["enabled"])
    if "probability" in coarse:
        config.augment.coarse_dropout_probability = float(coarse["probability"])
    if "max_holes" in coarse:
        config.augment.coarse_dropout_max_holes = int(coarse["max_holes"])
    if "max_height" in coarse:
        config.augment.coarse_dropout_max_height = int(coarse["max_height"])
    if "max_width" in coarse:
        config.augment.coarse_dropout_max_width = int(coarse["max_width"])

    randaugment = augmentation.get("randaugment", {}) or {}
    if not isinstance(randaugment, dict):
        raise TypeError("augmentation.randaugment 必须是 mapping。")
    if "enabled" in randaugment:
        config.augment.randaugment_enabled = bool(randaugment["enabled"])
    if "num_ops" in randaugment:
        config.augment.randaugment_num_ops = int(randaugment["num_ops"])
    if "magnitude" in randaugment:
        config.augment.randaugment_magnitude = int(randaugment["magnitude"])
    for key, attr, caster in (
        ("mixup_enabled", "mixup_enabled", bool),
        ("mixup_prob", "mixup_probability", float),
        ("mixup_alpha", "mixup_alpha", float),
        ("cutmix_enabled", "cutmix_enabled", bool),
        ("cutmix_prob", "cutmix_probability", float),
        ("cutmix_alpha", "cutmix_alpha", float),
        ("multiscale_train_enabled", "multiscale_train_enabled", bool),
    ):
        if key in augmentation:
            setattr(config.augment, attr, caster(augmentation[key]))
    if "multiscale_train_sizes" in augmentation:
        config.augment.multiscale_train_sizes = tuple(
            int(size) for size in augmentation["multiscale_train_sizes"]
        )
    if config.model.patch_size <= 0:
        raise ValueError("model.patch_size 必须大于 0。")
    if config.augment.randaugment_num_ops <= 0:
        raise ValueError("augmentation.randaugment.num_ops 必须大于 0。")
    if config.augment.randaugment_magnitude < 0:
        raise ValueError("augmentation.randaugment.magnitude 必须大于等于 0。")
    if not 0.0 <= config.augment.mixup_probability <= 1.0:
        raise ValueError("augmentation.mixup_prob 必须在 [0, 1] 范围内。")
    if not 0.0 <= config.augment.cutmix_probability <= 1.0:
        raise ValueError("augmentation.cutmix_prob 必须在 [0, 1] 范围内。")
    if config.augment.mixup_alpha <= 0.0 or config.augment.cutmix_alpha <= 0.0:
        raise ValueError("MixUp 与 CutMix alpha 必须大于 0。")
    enabled_probability_sum = 0.0
    if config.augment.mixup_enabled:
        enabled_probability_sum += config.augment.mixup_probability
    if config.augment.cutmix_enabled:
        enabled_probability_sum += config.augment.cutmix_probability
    if enabled_probability_sum > 1.0:
        raise ValueError("启用的 MixUp 与 CutMix 概率之和不能超过 1。")
    if not config.augment.multiscale_train_sizes:
        raise ValueError("augmentation.multiscale_train_sizes 不能为空。")

    preprocessing = payload.get("preprocessing", {}) or {}
    if not isinstance(preprocessing, dict):
        raise TypeError("preprocessing 配置必须是 mapping。")
    soft_masking = preprocessing.get("soft_masking", {}) or {}
    if not isinstance(soft_masking, dict):
        raise TypeError("preprocessing.soft_masking 必须是 mapping。")
    if "enabled" in soft_masking:
        config.preprocessing.soft_masking_enabled = bool(soft_masking["enabled"])
    if "mask_alpha" in soft_masking:
        config.preprocessing.soft_masking_alpha = float(soft_masking["mask_alpha"])
    if not 0.0 <= config.preprocessing.soft_masking_alpha <= 1.0:
        raise ValueError("preprocessing.soft_masking.mask_alpha 必须在 [0, 1] 范围内。")

    data = payload.get("data", {}) or {}
    if not isinstance(data, dict):
        raise TypeError("data 配置必须是 mapping。")
    if "num_workers" in data:
        config.data.num_workers = int(data["num_workers"])
    if "id_col" in data:
        config.data.id_col = str(data["id_col"])
    if "label_col" in data:
        config.data.label_col = str(data["label_col"])

    inference = payload.get("inference", {}) or {}
    if not isinstance(inference, dict):
        raise TypeError("inference 配置必须是 mapping。")
    if "batch_size" in inference:
        config.inference.batch_size = int(inference["batch_size"])
    if "threshold" in inference:
        config.inference.threshold = float(inference["threshold"])
    test = payload.get("test", {}) or {}
    if not isinstance(test, dict):
        raise TypeError("test 配置必须是 mapping。")
    if "tta_enabled" in test:
        config.inference.tta_enabled = bool(test["tta_enabled"])
    if "tta_scales" in test:
        config.inference.tta_scales = tuple(int(scale) for scale in test["tta_scales"])
    if "tta_flips" in test:
        config.inference.tta_flips = tuple(str(flip).lower() for flip in test["tta_flips"])
    if not config.inference.tta_scales:
        raise ValueError("test.tta_scales 不能为空。")
    if any(scale <= 0 for scale in config.inference.tta_scales):
        raise ValueError("test.tta_scales 必须全部大于 0。")
    supported_flips = {"none", "horizontal"}
    if not config.inference.tta_flips or not set(config.inference.tta_flips) <= supported_flips:
        raise ValueError(f"test.tta_flips 仅支持: {sorted(supported_flips)}")

    paths = payload.get("paths", {}) or {}
    pseudo_labeling = payload.get("pseudo_labeling", {}) or {}
    ensemble = payload.get("ensemble", {}) or {}
    calibration = payload.get("calibration", {}) or {}
    if not isinstance(pseudo_labeling, dict):
        raise TypeError("pseudo_labeling 配置必须是 mapping。")
    for key, attr, caster in (
        ("enabled", "enabled", bool),
        ("high_confidence_threshold", "high_confidence_threshold", float),
        ("low_confidence_threshold", "low_confidence_threshold", float),
        ("co_teaching_enabled", "co_teaching_enabled", bool),
        ("co_teaching_agreement_threshold", "co_teaching_agreement_threshold", float),
        ("pseudo_sample_weight", "pseudo_sample_weight", float),
        ("pseudo_lr_factor", "pseudo_lr_factor", float),
        ("pseudo_epoch_factor", "pseudo_epoch_factor", float),
    ):
        if key in pseudo_labeling:
            setattr(config.pseudo_labeling, attr, caster(pseudo_labeling[key]))
    if "co_teaching_model_dirs" in pseudo_labeling:
        config.pseudo_labeling.co_teaching_model_dirs = tuple(
            str(directory) for directory in pseudo_labeling["co_teaching_model_dirs"]
        )
    if "init_checkpoint_dir" in pseudo_labeling:
        config.pseudo_labeling.init_checkpoint_dir = _as_path(pseudo_labeling["init_checkpoint_dir"])
    if not 0.0 < config.pseudo_labeling.high_confidence_threshold < 1.0:
        raise ValueError("pseudo_labeling.high_confidence_threshold 必须在 (0, 1) 范围内。")
    if not 0.0 < config.pseudo_labeling.low_confidence_threshold < 1.0:
        raise ValueError("pseudo_labeling.low_confidence_threshold 必须在 (0, 1) 范围内。")
    if not 0.0 < config.pseudo_labeling.co_teaching_agreement_threshold < 1.0:
        raise ValueError("pseudo_labeling.co_teaching_agreement_threshold 必须在 (0, 1) 范围内。")
    if config.pseudo_labeling.pseudo_sample_weight <= 0.0:
        raise ValueError("pseudo_labeling.pseudo_sample_weight 必须大于 0。")
    if config.pseudo_labeling.pseudo_lr_factor <= 0.0:
        raise ValueError("pseudo_labeling.pseudo_lr_factor 必须大于 0。")
    if config.pseudo_labeling.pseudo_epoch_factor <= 0.0:
        raise ValueError("pseudo_labeling.pseudo_epoch_factor 必须大于 0。")

    if not isinstance(ensemble, dict):
        raise TypeError("ensemble 配置必须是 mapping。")
    if "oof_predictions" in ensemble:
        config.ensemble.oof_predictions = _as_path(ensemble["oof_predictions"])
    if "test_probabilities" in ensemble:
        config.ensemble.test_probabilities = _as_path(ensemble["test_probabilities"])

    if not isinstance(calibration, dict):
        raise TypeError("calibration 配置必须是 mapping。")
    for key, attr, caster in (
        ("enabled", "enabled", bool),
        ("temperature_scaling", "temperature_scaling", bool),
        ("temperature_min", "temperature_min", float),
        ("temperature_max", "temperature_max", float),
        ("prior_pos_rate", "prior_pos_rate", float),
        ("use_bayesian_correction", "use_bayesian_correction", bool),
        ("train_pos_prior", "train_pos_prior", float),
        ("threshold_strategy", "threshold_strategy", str),
        ("f1_threshold_step", "f1_threshold_step", float),
        ("threshold_iterations", "threshold_iterations", int),
        ("probability_epsilon", "probability_epsilon", float),
        ("threshold_tie_breaker", "threshold_tie_breaker", float),
    ):
        if key in calibration:
            setattr(config.calibration, attr, caster(calibration[key]))
    if not 0.0 < config.calibration.temperature_min < config.calibration.temperature_max:
        raise ValueError("calibration 温度搜索区间非法。")
    if not 0.0 < config.calibration.prior_pos_rate < 1.0:
        raise ValueError("calibration.prior_pos_rate 必须在 (0, 1) 范围内。")
    if not 0.0 < config.calibration.train_pos_prior < 1.0:
        raise ValueError("calibration.train_pos_prior 必须在 (0, 1) 范围内。")
    if config.calibration.threshold_strategy not in {"auto", "f1_optimal", "prior_aligned"}:
        raise ValueError("calibration.threshold_strategy 仅支持 auto、f1_optimal 或 prior_aligned。")
    if not 0.0 < config.calibration.f1_threshold_step <= 1.0:
        raise ValueError("calibration.f1_threshold_step 必须在 (0, 1] 范围内。")
    if config.calibration.threshold_iterations <= 0:
        raise ValueError("calibration.threshold_iterations 必须大于 0。")
    if not 0.0 < config.calibration.probability_epsilon < 0.5:
        raise ValueError("calibration.probability_epsilon 必须在 (0, 0.5) 范围内。")
    if not 0.0 <= config.calibration.threshold_tie_breaker <= 1.0:
        raise ValueError("calibration.threshold_tie_breaker 必须在 [0, 1] 范围内。")

    if not isinstance(paths, dict):
        raise TypeError("paths 配置必须是 mapping。")
    for key in ("data_dir", "weights_dir", "outputs_dir", "docs_dir"):
        if key in paths:
            setattr(
                config.paths, key, _as_path(paths[key]) or getattr(config.paths, key)
            )
    config.paths.__post_init__()
    if "dataset_dir" in paths:
        dataset_dir = _as_path(paths["dataset_dir"])
        if dataset_dir is None:
            raise ValueError("paths.dataset_dir 不能为空。")
        config.paths.train_csv = dataset_dir / "train_labels.csv"
        config.paths.train_image_dir = dataset_dir / "train_images"
        config.paths.test_image_dir = dataset_dir / "test_images"
        config.paths.pseudo_labels_csv = dataset_dir / "pseudo_labels.csv"
    for key in (
        "train_csv",
        "train_image_dir",
        "test_image_dir",
        "original_image_dir",
        "pseudo_labels_csv",
        "sample_submission_csv",
    ):
        if key in paths:
            setattr(config.paths, key, _as_path(paths[key]) or getattr(config.paths, key))


def resolve_image_dir(directory: Path, expected_image: str | None = None) -> Path:
    """兼容目录中重复嵌套一层文件夹的竞赛数据布局。"""

    candidates = (directory, directory / directory.name)
    for candidate in candidates:
        if expected_image and (candidate / expected_image).is_file():
            return candidate
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.[jJpP][pPnN][gG]")):
            return candidate
    raise FileNotFoundError(
        f"未找到图像目录或样例文件 {expected_image!r}，已检查: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def load_config(
    config_path: str | Path | None = None,
    validate_paths: bool = False,
) -> ExperimentConfig:
    """创建配置，并可选从 YAML 覆盖实验超参。"""

    config = ExperimentConfig()
    path = _as_path(config_path)
    if path is not None:
        _apply_yaml(config, _load_yaml(path), path)
    config.create_output_dirs()
    if validate_paths:
        required_files = (
            config.paths.train_csv,
            config.paths.sample_submission_csv,
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"缺少必要数据文件: {missing}")
        if config.model.use_pretrained and not config.pretrained_path.is_file():
            raise FileNotFoundError(
                f"缺少离线预训练权重: {config.pretrained_path}。"
                "请先执行 README.md 中的权重下载命令。"
            )
    return config
