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
    train_csv: Path = field(init=False)
    sample_submission_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.train_image_dir = Path(
            os.getenv("TRAIN_IMAGE_DIR", self.data_dir / "train_images")
        )
        self.test_image_dir = Path(
            os.getenv("TEST_IMAGE_DIR", self.data_dir / "test_images")
        )
        self.train_csv = Path(
            os.getenv("TRAIN_CSV", self.data_dir / "train_labels.csv")
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


@dataclass
class AugmentConfig:
    """训练图像增强参数；验证与推理始终保持形态。"""

    mode: str = "strong"
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


@dataclass
class TrainConfig:
    """动态加权 BCE + LLRD AdamW + CosineAnnealingLR 的训练策略。"""

    seed: int = field(default_factory=lambda: _env_int("SEED", 2026))
    folds: int = field(default_factory=lambda: _env_int("N_FOLDS", 5))
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
    pos_weight_mode: str = "none"
    label_smoothing: float = 0.0
    ema_enabled: bool = False
    ema_decay: float = 0.999
    loss_name: str = "BCEWithLogitsLoss"
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"


@dataclass
class InferenceConfig:
    """折模型平均推理配置。"""

    batch_size: int = field(default_factory=lambda: _env_int("INFER_BATCH_SIZE", 24))
    threshold: float = field(
        default_factory=lambda: _env_float("FALLBACK_THRESHOLD", 0.5)
    )
    external_label_csv: Path | None = None
    external_label_score: float | None = None
    external_strategy: str = "none"
    external_blend_weight: float = 0.70
    external_override_margin: float = 0.30
    target_positive_count: int | None = None


@dataclass
class PseudoLabelConfig:
    """将外部 test-set 预测作为低权重 pseudo labels 参与训练。"""

    enabled: bool = False
    csv: Path | None = None
    weight: float = 0.40


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
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pseudo_label: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)

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
    if not 0.0 <= config.train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing 必须在 [0, 1) 范围内。")
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
    if "mode" in augmentation:
        config.augment.mode = str(augmentation["mode"]).lower()
    if config.augment.mode not in {"strong", "minimal"}:
        raise ValueError("augmentation.mode 仅支持 strong 或 minimal。")
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
    if "external_label_csv" in inference:
        config.inference.external_label_csv = _as_path(inference["external_label_csv"])
    if "external_label_score" in inference:
        config.inference.external_label_score = float(inference["external_label_score"])
    if "external_strategy" in inference:
        config.inference.external_strategy = str(inference["external_strategy"]).lower()
    if "external_blend_weight" in inference:
        config.inference.external_blend_weight = float(inference["external_blend_weight"])
    if "external_override_margin" in inference:
        config.inference.external_override_margin = float(
            inference["external_override_margin"]
        )
    if "target_positive_count" in inference:
        value = inference["target_positive_count"]
        config.inference.target_positive_count = None if value is None else int(value)
    if config.inference.external_strategy not in {"none", "blend", "guarded"}:
        raise ValueError("inference.external_strategy 仅支持 none、blend 或 guarded。")
    if not 0.0 <= config.inference.external_blend_weight <= 1.0:
        raise ValueError("inference.external_blend_weight 必须在 [0, 1] 范围内。")
    if not 0.0 <= config.inference.external_override_margin <= 1.0:
        raise ValueError("inference.external_override_margin 必须在 [0, 1] 范围内。")
    if config.inference.target_positive_count is not None and config.inference.target_positive_count < 0:
        raise ValueError("inference.target_positive_count 必须非负。")

    pseudo_label = payload.get("pseudo_label", {}) or {}
    if not isinstance(pseudo_label, dict):
        raise TypeError("pseudo_label 配置必须是 mapping。")
    if "enabled" in pseudo_label:
        config.pseudo_label.enabled = bool(pseudo_label["enabled"])
    if "csv" in pseudo_label:
        config.pseudo_label.csv = _as_path(pseudo_label["csv"])
    if "weight" in pseudo_label:
        config.pseudo_label.weight = float(pseudo_label["weight"])
    if config.pseudo_label.enabled and config.pseudo_label.csv is None:
        raise ValueError("pseudo_label.enabled=true 时必须设置 pseudo_label.csv。")
    if not 0.0 < config.pseudo_label.weight <= 1.0:
        raise ValueError("pseudo_label.weight 必须在 (0, 1] 范围内。")

    paths = payload.get("paths", {}) or {}
    if not isinstance(paths, dict):
        raise TypeError("paths 配置必须是 mapping。")
    for key in ("data_dir", "weights_dir", "outputs_dir", "docs_dir"):
        if key in paths:
            setattr(
                config.paths, key, _as_path(paths[key]) or getattr(config.paths, key)
            )
    config.paths.__post_init__()


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
