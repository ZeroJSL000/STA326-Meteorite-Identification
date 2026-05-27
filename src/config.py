"""集中维护严格 Baseline 训练、验证和推理配置。"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


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
    """字段名与固定输入分辨率。"""

    id_col: str = "id"
    label_col: str = "label"
    image_size: int = 384
    num_workers: int = field(default_factory=lambda: _env_int("NUM_WORKERS", 8))


@dataclass
class ModelConfig:
    """历史最佳 Baseline 上接入 GeM 的 ConvNeXtV2 单 logit 分类器。"""

    name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        )
    )
    drop_rate: float = field(default_factory=lambda: _env_float("DROP_RATE", 0.0))
    drop_path_rate: float = field(
        default_factory=lambda: _env_float("DROP_PATH_RATE", 0.0)
    )
    gem_p: float = field(default_factory=lambda: _env_float("GEM_P", 3.0))
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
    """仅作用于训练图像的强增强参数。

    验证与推理始终保持形态。
    """

    color_jitter: float = 0.45
    clahe_probability: float = 0.6
    geometry_probability: float = 0.5
    degradation_probability: float = 0.35
    coarse_dropout_probability: float = 0.45
    tta_names: tuple[str, ...] = ("identity",)


@dataclass
class TrainConfig:
    """动态加权 BCE + AdamW + CosineAnnealingLR 的训练策略。"""

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
    min_learning_rate: float = field(
        default_factory=lambda: _env_float("MIN_LEARNING_RATE", 2.0e-7)
    )
    weight_decay: float = field(
        default_factory=lambda: _env_float("WEIGHT_DECAY", 0.05)
    )
    patience: int = field(default_factory=lambda: _env_int("PATIENCE", 7))
    max_grad_norm: float = 1.0
    amp: bool = field(default_factory=lambda: _env_bool("AMP", True))
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


@dataclass
class ExperimentConfig:
    """一次 Baseline 实验的完整配置。"""

    experiment_name: str = field(
        default_factory=lambda: os.getenv(
            "EXPERIMENT_NAME",
            "convnextv2_base_384_gem_rollback",
        )
    )
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

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


def load_config(validate_paths: bool = False) -> ExperimentConfig:
    """创建配置，并可选检查数据与离线权重是否齐备。"""

    config = ExperimentConfig()
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
