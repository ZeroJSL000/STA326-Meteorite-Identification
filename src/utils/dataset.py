"""二分类图像数据集与保持物体比例的增强流程。"""

from __future__ import annotations

import warnings
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment

from utils.config import AugmentConfig


def _pad_to_square(image_size: int) -> A.BasicTransform:
    """使用零填充保持原始宽高比并兼容增强库版本差异。"""

    try:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=1.0,
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=1.0,
        )


def _aspect_safe_resize(image_size: int) -> list[A.BasicTransform]:
    """严格执行 LongestMaxSize 后零填充，不拉伸石头形态。"""

    return [
        A.LongestMaxSize(max_size=image_size, p=1.0),
        _pad_to_square(image_size),
    ]


def _shift_scale_rotate(probability: float) -> A.BasicTransform:
    try:
        return A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=probability,
        )
    except TypeError:
        return A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.12,
            rotate_limit=20,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=probability,
        )


def _image_compression() -> A.BasicTransform:
    try:
        return A.ImageCompression(quality_range=(75, 98), p=1.0)
    except TypeError:
        return A.ImageCompression(quality_lower=75, quality_upper=98, p=1.0)


def _coarse_dropout(image_size: int, config: AugmentConfig) -> A.BasicTransform:
    """按配置遮挡局部伪特征，降低对标尺、手指等噪声的记忆。"""

    min_size = max(1, int(image_size * 0.03))
    max_height = min(config.coarse_dropout_max_height, image_size)
    max_width = min(config.coarse_dropout_max_width, image_size)
    if config.coarse_dropout_max_holes <= 0:
        raise ValueError("coarse_dropout_max_holes 必须大于 0。")
    try:
        return A.CoarseDropout(
            num_holes_range=(1, config.coarse_dropout_max_holes),
            hole_height_range=(min_size, max_height),
            hole_width_range=(min_size, max_width),
            fill=0,
            p=config.coarse_dropout_probability,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=config.coarse_dropout_max_holes,
            max_height=max_height,
            max_width=max_width,
            min_height=min_size,
            min_width=min_size,
            fill_value=0,
            p=config.coarse_dropout_probability,
        )


class TorchvisionRandAugment(A.ImageOnlyTransform):
    """将 torchvision RandAugment 接入 Albumentations 图像流水线。"""

    def __init__(self, num_ops: int, magnitude: int, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.transform = RandAugment(num_ops=num_ops, magnitude=magnitude)

    def apply(self, image: np.ndarray, **params: object) -> np.ndarray:
        return np.asarray(self.transform(Image.fromarray(image)))

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("num_ops", "magnitude")


def build_train_transforms(image_size: int, config: AugmentConfig) -> A.Compose:
    """构建 GAP + LLRD 实验的 aspect-safe 强增强。"""

    transforms: list[A.BasicTransform] = [
        *_aspect_safe_resize(image_size),
        _shift_scale_rotate(config.geometry_probability),
        A.ColorJitter(
            brightness=config.color_jitter,
            contrast=config.color_jitter,
            saturation=config.color_jitter,
            hue=0.12,
            p=0.75,
        ),
        A.RandomGamma(gamma_limit=(70, 135), p=0.45),
        A.HueSaturationValue(
            hue_shift_limit=18,
            sat_shift_limit=32,
            val_shift_limit=22,
            p=0.5,
        ),
        A.CLAHE(
            clip_limit=(1.0, 2.5),
            tile_grid_size=(8, 8),
            p=config.clahe_probability,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(p=1.0),
                _image_compression(),
            ],
            p=config.degradation_probability,
        ),
    ]
    if config.coarse_dropout_enabled:
        transforms.append(_coarse_dropout(image_size, config))
    if config.randaugment_enabled:
        transforms.append(
            TorchvisionRandAugment(
                num_ops=config.randaugment_num_ops,
                magnitude=config.randaugment_magnitude,
            )
        )
    transforms.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(transforms)


def build_valid_transforms(image_size: int, tta_name: str = "identity") -> A.Compose:
    """验证与推理只做保持比例的缩放、零填充和标准化。"""

    if tta_name != "identity":
        raise ValueError("严格 Baseline 推理仅允许 identity 预处理。")
    return A.Compose(
        [
            *_aspect_safe_resize(image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


class BinaryImageDataset(Dataset):
    """读取 CSV 指定图像并转换为单标签二分类样本。"""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        transform: A.Compose,
        id_col: str = "id",
        label_col: str | None = "label",
        soft_masking_enabled: bool = False,
        soft_masking_alpha: float = 0.7,
        original_image_dir: Path | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.id_col = id_col
        self.label_col = label_col
        self.soft_masking_enabled = soft_masking_enabled
        self.soft_masking_alpha = float(soft_masking_alpha)
        self.original_image_dir = (
            Path(original_image_dir) if original_image_dir is not None else None
        )
        self._soft_masking_warnings: set[str] = set()
        if id_col not in self.frame:
            raise KeyError(f"数据表中没有图像 ID 列: {id_col}")
        if label_col is not None and label_col not in self.frame:
            raise KeyError(f"数据表中没有标签列: {label_col}")
        if not 0.0 <= self.soft_masking_alpha <= 1.0:
            raise ValueError("soft_masking_alpha 必须在 [0, 1] 范围内。")

    def __len__(self) -> int:
        return len(self.frame)

    @staticmethod
    def _load_rgb(image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"读取图像失败: {image_path}") from error

    def _warn_soft_masking(self, key: str, message: str) -> None:
        if key not in self._soft_masking_warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._soft_masking_warnings.add(key)

    def _load_image(self, image_id: str, image_dir: Path | None = None, use_soft_masking: bool = True) -> np.ndarray:
        masked_path = (image_dir or self.image_dir) / image_id
        masked_image = self._load_rgb(masked_path)
        if not self.soft_masking_enabled or not use_soft_masking:
            return masked_image
        if self.original_image_dir is None:
            self._warn_soft_masking(
                "missing_original_dir",
                "Soft-Masking 已启用，但未配置可用的原图目录；回退到纯掩码图。",
            )
            return masked_image
        original_path = self.original_image_dir / image_id
        try:
            original_image = self._load_rgb(original_path)
        except RuntimeError:
            self._warn_soft_masking(
                "missing_original_image",
                f"Soft-Masking 找不到原图 {original_path}；回退到纯掩码图。",
            )
            return masked_image
        if original_image.shape != masked_image.shape:
            self._warn_soft_masking(
                "original_shape_mismatch",
                "Soft-Masking 原图与掩码图尺寸不一致；将原图缩放到掩码图尺寸。",
            )
            original_image = cv2.resize(
                original_image,
                (masked_image.shape[1], masked_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        alpha = self.soft_masking_alpha
        return np.clip(
            original_image.astype(np.float32) * (1.0 - alpha)
            + masked_image.astype(np.float32) * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        image_id = str(self.frame.at[index, self.id_col])
        image_dir = None
        if "image_dir" in self.frame:
            image_dir = Path(str(self.frame.at[index, "image_dir"]))
        use_soft_masking = True
        if "use_soft_masking" in self.frame:
            use_soft_masking = bool(self.frame.at[index, "use_soft_masking"])
        image_array = self._load_image(image_id, image_dir, use_soft_masking)
        sample: dict[str, Tensor | str | int] = {
            "image": self.transform(image=image_array)["image"],
            "id": image_id,
            "index": index,
        }
        if self.label_col is not None:
            sample["target"] = torch.tensor(
                float(self.frame.at[index, self.label_col]),
                dtype=torch.float32,
            )
        if "sample_weight" in self.frame:
            sample["sample_weight"] = torch.tensor(
                float(self.frame.at[index, "sample_weight"]),
                dtype=torch.float32,
            )
        return sample
