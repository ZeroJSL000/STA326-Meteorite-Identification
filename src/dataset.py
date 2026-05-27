"""二分类图像数据集与保持物体比例的增强流程。"""

from __future__ import annotations

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

from config import AugmentConfig


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


def _coarse_dropout(image_size: int, probability: float) -> A.BasicTransform:
    """增强局部遮挡，降低对标尺、手指等局部伪特征的依赖。"""

    min_size = int(image_size * 0.03)
    max_size = min(64, image_size)
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(min_size, max_size),
            hole_width_range=(min_size, max_size),
            fill=0,
            p=probability,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=8,
            max_height=max_size,
            max_width=max_size,
            min_height=min_size,
            min_width=min_size,
            fill_value=0,
            p=probability,
        )


def build_train_transforms(image_size: int, config: AugmentConfig) -> A.Compose:
    """构建回退基线上的 aspect-safe 强增强。"""

    return A.Compose(
        [
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
            _coarse_dropout(image_size, config.coarse_dropout_probability),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


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
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.id_col = id_col
        self.label_col = label_col
        if id_col not in self.frame:
            raise KeyError(f"数据表中没有图像 ID 列: {id_col}")
        if label_col is not None and label_col not in self.frame:
            raise KeyError(f"数据表中没有标签列: {label_col}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        image_id = str(self.frame.at[index, self.id_col])
        image_path = self.image_dir / image_id
        try:
            with Image.open(image_path) as image:
                image_array = np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"读取图像失败: {image_path}") from error
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
        return sample
