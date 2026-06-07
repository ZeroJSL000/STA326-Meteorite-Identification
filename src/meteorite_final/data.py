"""Deterministic image loading and final inference preprocessing."""

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


def _pad_to_square(image_size: int) -> A.BasicTransform:
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


def final_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            _pad_to_square(image_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


class TestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        image_size: int,
        id_column: str,
    ) -> None:
        if id_column not in frame:
            raise KeyError(f"Missing image ID column: {id_column}")
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_column = id_column
        self.transform = final_transform(image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_id = str(self.frame.at[index, self.id_column])
        image_path = self.image_dir / image_id
        try:
            with Image.open(image_path) as image:
                array = np.asarray(image.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Unable to read image: {image_path}") from error
        return {
            "image": self.transform(image=array)["image"],
            "id": image_id,
            "index": torch.tensor(index),
        }
