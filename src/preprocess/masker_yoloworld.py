"""YOLO-World based zero-shot background masking."""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.preedit import (
    PROJECT_ROOT,
    Masker,
    MaskOutputDirectories,
    register_masker,
)

from .visualizer import save_comparison


LOGGER = logging.getLogger(__name__)


class YoloWorldMasker(Masker):
    """Mask image backgrounds using locally loaded YOLO-World weights."""

    WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov8l-worldv2.pt"
    CLASS_NAMES = ("meteorite", "stone", "dark rock", "hand", "ruler", "fingers")
    BACKGROUND_COLOR = (128, 128, 128)
    PREDICTION_CONFIDENCE = 0.02
    MIN_BOX_CONFIDENCE = 0.015
    MAX_ASPECT_RATIO = 5.0

    def __init__(self) -> None:
        if not self.WEIGHTS_PATH.is_file():
            message = (
                "请前往官方渠道下载 yolov8l-worldv2.pt 并放置于项目的 "
                "weights/ 目录下，禁止代码自动下载！"
            )
            print(f"\n{'=' * 80}\n{message}\n{'=' * 80}\n")
            raise FileNotFoundError(message)

        from ultralytics import YOLO

        self.model: Any = YOLO(self.WEIGHTS_PATH)
        self.model.set_classes(list(self.CLASS_NAMES))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, *, data_dir: Path, output_dirs: MaskOutputDirectories) -> None:
        """Mask all JPEG images under data_dir while isolating failed inputs."""
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {data_dir}")

        for image_path in self._iter_images(data_dir):
            file_name = image_path.name
            output_path = output_dirs.images / file_name
            failed_path = output_dirs.failed / file_name
            vis_path = output_dirs.visualizations / file_name

            try:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGB")
            except (OSError, UnidentifiedImageError):
                LOGGER.exception("Unable to read image: %s", image_path)
                self._copy_failed_image(image_path, failed_path)
                continue

            try:
                results = self.model.predict(
                    image_path,
                    conf=self.PREDICTION_CONFIDENCE,
                    device=self.device,
                    verbose=False,
                )
                subject_box = self._select_subject_box(results, image.size)
            except Exception:
                LOGGER.exception("YOLO-World inference failed: %s", image_path)
                self._copy_failed_image(image_path, failed_path)
                self._save_failed_comparison(image, vis_path)
                continue

            if subject_box is None:
                self._copy_failed_image(image_path, failed_path)
                self._save_failed_comparison(image, vis_path)
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                masked_image = self._save_masked_image(image, subject_box, output_path)
            except OSError:
                LOGGER.exception("Unable to save masked image: %s", output_path)
                self._copy_failed_image(image_path, failed_path)
                self._save_failed_comparison(image, vis_path)
                continue

            try:
                save_comparison(image, subject_box, masked_image, vis_path)
            except OSError:
                LOGGER.exception("Unable to save comparison image: %s", vis_path)

    def _select_subject_box(
        self,
        results: Any,
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        """Return the largest valid meteorite or rock-like box."""
        image_width, image_height = image_size
        largest_box: tuple[int, int, int, int] | None = None
        largest_area = 0

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            coordinates = boxes.xyxy.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            for coordinates_xyxy, class_id, confidence in zip(
                coordinates,
                class_ids,
                confidences,
                strict=True,
            ):
                class_index = int(class_id)
                if not 0 <= class_index < len(self.CLASS_NAMES):
                    continue

                class_name = self.CLASS_NAMES[class_index]
                x1, y1, x2, y2 = (float(value) for value in coordinates_xyxy)
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue

                aspect_ratio = max(box_width / box_height, box_height / box_width)
                if (
                    class_name in {"hand", "ruler", "fingers"}
                    or confidence < self.MIN_BOX_CONFIDENCE
                ):
                    continue

                if aspect_ratio > self.MAX_ASPECT_RATIO or class_name not in {
                    "meteorite",
                    "stone",
                    "dark rock",
                }:
                    continue

                clipped_box = (
                    max(0, math.floor(x1)),
                    max(0, math.floor(y1)),
                    min(image_width, math.ceil(x2)),
                    min(image_height, math.ceil(y2)),
                )
                left, top, right, bottom = clipped_box
                area = (right - left) * (bottom - top)
                if area > largest_area:
                    largest_box = clipped_box
                    largest_area = area

        return largest_box

    @classmethod
    def _iter_images(cls, data_dir: Path) -> Iterator[Path]:
        for image_path in sorted(data_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg"}:
                yield image_path

    @classmethod
    def _save_masked_image(
        cls,
        image: Image.Image,
        subject_box: tuple[int, int, int, int],
        output_path: Path,
    ) -> Image.Image:
        masked_image = Image.new("RGB", image.size, cls.BACKGROUND_COLOR)
        masked_image.paste(image.crop(subject_box), subject_box[:2])
        masked_image.save(output_path)
        return masked_image

    @staticmethod
    def _save_failed_comparison(image: Image.Image, vis_path: Path) -> None:
        try:
            save_comparison(image, None, None, vis_path)
        except OSError:
            LOGGER.exception("Unable to save failed comparison image: %s", vis_path)

    @staticmethod
    def _copy_failed_image(image_path: Path, failed_path: Path) -> None:
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(image_path, failed_path)
        except OSError:
            LOGGER.exception("Unable to copy failed image: %s", image_path)


register_masker("yoloworld", lambda: YoloWorldMasker())
