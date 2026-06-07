"""Cascade YOLO-World and rembg background masking."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.preedit import (
    PROJECT_ROOT,
    Masker,
    MaskOutputDirectories,
    register_masker,
)

from .visualizer import save_comparison


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _env_float(name: str, default: float) -> float:
    """Read a floating-point tuning parameter from the environment."""
    try:
        return float(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是浮点数。") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean tuning parameter from the environment."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


class CascadeMasker(Masker):
    """Try YOLO-World first, then use rembg as a salient-object fallback."""

    YOLO_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov8l-worldv2.pt"
    RMBG_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "u2net.onnx"
    CLASS_NAMES = ("meteorite", "stone", "dark rock", "hand", "ruler", "fingers")
    BACKGROUND_COLOR = (128, 128, 128)
    PREDICTION_CONFIDENCE = 0.02
    MIN_BOX_CONFIDENCE = 0.015
    MAX_ASPECT_RATIO = 5.0

    def __init__(self) -> None:
        self._require_local_weight(
            self.YOLO_WEIGHTS_PATH,
            "缺少 YOLO-World 权重 weights/yolov8l-worldv2.pt，"
            "请手动下载并放入 weights/ 目录，禁止自动联网下载！",
        )
        self._require_local_weight(
            self.RMBG_WEIGHTS_PATH,
            "缺少 RMBG 权重 weights/u2net.onnx，"
            "请手动下载并放入 weights/ 目录，禁止自动联网下载！",
        )

        self.prediction_confidence = _env_float(
            "CASCADE_YOLO_PREDICTION_CONFIDENCE",
            self.PREDICTION_CONFIDENCE,
        )
        self.min_box_confidence = _env_float(
            "CASCADE_YOLO_MIN_BOX_CONFIDENCE",
            self.MIN_BOX_CONFIDENCE,
        )
        self.max_aspect_ratio = _env_float(
            "CASCADE_YOLO_MAX_ASPECT_RATIO",
            self.MAX_ASPECT_RATIO,
        )
        self._validate_tuning_parameters()
        self.skip_existing = _env_bool("CASCADE_SKIP_EXISTING", True)
        self.rembg_secondary_yolo = _env_bool("CASCADE_RMBG_SECONDARY_YOLO", True)
        self.excluded_image_ids = self._load_excluded_image_ids(
            os.getenv("CASCADE_EXCLUDE_FILE")
        )

        from ultralytics import YOLO

        self.yolo_model: Any = YOLO(self.YOLO_WEIGHTS_PATH)
        self.yolo_model.set_classes(list(self.CLASS_NAMES))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.environ["U2NET_HOME"] = str(PROJECT_ROOT / "weights")
        self._preload_packaged_cudnn()
        from rembg import new_session, remove

        try:
            self.rembg_session: Any = new_session("u2net")
        except Exception as exc:
            raise RuntimeError(
                "无法从本地 weights/u2net.onnx 初始化 RMBG session。"
            ) from exc
        self.rembg_remove: Any = remove
        self._warn_if_rembg_uses_cpu()

    def run(self, *, data_dir: Path, output_dirs: MaskOutputDirectories) -> None:
        """Mask every supported image, routing YOLO misses through rembg."""
        if not data_dir.is_dir():
            raise NotADirectoryError(f"输入图片目录不存在: {data_dir}")

        for image_path in self._iter_images(data_dir):
            file_name = image_path.name
            output_path = output_dirs.images / file_name
            failed_path = output_dirs.failed / file_name
            vis_path = output_dirs.visualizations / file_name

            if self._is_excluded(image_path):
                LOGGER.warning("跳过人工排除毒样本: %s", image_path)
                self._remove_output_artifacts(output_path, failed_path, vis_path)
                continue
            if self.skip_existing and output_path.is_file():
                self._remove_output_artifacts(failed_path)
                continue

            try:
                with Image.open(image_path) as source_image:
                    image = ImageOps.exif_transpose(source_image).convert("RGB")
            except (OSError, UnidentifiedImageError):
                LOGGER.exception("无法读取图片: %s", image_path)
                self._record_failure(image_path, output_path, failed_path)
                continue

            subject_box = self._predict_yolo_box(image, image_path)
            if subject_box is not None:
                masked_image = self._mask_with_box(image, subject_box)
                if self._save_success(
                    masked_image,
                    output_path=output_path,
                    failed_path=failed_path,
                ):
                    self._save_comparison(image, subject_box, masked_image, vis_path)
                    continue

            try:
                masked_image = self._mask_with_rembg(image)
            except Exception:
                LOGGER.exception("RMBG fallback 失败: %s", image_path)
                self._record_failure(image_path, output_path, failed_path)
                self._save_comparison(image, None, None, vis_path)
                continue

            try:
                secondary_box = self._predict_secondary_yolo_box(
                    masked_image,
                    image_path,
                )
                if secondary_box is not None:
                    masked_image = self._mask_with_box(masked_image, secondary_box)
            except Exception:
                LOGGER.exception(
                    "RMBG 后置 YOLO 处理失败，保留 RMBG alpha 输出: %s",
                    image_path,
                )
                secondary_box = None

            if not self._save_success(
                masked_image,
                output_path=output_path,
                failed_path=failed_path,
            ):
                self._record_failure(image_path, output_path, failed_path)
                self._save_comparison(image, None, None, vis_path)
                continue

            self._save_comparison(image, secondary_box, masked_image, vis_path)

    def _predict_yolo_box(
        self,
        image: Image.Image,
        image_path: Path,
    ) -> tuple[int, int, int, int] | None:
        """Return a valid YOLO subject box, or None to trigger rembg."""
        try:
            results = self.yolo_model.predict(
                image,
                conf=self.prediction_confidence,
                device=self.device,
                verbose=False,
            )
            return self._select_subject_box(results, image.size)
        except Exception:
            LOGGER.exception("YOLO-World 推理失败，转入 RMBG fallback: %s", image_path)
            return None

    def _predict_secondary_yolo_box(
        self,
        rembg_image: Image.Image,
        image_path: Path,
    ) -> tuple[int, int, int, int] | None:
        """Crop a successful RMBG mask when YOLO can isolate its main subject."""
        if not self.rembg_secondary_yolo:
            return None
        try:
            results = self.yolo_model.predict(
                rembg_image,
                conf=self.prediction_confidence,
                device=self.device,
                verbose=False,
            )
            return self._select_subject_box(results, rembg_image.size)
        except Exception:
            LOGGER.exception(
                "RMBG 后置 YOLO 推理失败，保留 RMBG alpha 输出: %s",
                image_path,
            )
            return None

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
                    or confidence < self.min_box_confidence
                ):
                    continue
                if aspect_ratio > self.max_aspect_ratio or class_name not in {
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
        """Yield supported images recursively in deterministic order."""
        for image_path in sorted(data_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                yield image_path

    def _is_excluded(self, image_path: Path) -> bool:
        """Return whether an image is listed for manual exclusion."""
        return (
            image_path.name in self.excluded_image_ids
            or image_path.stem in self.excluded_image_ids
        )

    @staticmethod
    def _load_excluded_image_ids(exclude_file: str | None) -> set[str]:
        """Load manually excluded image IDs, allowing names or bare stems."""
        if not exclude_file:
            return set()

        path = Path(exclude_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到人工排除列表: {path}")

        excluded_ids: set[str] = set()
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            image_id = raw_line.partition("#")[0].strip()
            if image_id:
                excluded_ids.add(Path(image_id).name)
        LOGGER.info("已加载 %d 个人工排除样本: %s", len(excluded_ids), path)
        return excluded_ids

    @staticmethod
    def _remove_output_artifacts(*paths: Path) -> None:
        """Remove stale artifacts without interrupting preprocessing."""
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("无法清理旧输出: %s", path)

    @classmethod
    def _mask_with_box(
        cls,
        image: Image.Image,
        subject_box: tuple[int, int, int, int],
    ) -> Image.Image:
        """Paste a rectangular YOLO foreground onto an RGB gray canvas."""
        masked_image = Image.new("RGB", image.size, cls.BACKGROUND_COLOR)
        masked_image.paste(image.crop(subject_box), subject_box[:2])
        return masked_image

    def _mask_with_rembg(self, image: Image.Image) -> Image.Image:
        """Paste an alpha-masked rembg foreground onto an RGB gray canvas."""
        removed = self.rembg_remove(image, session=self.rembg_session)
        if not isinstance(removed, Image.Image):
            raise TypeError("rembg.remove 未返回 PIL.Image.Image。")

        foreground = removed.convert("RGBA")
        if foreground.size != image.size:
            raise ValueError(
                f"RMBG 输出尺寸异常: 输入 {image.size}, 输出 {foreground.size}"
            )

        masked_image = Image.new("RGB", image.size, self.BACKGROUND_COLOR)
        alpha = foreground.getchannel("A")
        masked_image.paste(foreground.convert("RGB"), (0, 0), alpha)
        return masked_image

    @staticmethod
    def _save_success(
        masked_image: Image.Image,
        *,
        output_path: Path,
        failed_path: Path,
    ) -> bool:
        """Save one successful mask and remove a stale failed copy."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            masked_image.save(output_path)
            failed_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("无法保存掩码图片: %s", output_path)
            return False
        return True

    @classmethod
    def _record_failure(
        cls,
        image_path: Path,
        output_path: Path,
        failed_path: Path,
    ) -> None:
        """Copy one failed input and remove a stale successful mask."""
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.unlink(missing_ok=True)
            shutil.copy2(image_path, failed_path)
        except OSError:
            LOGGER.exception("无法记录失败图片: %s", image_path)

    @staticmethod
    def _save_comparison(
        image: Image.Image,
        subject_box: tuple[int, int, int, int] | None,
        masked_image: Image.Image | None,
        vis_path: Path,
    ) -> None:
        """Save a visualization without failing the masking pipeline."""
        try:
            save_comparison(image, subject_box, masked_image, vis_path)
        except OSError:
            LOGGER.exception("无法保存对比图: %s", vis_path)

    @staticmethod
    def _require_local_weight(weight_path: Path, message: str) -> None:
        """Refuse to initialize if an offline model weight is missing."""
        if not weight_path.is_file():
            print(f"\n{'=' * 80}\n{message}\n{'=' * 80}\n")
            raise FileNotFoundError(message)

    @staticmethod
    def _preload_packaged_cudnn() -> None:
        """Expose the uv-installed cuDNN shared library to ONNX Runtime."""
        try:
            import nvidia.cudnn
        except ImportError as exc:
            raise RuntimeError(
                "缺少 cuDNN GPU 运行库，请执行 `uv add nvidia-cudnn-cu12`。"
            ) from exc

        package_dir = Path(next(iter(nvidia.cudnn.__path__)))
        cudnn_library = package_dir / "lib" / "libcudnn.so.9"
        if not cudnn_library.is_file():
            raise FileNotFoundError(f"找不到 cuDNN 动态库: {cudnn_library}")
        try:
            ctypes.CDLL(str(cudnn_library), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"无法加载 cuDNN 动态库: {cudnn_library}") from exc

    def _warn_if_rembg_uses_cpu(self) -> None:
        """Warn clearly if ONNX Runtime could not activate its CUDA provider."""
        inner_session = getattr(self.rembg_session, "inner_session", None)
        providers = inner_session.get_providers() if inner_session is not None else []
        if "CUDAExecutionProvider" not in providers:
            LOGGER.warning(
                "RMBG 未启用 CUDAExecutionProvider，将回退到 CPU。当前 providers: %s",
                providers,
            )

    def _validate_tuning_parameters(self) -> None:
        """Reject unsafe tuning values before expensive model initialization."""
        if self.prediction_confidence <= 0:
            raise ValueError("CASCADE_YOLO_PREDICTION_CONFIDENCE 必须大于 0。")
        if self.min_box_confidence <= 0:
            raise ValueError("CASCADE_YOLO_MIN_BOX_CONFIDENCE 必须大于 0。")
        if self.max_aspect_ratio < 1:
            raise ValueError("CASCADE_YOLO_MAX_ASPECT_RATIO 必须大于等于 1。")


register_masker("cascade", lambda: CascadeMasker())
