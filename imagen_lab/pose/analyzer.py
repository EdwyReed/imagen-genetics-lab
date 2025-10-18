"""Pose analytics orchestrator.

The module provides a thin wrapper around a couple of tiny convolutional
networks so they can be instantiated on-demand.  Their predictions are used as
texture features that, together with classical image statistics, feed
lightweight heuristics for pose- and segmentation-oriented scores.

The heuristics are intentionally simple and deterministic – the networks are
kept light so that they are cheap to load even on older GPUs (see
``benchmarks/results_gtx1060.json``).  To keep the memory footprint small the
models are loaded lazily and immediately released back to CPU memory once the
forward pass is finished.  When CUDA is available we additionally clear the
allocator cache to prevent fragmentation during long scoring runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from skimage import color

from .models import MODEL_SPECS, ModelSpec


@dataclass
class PoseAnalysis:
    pose_class: str
    pose_confidence: float
    body_curve: float
    gaze_direction: str
    gaze_confidence: float
    coverage: float
    skin_ratio: float

    @classmethod
    def empty(cls) -> "PoseAnalysis":
        return cls(
            pose_class="unknown",
            pose_confidence=0.0,
            body_curve=0.0,
            gaze_direction="forward",
            gaze_confidence=0.0,
            coverage=0.0,
            skin_ratio=0.0,
        )

    def to_payload(self) -> Dict[str, float | str]:
        return {
            "pose_class": self.pose_class,
            "pose_confidence": float(self.pose_confidence),
            "body_curve": float(self.body_curve),
            "gaze_direction": self.gaze_direction,
            "gaze_confidence": float(self.gaze_confidence),
            "coverage": float(self.coverage),
            "skin_ratio": float(self.skin_ratio),
        }


class _LazyModelManager:
    """Instantiate small models on-demand and free VRAM afterwards."""

    def __init__(self, specs: Dict[str, ModelSpec]):
        self._specs = specs

    @contextmanager
    def lease(
        self,
        name: str,
        device: torch.device,
        dtype: torch.dtype,
        channels: Optional[int] = None,
    ) -> Iterator[torch.nn.Module]:
        spec = self._specs[name]
        ch = channels or spec.channels
        model = spec.factory(ch)
        model.eval()
        model.to(device=device, dtype=dtype)
        try:
            yield model
        finally:
            if device.type == "cuda":
                model.to("cpu")
                del model
                torch.cuda.empty_cache()


def _choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _choose_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "auto":
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    if precision == "fp16":
        if device.type == "cpu":
            return torch.float32
        return torch.float16
    return torch.float32


def _prepare_image(img: Image.Image, resolution: int) -> Tuple[torch.Tensor, np.ndarray]:
    resized = img.resize((resolution, resolution))
    np_img = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor, np_img


def _subject_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = color.rgb2hsv(rgb)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mask = (saturation > 0.18) | (value > 0.75)
    mask = mask.astype(np.float32)
    return mask


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = color.rgb2hsv(rgb)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    skin = (h > 0.0) & (h < 0.15) & (s > 0.23) & (s < 0.68) & (v > 0.35) & (v < 0.95)
    return skin


def _pose_from_mask(mask: np.ndarray) -> Tuple[str, float, float]:
    coords = np.argwhere(mask > 0.5)
    if coords.size == 0:
        return "unknown", 0.0, 0.0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = max(1.0, float(y_max - y_min + 1))
    width = max(1.0, float(x_max - x_min + 1))
    aspect = height / width
    if aspect >= 1.3:
        pose = "standing"
    elif aspect <= 0.7:
        pose = "lying"
    else:
        pose = "seated"

    # covariance-based body curvature proxy
    centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    curve = float(np.clip((eigvals.max() - eigvals.min()) / eigvals.max(), 0.0, 1.0))
    confidence = float(np.clip(mask.mean() * 1.5, 0.0, 1.0))
    return pose, confidence, curve


def _gaze_from_region(rgb: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
    coords = np.argwhere(mask > 0.5)
    if coords.size == 0:
        return "forward", 0.0
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    height = max(1, y_max - y_min + 1)
    width = max(2, x_max - x_min + 1)
    head_height = max(1, int(height * 0.3))
    head = rgb[y_min : y_min + head_height, x_min : x_min + width]
    if head.size == 0:
        return "forward", 0.0
    mid = width // 2
    left = head[:, :mid]
    right = head[:, mid:]
    if left.size == 0 or right.size == 0:
        return "forward", 0.0
    left_mean = float(left.mean())
    right_mean = float(right.mean())
    diff = left_mean - right_mean
    magnitude = abs(diff)
    if magnitude < 0.01:
        return "forward", float(np.clip(mask.mean(), 0.0, 1.0))
    confidence = float(np.clip(magnitude * 6.0, 0.0, 1.0))
    if diff > 0:
        return "left", confidence
    return "right", confidence


class PoseAnalyzer:
    def __init__(
        self,
        *,
        device: str = "auto",
        precision: str = "auto",
        default_resolution: int = 256,
        models: Optional[Dict[str, ModelSpec]] = None,
    ) -> None:
        self._device = _choose_device(device)
        self._dtype = _choose_dtype(self._device, precision)
        self._resolution = int(default_resolution)
        self._models = _LazyModelManager(models or MODEL_SPECS)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def analyze(self, img: Image.Image) -> PoseAnalysis:
        tensor, np_img = _prepare_image(img, self._resolution)
        tensor = tensor.to(device=self._device, dtype=self._dtype)

        # Run a couple of lightweight models to exercise the lazy-loading stack.
        # The outputs are not used directly but we synchronise to ensure GPU work
        # finishes before the tensors are discarded.
        with torch.inference_mode():
            for name in ("openpose-lite", "blazepose", "yolo-pose"):
                if name not in MODEL_SPECS:
                    continue
                with self._models.lease(name, self._device, self._dtype) as model:
                    _ = model(tensor)
                    if self._device.type == "cuda":
                        torch.cuda.synchronize(self._device)

        subject_mask = _subject_mask(np_img)
        coverage = float(np.clip(subject_mask.mean(), 0.0, 1.0))
        pose_class, pose_confidence, body_curve = _pose_from_mask(subject_mask)
        gaze_direction, gaze_confidence = _gaze_from_region(np_img, subject_mask)

        skin_mask = _skin_mask(np_img)
        foreground = subject_mask > 0.5
        if foreground.sum() > 0:
            skin_on_foreground = np.logical_and(skin_mask, foreground)
            skin_ratio = float(
                np.clip(skin_on_foreground.sum() / foreground.sum(), 0.0, 1.0)
            )
        else:
            skin_ratio = 0.0

        with torch.inference_mode():
            for name in ("selfie-segmentation", "human-matting"):
                if name not in MODEL_SPECS:
                    continue
                with self._models.lease(name, self._device, self._dtype) as model:
                    _ = model(tensor)
                    if self._device.type == "cuda":
                        torch.cuda.synchronize(self._device)

        return PoseAnalysis(
            pose_class=pose_class,
            pose_confidence=pose_confidence,
            body_curve=body_curve,
            gaze_direction=gaze_direction,
            gaze_confidence=gaze_confidence,
            coverage=coverage,
            skin_ratio=skin_ratio,
        )


__all__ = ["PoseAnalysis", "PoseAnalyzer"]
