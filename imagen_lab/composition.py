"""Composition analysis utilities based on lightweight object detection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, filters

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    label: Optional[int] = None


@dataclass
class CompositionMetrics:
    cropping_tightness: float
    thirds_alignment: float
    negative_space: float
    primary_bbox: Optional[BBox]
    num_detections: int
    saliency_mean: float
    saliency_coverage: float

    @classmethod
    def empty(cls) -> "CompositionMetrics":
        return cls(
            cropping_tightness=0.0,
            thirds_alignment=0.0,
            negative_space=1.0,
            primary_bbox=None,
            num_detections=0,
            saliency_mean=0.0,
            saliency_coverage=0.0,
        )

    def to_payload(self) -> dict:
        payload = {
            "cropping_tightness": float(self.cropping_tightness),
            "thirds_alignment": float(self.thirds_alignment),
            "negative_space": float(self.negative_space),
            "num_detections": int(self.num_detections),
            "saliency_mean": float(self.saliency_mean),
            "saliency_coverage": float(self.saliency_coverage),
        }
        if self.primary_bbox is not None:
            payload["primary_bbox"] = [float(x) for x in self.primary_bbox]
        return payload


class CompositionAnalyzer:
    """Estimates high-level composition metrics for illustrations."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_name: str = "yolov8n.pt",
        conf: float = 0.25,
        max_det: int = 8,
    ) -> None:
        self.device = device
        self.conf = conf
        self.max_det = max_det
        self._model = None
        self._use_half = False
        if YOLO is None:
            return
        try:
            self._model = YOLO(model_name)  # type: ignore[call-arg]
            if device.startswith("cuda"):
                self._model.to(device)
                self._use_half = True
        except Exception:
            self._model = None
            self._use_half = False

    @property
    def available(self) -> bool:
        return self._model is not None

    def analyze(self, image: Image.Image) -> CompositionMetrics:
        saliency = _compute_saliency(image)
        detections = self._detect(image)
        if not detections:
            fallback_bbox = _saliency_bbox(saliency)
            if fallback_bbox is not None:
                detections = [Detection(fallback_bbox, confidence=1.0, label=None)]
        return _metrics_from(saliency, detections)

    def _detect(self, image: Image.Image) -> List[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(  # type: ignore[operator]
                image,
                conf=self.conf,
                max_det=self.max_det,
                verbose=False,
                half=self._use_half,
                device=self.device,
            )
        except Exception:
            return []
        detections: List[Detection] = []
        width, height = image.size
        if not results:
            return detections
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return detections
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return detections
        xyxy_np = xyxy.cpu().numpy()
        confs = getattr(boxes, "conf", None)
        labels = getattr(boxes, "cls", None)
        conf_np = confs.cpu().numpy() if confs is not None else None
        label_np = labels.cpu().numpy() if labels is not None else None
        for idx, coords in enumerate(xyxy_np):
            x1, y1, x2, y2 = coords.tolist()
            norm_bbox = (
                float(np.clip(x1 / width, 0.0, 1.0)),
                float(np.clip(y1 / height, 0.0, 1.0)),
                float(np.clip(x2 / width, 0.0, 1.0)),
                float(np.clip(y2 / height, 0.0, 1.0)),
            )
            confidence = float(conf_np[idx]) if conf_np is not None else 0.0
            label = int(label_np[idx]) if label_np is not None else None
            detections.append(Detection(norm_bbox, confidence=confidence, label=label))
        return detections

    def visualize(self, image: Image.Image, metrics: CompositionMetrics) -> Image.Image:
        overlay = image.convert("RGBA")
        draw = ImageDraw.Draw(overlay, "RGBA")
        width, height = overlay.size
        thirds_x = [width / 3, 2 * width / 3]
        thirds_y = [height / 3, 2 * height / 3]
        for x in thirds_x:
            draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 80), width=1)
        for y in thirds_y:
            draw.line([(0, y), (width, y)], fill=(255, 255, 255, 80), width=1)
        if metrics.primary_bbox is not None:
            x1, y1, x2, y2 = metrics.primary_bbox
            box = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(box, outline=(255, 0, 0, 200), width=3)
        return overlay


def _compute_saliency(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = color.rgb2lab(arr)
    blurred = filters.gaussian(lab, sigma=3.0, channel_axis=2)
    mean = lab.mean(axis=(0, 1), keepdims=True)
    diff = blurred - mean
    saliency = np.linalg.norm(diff, axis=2)
    saliency -= saliency.min()
    max_val = saliency.max()
    if max_val > 1e-6:
        saliency /= max_val
    return saliency.astype(np.float32)


def _saliency_bbox(saliency: np.ndarray, threshold: float = 0.55) -> Optional[BBox]:
    if saliency.size == 0:
        return None
    mask = saliency >= (saliency.max() * threshold)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    height, width = saliency.shape
    x1 = float(xs.min() / width)
    y1 = float(ys.min() / height)
    x2 = float((xs.max() + 1) / width)
    y2 = float((ys.max() + 1) / height)
    return (x1, y1, x2, y2)


def _metrics_from(saliency: np.ndarray, detections: Sequence[Detection]) -> CompositionMetrics:
    if saliency.size == 0:
        return CompositionMetrics.empty()
    total_saliency = float(saliency.sum())
    saliency_mean = float(saliency.mean())
    salient_mask = saliency >= 0.5 * saliency.max()
    saliency_coverage = float(salient_mask.mean())
    if total_saliency <= 1e-6:
        empty = CompositionMetrics.empty()
        empty.saliency_mean = saliency_mean
        empty.saliency_coverage = saliency_coverage
        return empty

    primary = _select_primary_detection(detections, saliency)
    if primary is None:
        empty = CompositionMetrics.empty()
        empty.saliency_mean = saliency_mean
        empty.saliency_coverage = saliency_coverage
        empty.num_detections = len(detections)
        return empty

    x1, y1, x2, y2 = primary.bbox
    height, width = saliency.shape
    x1_px = max(0, min(width - 1, int(math.floor(x1 * width))))
    y1_px = max(0, min(height - 1, int(math.floor(y1 * height))))
    x2_px = max(0, min(width, int(math.ceil(x2 * width))))
    y2_px = max(0, min(height, int(math.ceil(y2 * height))))
    inside_saliency = float(saliency[y1_px:y2_px, x1_px:x2_px].sum())
    saliency_fraction = inside_saliency / total_saliency if total_saliency > 0 else 0.0

    margin = min(x1, y1, 1.0 - x2, 1.0 - y2)
    margin_norm = np.clip(margin / 0.25, 0.0, 1.0)
    cropping_tightness = float(np.clip((1.0 - margin_norm) * saliency_fraction, 0.0, 1.0))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    thirds_points = ((1.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0))
    dist = min(math.hypot(cx - tx, cy - ty) for tx, ty in thirds_points)
    max_dist = math.sqrt(2.0) / 3.0
    thirds_alignment = float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    negative_space = float(np.clip(1.0 - saliency_fraction, 0.0, 1.0))

    return CompositionMetrics(
        cropping_tightness=cropping_tightness,
        thirds_alignment=thirds_alignment,
        negative_space=negative_space,
        primary_bbox=primary.bbox,
        num_detections=len(detections),
        saliency_mean=saliency_mean,
        saliency_coverage=saliency_coverage,
    )


def _select_primary_detection(detections: Sequence[Detection], saliency: np.ndarray) -> Optional[Detection]:
    if not detections:
        return None
    height, width = saliency.shape
    best_score = -1.0
    best_det: Optional[Detection] = None
    total_saliency = float(saliency.sum())
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        x1_px = max(0, min(width - 1, int(math.floor(x1 * width))))
        y1_px = max(0, min(height - 1, int(math.floor(y1 * height))))
        x2_px = max(0, min(width, int(math.ceil(x2 * width))))
        y2_px = max(0, min(height, int(math.ceil(y2 * height))))
        region_saliency = float(saliency[y1_px:y2_px, x1_px:x2_px].sum())
        saliency_weight = region_saliency / total_saliency if total_saliency > 0 else 0.0
        area = (x2 - x1) * (y2 - y1)
        score = det.confidence * (0.6 * saliency_weight + 0.4 * area)
        if score > best_score:
            best_score = score
            best_det = det
    return best_det
