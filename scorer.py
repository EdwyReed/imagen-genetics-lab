# scorer.py
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import opennsfw2 as n2
import torch
from PIL import Image
from skimage import color, filters
import open_clip

from imagen_lab.pose import PoseAnalysis, PoseAnalyzer
from imagen_lab.composition import CompositionAnalyzer, CompositionMetrics

from imagen_lab.scoring import (
    DEFAULT_STYLE_WEIGHTS,
    ClipTextHeadsConfig,
    StyleComposition,
    StyleMixer,
    WeightProfileTable,
    load_clip_text_heads,
)

# =========================
# Config & Text Anchors
# =========================

_CLIP_TEXT_HEADS: ClipTextHeadsConfig = load_clip_text_heads()
_CLIP_ARCH = _CLIP_TEXT_HEADS.clip_model

# Температура softmax для повышения контраста классов (ниже — контрастнее)
DEFAULT_TAU: float = 0.07

# Калибровка распределений (опционально). Можно проставить (p20, p80) после быстрых измерений.
DEFAULT_CAL_STYLE: Optional[Tuple[float, float]] = None  # например (0.35, 0.75)
DEFAULT_CAL_ILLU: Optional[Tuple[float, float]] = None  # например (0.45, 0.85)

# Веса итоговой style-метрики (композиция из трёх компонент)
W_CLIP, W_SPEC, W_ILLU = (
    DEFAULT_STYLE_WEIGHTS["clip"],
    DEFAULT_STYLE_WEIGHTS["spec"],
    DEFAULT_STYLE_WEIGHTS["illu"],
)

# Версия схемы логов для SQLite/JSONL. Увеличивайте при изменении структуры записей.
SCORES_SCHEMA_VERSION = 4


@dataclass
class AutoWeightsSettings:
    enabled: bool = False
    ema_alpha: float = 0.25
    momentum: float = 0.35
    target: float = 0.85
    min_component: float = 0.05
    min_weight: float = 0.05
    max_weight: float = 0.9
    min_gain: float = 0.4
    max_gain: float = 2.5
    initial_level: float = 0.7

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]] | None) -> "AutoWeightsSettings":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            ema_alpha=float(raw.get("ema_alpha", 0.25)),
            momentum=float(raw.get("momentum", 0.35)),
            target=float(raw.get("target", 0.85)),
            min_component=float(raw.get("min_component", 0.05)),
            min_weight=float(raw.get("min_weight", 0.05)),
            max_weight=float(raw.get("max_weight", 0.9)),
            min_gain=float(raw.get("min_gain", 0.4)),
            max_gain=float(raw.get("max_gain", 2.5)),
            initial_level=float(raw.get("initial_level", 0.7)),
        )





@dataclass
class ScoreResult:
    nsfw: float
    style: float
    clip_style: float
    specular: float
    illu_bias: float
    clip_heads: Dict[str, Dict[str, Any]]
    specular_metrics: SpecularMetrics
    embedding: Optional[np.ndarray] = None
    style_contributions: Dict[str, float] = field(default_factory=dict)
    style_weights: Dict[str, float] = field(default_factory=dict)
    pose_metrics: PoseAnalysis = field(default_factory=PoseAnalysis.empty)
    composition_metrics: CompositionMetrics = field(default_factory=CompositionMetrics.empty)


@dataclass
class ScoredImage:
    path: Path
    nsfw: int
    style: int
    clip_style: int
    specular: int
    illu_bias: int
    embedding: Optional[np.ndarray] = None
    style_raw: float = 0.0
    clip_style_raw: float = 0.0
    specular_raw: float = 0.0
    illu_bias_raw: float = 0.0
    style_contributions: Dict[str, float] = field(default_factory=dict)
    style_weights: Dict[str, float] = field(default_factory=dict)
    pose_class: str = "unknown"
    pose_confidence: int = 0
    body_curve: int = 0
    gaze_direction: str = "forward"
    gaze_confidence: int = 0
    coverage: int = 0
    skin_ratio: int = 0
    cropping_tightness: int = 0
    thirds_alignment: int = 0
    negative_space: int = 100
    composition_raw: Dict[str, object] = field(default_factory=dict)

    def fitness(self, w_style: float, w_nsfw: float) -> float:
        return float(w_style * float(self.style) + w_nsfw * float(self.nsfw))

# =========================
# SQLite helpers
# =========================

def _ensure_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS scores(
      id INTEGER PRIMARY KEY,
      path TEXT UNIQUE,
      ts INTEGER,
      nsfw REAL,
      style REAL,
      clip_style REAL,
      specular REAL,
      illu_bias REAL,
      notes TEXT,
      schema_version INTEGER NOT NULL DEFAULT 1
    )
    """)
    cur.execute("PRAGMA table_info(scores)")
    existing_columns = {row[1] for row in cur.fetchall()}
    if "schema_version" not in existing_columns:
        cur.execute(
            "ALTER TABLE scores ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1"
        )
    conn.commit()
    conn.close()


def _insert_db(db_path: Path, rows: List[Tuple]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO scores(path, ts, nsfw, style, clip_style, specular, illu_bias, notes, schema_version)
    VALUES(?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


# =========================
# Specular / Wetness index
# =========================

@dataclass(frozen=True)
class MaskRange:
    lower: Tuple[float, float, float]
    upper: Tuple[float, float, float]


@dataclass
class MaskDefinition:
    name: str
    space: str = "hsv"
    ranges: Tuple[MaskRange, ...] = ()
    min_coverage: float = 0.0

    def build_mask(self, spaces: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.ranges:
            rgb_data = spaces.get("rgb")
            if rgb_data is None:
                raise ValueError("RGB space must be provided for default mask")
            return np.ones(rgb_data.shape[:2], dtype=np.float32)
        if self.space not in spaces:
            raise ValueError(f"Unsupported color space '{self.space}' for mask '{self.name}'")
        data = spaces[self.space]
        mask = np.zeros(data.shape[:2], dtype=bool)
        for rng in self.ranges:
            lower = np.array(rng.lower, dtype=np.float32)
            upper = np.array(rng.upper, dtype=np.float32)
            cond = np.all((data >= lower) & (data <= upper), axis=-1)
            mask |= cond
        return mask.astype(np.float32)


@dataclass
class SpecularZoneMetrics:
    coverage: float = 0.0
    highlight_ratio: float = 0.0
    highlight_density: float = 0.0
    sharpness: float = 0.0
    score: float = 0.0

    def to_payload(self) -> Dict[str, float]:
        return {
            "coverage": float(self.coverage),
            "highlight_ratio": float(self.highlight_ratio),
            "highlight_density": float(self.highlight_density),
            "sharpness": float(self.sharpness),
            "score": float(self.score),
        }


@dataclass
class SpecularAggregate:
    weighted_score: float = 0.0
    mean_score: float = 0.0
    coverage: float = 0.0

    def to_payload(self) -> Dict[str, float]:
        return {
            "weighted_score": float(self.weighted_score),
            "mean_score": float(self.mean_score),
            "coverage": float(self.coverage),
        }


@dataclass
class SpecularMetrics:
    score: float
    highlight_ratio: float
    sharpness: float
    zones: Dict[str, SpecularZoneMetrics] = field(default_factory=dict)
    aggregate: SpecularAggregate = field(default_factory=SpecularAggregate)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "highlight_ratio": float(self.highlight_ratio),
            "sharpness": float(self.sharpness),
            "zones": {name: zone.to_payload() for name, zone in self.zones.items()},
            "aggregate": self.aggregate.to_payload(),
        }

    @classmethod
    def empty(cls) -> "SpecularMetrics":
        return cls(0.0, 0.0, 0.0, {}, SpecularAggregate())


DEFAULT_SPECULAR_MASKS: Tuple[MaskDefinition, ...] = (
    MaskDefinition(
        name="skin",
        space="hsv",
        ranges=(
            MaskRange(lower=(0.0, 0.2, 0.2), upper=(0.14, 0.75, 0.97)),
            MaskRange(lower=(0.94, 0.2, 0.2), upper=(1.0, 0.75, 0.97)),
        ),
    ),
    MaskDefinition(
        name="fabric",
        space="lab",
        ranges=(
            MaskRange(lower=(20.0, -40.0, -40.0), upper=(95.0, 60.0, 60.0)),
        ),
    ),
    MaskDefinition(
        name="background",
        space="hsv",
        ranges=(
            MaskRange(lower=(0.0, 0.0, 0.0), upper=(1.0, 0.35, 0.9)),
        ),
    ),
)


def specular_index(
    img: Image.Image,
    masks: Optional[Sequence[MaskDefinition]] = None,
    highlight_quantile: float = 0.97,
) -> SpecularMetrics:
    """
    Индекс «влажности/бликов» (0..1) и зональные показатели:
      - Берём яркостный канал V из HSV.
      - Порог по верхнему квантили (по умолчанию p=0.97) → маска бликов.
      - Доля маски + локальная резкость по Лапласу (на бликах).
      - Для указанных масок (кожа, ткань, фон) считаем показатели и агрегаты.
    Эмпирические нормировки подобраны для постеров ~1–4K.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = color.rgb2hsv(arr)
    lab = color.rgb2lab(arr)
    spaces = {"rgb": arr, "hsv": hsv, "lab": lab}

    V = hsv[..., 2]
    thr = np.quantile(V, highlight_quantile)
    highlight_mask = (V >= thr).astype(np.float32)

    ratio = float(highlight_mask.mean())  # обычно ~0.0..0.05

    lap = filters.laplace(V, ksize=3)
    lap_abs = np.abs(lap)
    sharp = float((lap_abs * highlight_mask).mean())

    ratio_n = min(ratio / 0.04, 1.0)   # saturate на 4% площади бликов
    sharp_n = min(sharp * 12.0, 1.0)   # эмпирика
    score = 0.6 * ratio_n + 0.4 * sharp_n

    zones: Dict[str, SpecularZoneMetrics] = {}
    zone_stats: List[Tuple[float, float]] = []
    masks_to_use = masks or DEFAULT_SPECULAR_MASKS

    total_pixels = float(arr.shape[0] * arr.shape[1])

    for mask_def in masks_to_use:
        try:
            zone_mask = mask_def.build_mask(spaces)
        except ValueError:
            continue

        coverage = float(zone_mask.mean())
        if coverage <= max(mask_def.min_coverage, 1e-5):
            zones[mask_def.name] = SpecularZoneMetrics(coverage=coverage)
            continue

        zone_pixels = float(zone_mask.sum())
        zone_highlight = highlight_mask * zone_mask
        highlight_pixels = float(zone_highlight.sum())

        if zone_pixels <= 1.0 or highlight_pixels <= 0.0:
            zones[mask_def.name] = SpecularZoneMetrics(
                coverage=coverage,
                highlight_ratio=0.0,
                highlight_density=0.0,
                sharpness=0.0,
                score=0.0,
            )
            zone_stats.append((coverage, 0.0))
            continue

        highlight_ratio_zone = highlight_pixels / zone_pixels
        highlight_density = highlight_pixels / total_pixels
        zone_sharp = float((lap_abs * zone_highlight).sum() / zone_pixels)

        zone_ratio_n = min(highlight_ratio_zone / 0.04, 1.0)
        zone_sharp_n = min(zone_sharp * 12.0, 1.0)
        zone_score = float(max(0.0, min(1.0, 0.6 * zone_ratio_n + 0.4 * zone_sharp_n)))

        zone_metric = SpecularZoneMetrics(
            coverage=coverage,
            highlight_ratio=float(highlight_ratio_zone),
            highlight_density=float(highlight_density),
            sharpness=float(zone_sharp),
            score=zone_score,
        )
        zones[mask_def.name] = zone_metric
        zone_stats.append((coverage, zone_score))

    if zone_stats:
        covered = sum(c for c, _ in zone_stats if c > 0)
        weighted = (
            sum(c * s for c, s in zone_stats if c > 0) / covered
            if covered > 0
            else 0.0
        )
        mean = sum(s for _, s in zone_stats) / len(zone_stats)
        aggregate = SpecularAggregate(
            weighted_score=float(weighted),
            mean_score=float(mean),
            coverage=float(covered),
        )
    else:
        aggregate = SpecularAggregate()

    return SpecularMetrics(
        score=float(max(0.0, min(1.0, score))),
        highlight_ratio=float(ratio),
        sharpness=float(sharp),
        zones=zones,
        aggregate=aggregate,
    )


# =========================
# CLIP backend & style prob
# =========================

def _encode_texts(model, tok, device, texts):
    toks = tok(texts).to(device)
    with torch.no_grad():
        vecs = model.encode_text(toks)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs


@dataclass
class ClipHeadAnchors:
    key: str
    display_name: Optional[str]
    primary: Optional[str]
    calibration: Dict[str, Tuple[float, float]]
    embeddings: Dict[str, torch.Tensor]
    labels: List[str]


@dataclass
class ClipBackend:
    device: str
    model: Any
    preprocess: Any
    tok: Any
    heads: Dict[str, ClipHeadAnchors]


def _load_clip(device: str = "auto") -> ClipBackend:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = _CLIP_ARCH
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pre, device=device
    )
    tok = open_clip.get_tokenizer(name)
    model.eval()

    heads: Dict[str, ClipHeadAnchors] = {}
    for head_cfg in _CLIP_TEXT_HEADS.heads:
        embeddings = {
            group.label: _encode_texts(model, tok, device, group.prompts)
            for group in head_cfg.groups
        }
        labels = [group.label for group in head_cfg.groups]
        heads[head_cfg.key] = ClipHeadAnchors(
            key=head_cfg.key,
            display_name=head_cfg.display_name,
            primary=head_cfg.primary,
            calibration=dict(head_cfg.calibration),
            embeddings=embeddings,
            labels=labels,
        )

    return ClipBackend(
        device=device,
        model=model,
        preprocess=preprocess,
        tok=tok,
        heads=heads,
    )


@torch.inference_mode()
def _image_emb(clip: ClipBackend, img: Image.Image) -> torch.Tensor:
    x = clip.preprocess(img.convert("RGB")).unsqueeze(0).to(clip.device)
    with torch.amp.autocast('cuda', enabled=(clip.device == "cuda"), cache_enabled=True):
        im = clip.model.encode_image(x)
        im = im / im.norm(dim=-1, keepdim=True)
    return im  # (1, d)


def _calibrate(p: float, cal: Optional[Tuple[float, float]]) -> float:
    """
    Линейное «растяжение» вероятности по перцентилям датасета.
    cal=(p20, p80) → [0..1] так, чтобы p20→0, p80→1, остальное — clamp.
    """
    if not cal:
        return p
    lo, hi = cal
    if hi <= lo:
        return p
    return float(max(0.0, min(1.0, (p - lo) / (hi - lo))))


CalibrationOverrides = Dict[Tuple[str, str], Tuple[float, float]]


@torch.inference_mode()
def clip_head_probabilities(
    clip: ClipBackend,
    img: Image.Image,
    tau: float = DEFAULT_TAU,
    calibration_overrides: Optional[CalibrationOverrides] = None,
    embedding: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Рассчитывает вероятности для всех текстовых голов, определённых в конфиге.
    Возвращает словарь {head_key: {label: prob}}.
    """
    im = embedding if embedding is not None else _image_emb(clip, img)  # (1, d)
    overrides = calibration_overrides or {}

    results: Dict[str, Dict[str, float]] = {}
    with torch.amp.autocast('cuda', enabled=(clip.device == "cuda"), cache_enabled=True):
        for key, head in clip.heads.items():
            logits: List[torch.Tensor] = []
            for label in head.labels:
                sims = im @ head.embeddings[label].T  # (1, K)
                logits.append(sims.max(dim=1).values)  # (1,)
            if not logits:
                continue
            logits_tensor = torch.stack(logits).squeeze(-1) / max(tau, 1e-6)
            probs = torch.softmax(logits_tensor, dim=0)
            head_probs: Dict[str, float] = {}
            for idx, label in enumerate(head.labels):
                prob = float(probs[idx].item())
                cal = overrides.get((key, label))
                if cal is None:
                    cal = head.calibration.get(label)
                head_probs[label] = _calibrate(prob, cal)
            results[key] = head_probs
    return results


# =========================
# NSFW via OpenNSFW2
# =========================

def nsfw_score(path: Path) -> float:
    """Вероятность NSFW от 0..1 по opennsfw2."""
    try:
        p = float(n2.predict_image(str(path)))
    except Exception:
        p = 0.0
    return p


# =========================
# Public API: DualScorer
# =========================

def _ts() -> int:
    return int(time.time())


class DualScorer:
    """
    Двухканальный скорер:
      - nsfw (0..100) по opennsfw2
      - style (0..100) = w_clip*clip_style + w_spec*specular + w_illu*illu_bias

    Сохраняет результаты в:
      - SQLite (scores.sqlite, таблица scores)
      - JSONL (scores.jsonl) — удобно для быстрых фильтров через jq/pandas
    """

    def __init__(
        self,
        device: str = "auto",
        batch: int = 4,
        db_path: Path | str = "scores.sqlite",
        jsonl_path: Path | str = "scores.jsonl",
        weights: Mapping[str, float] | None = None,
        tau: Optional[float] = None,
        cal_style: Optional[Tuple[float, float]] = None,
        cal_illu: Optional[Tuple[float, float]] = None,
        auto_weights: Optional[Dict[str, float]] | None = None,
        weight_table: Optional[WeightProfileTable] = None,
        weight_profile: str = "default",
        persist_profile_updates: bool = False,
        composition_enabled: bool = True,
    ):
        # device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch = max(1, int(batch))

        # backend
        self.clip = _load_clip(self.device)
        self.pose_analyzer = PoseAnalyzer(device=self.device)

        # storage
        self.db = Path(db_path)
        self.jsonl = Path(jsonl_path)
        _ensure_db(self.db)

        # weights & params
        self._persist_weights = bool(persist_profile_updates)
        self.style_mixer = StyleMixer(
            weights=weights,
            defaults=DEFAULT_STYLE_WEIGHTS,
            weight_table=weight_table,
            profile=weight_profile,
            persist_updates=self._persist_weights,
        )
        self.w = dict(self.style_mixer.weights)
        self.tau = float(tau) if tau is not None else DEFAULT_TAU
        self.cal_style = cal_style if cal_style is not None else DEFAULT_CAL_STYLE
        self.cal_illu = cal_illu if cal_illu is not None else DEFAULT_CAL_ILLU
        self._base_w = dict(self.w)
        self._auto = AutoWeightsSettings.from_dict(auto_weights)
        self._ema = {k: self._auto.initial_level for k in ("clip", "spec", "illu")}
        if self._auto.enabled:
            self._apply_weight_bounds()
        self.embedding_cache = None
        self._composition_enabled = bool(composition_enabled)
        self.composition_analyzer: Optional[CompositionAnalyzer]
        if self._composition_enabled:
            self.composition_analyzer = CompositionAnalyzer(device=self.device)
            if not self.composition_analyzer.available:
                self.composition_analyzer = None
        else:
            self.composition_analyzer = None

    def _apply_weight_bounds(self) -> None:
        if not self._auto.enabled:
            return
        for key in ("clip", "spec", "illu"):
            self.w[key] = max(self._auto.min_weight, min(self._auto.max_weight, self.w[key]))
        total = sum(self.w.values())
        if total > 0:
            for key in self.w:
                self.w[key] /= total
        self._sync_style_weights(persist=self._persist_weights)

    def _sync_style_weights(self, *, persist: bool = False) -> None:
        self.w = self.style_mixer.set_weights(self.w, persist=persist)

    def _update_auto_weights(self, clip_style: float, specular: float, illu_bias: float) -> None:
        if not self._auto.enabled:
            return
        alpha = max(0.0, min(1.0, self._auto.ema_alpha))
        values = {"clip": clip_style, "spec": specular, "illu": illu_bias}
        for key, value in values.items():
            self._ema[key] = (1 - alpha) * self._ema[key] + alpha * value

        raw: Dict[str, float] = {}
        for key in ("clip", "spec", "illu"):
            ema = max(self._ema[key], self._auto.min_component)
            gain = self._auto.target / ema if ema > 0 else self._auto.max_gain
            gain = max(self._auto.min_gain, min(self._auto.max_gain, gain))
            raw[key] = self._base_w[key] * gain

        total_raw = sum(raw.values())
        if total_raw <= 0:
            return
        normalized = {k: raw[k] / total_raw for k in raw}

        mu = max(0.0, min(1.0, self._auto.momentum))
        for key in normalized:
            self.w[key] = (1 - mu) * self.w[key] + mu * normalized[key]

        self._apply_weight_bounds()

    # ---- composition ----
    def style_from_components(self, clip_style: float, specular: float, illu_bias: float) -> float:
        composition = self.style_mixer.compose(clip_style, specular, illu_bias)
        return composition.total

    def compose_style(
        self, clip_style: float, specular: float, illu_bias: float
    ) -> StyleComposition:
        return self.style_mixer.compose(clip_style, specular, illu_bias)

    def current_weights(self) -> Dict[str, float]:
        return dict(self.w)

    # ---- single image ----
    def score_one(self, path: Path) -> ScoreResult:
        """
        Считает метрики для одного изображения и возвращает развёрнутый результат.
        """
        img = Image.open(path).convert("RGB")

        calibration_overrides: CalibrationOverrides = {}
        style_head = self.clip.heads.get("style")
        if style_head and self.cal_style:
            positive = style_head.primary or (style_head.labels[0] if style_head.labels else None)
            if positive:
                calibration_overrides[(style_head.key, positive)] = self.cal_style

        illu_head = self.clip.heads.get("illustration")
        if illu_head and self.cal_illu:
            positive = illu_head.primary or (illu_head.labels[0] if illu_head.labels else None)
            if positive:
                calibration_overrides[(illu_head.key, positive)] = self.cal_illu

        image_embedding = _image_emb(self.clip, img)

        head_probs = clip_head_probabilities(
            self.clip,
            img,
            tau=self.tau,
            calibration_overrides=calibration_overrides,
            embedding=image_embedding,
        )

        clip_style = 0.0
        if style_head:
            positive = style_head.primary or (style_head.labels[0] if style_head.labels else None)
            if positive:
                clip_style = head_probs.get(style_head.key, {}).get(positive, 0.0)

        illu_bias = 0.0
        if illu_head:
            positive = illu_head.primary or (illu_head.labels[0] if illu_head.labels else None)
            if positive:
                illu_bias = head_probs.get(illu_head.key, {}).get(positive, 0.0)

        spec_metrics = specular_index(img)
        spec = spec_metrics.score
        nsfw = nsfw_score(path)
        pose_metrics = self.pose_analyzer.analyze(img)
        composition_metrics = CompositionMetrics.empty()
        if self.composition_analyzer is not None:
            composition_metrics = self.composition_analyzer.analyze(img)
        composition = self.style_mixer.compose(clip_style, spec, illu_bias)
        style = composition.total
        self._update_auto_weights(clip_style, spec, illu_bias)

        clip_head_payload: Dict[str, Dict[str, Any]] = {}
        for key, head in self.clip.heads.items():
            clip_head_payload[key] = {
                "display_name": head.display_name or key,
                "primary": head.primary,
                "probabilities": dict(head_probs.get(key, {})),
            }

        embedding_vec = image_embedding.detach().float().cpu().numpy().reshape(-1)

        return ScoreResult(
            nsfw=nsfw,
            style=style,
            clip_style=clip_style,
            specular=spec,
            illu_bias=illu_bias,
            clip_heads=clip_head_payload,
            specular_metrics=spec_metrics,
            embedding=embedding_vec,
            style_contributions=composition.contributions,
            style_weights=composition.weights,
            pose_metrics=pose_metrics,
            composition_metrics=composition_metrics,
        )

    # ---- batch & save ----
    def score_and_save(self, paths: List[Path], notes: str = "") -> List[ScoredImage]:
        """
        Считает метрики для списка путей, пишет в SQLite и JSONL,
        возвращает список структур ScoredImage.
        """
        if not paths:
            return []

        rows_db: List[Tuple] = []
        out_images: List[ScoredImage] = []
        t = _ts()

        # простая последовательная обработка (GPU/CPU-агностично)
        with self.jsonl.open("a", encoding="utf-8") as jf:
            for p in paths:
                try:
                    result = self.score_one(p)
                except Exception:
                    # если картинка битая/не читается
                    result = ScoreResult(
                        nsfw=0.0,
                        style=0.0,
                        clip_style=0.0,
                        specular=0.0,
                        illu_bias=0.0,
                        clip_heads={},
                        specular_metrics=SpecularMetrics.empty(),
                        embedding=None,
                        style_contributions={},
                        style_weights={},
                        pose_metrics=PoseAnalysis.empty(),
                        composition_metrics=CompositionMetrics.empty(),
                    )

                nsfw100 = int(round(result.nsfw * 100))
                style100 = int(round(result.style * 100))
                cs100 = int(round(result.clip_style * 100))
                sp100 = int(round(result.specular * 100))
                ib100 = int(round(result.illu_bias * 100))

                spec_payload = result.specular_metrics.to_payload()
                zone_scores = {
                    name: int(round(zone.get("score", 0.0) * 100))
                    for name, zone in spec_payload.get("zones", {}).items()
                }
                rec = {
                    "path": str(p),
                    "ts": t,
                    "schema_version": SCORES_SCHEMA_VERSION,
                    "nsfw": nsfw100,
                    "style": style100,
                    "clip_style": cs100,
                    "specular": sp100,
                    "illu_bias": ib100,
                    "clip_heads": result.clip_heads,
                    "specular_details": spec_payload,
                    "specular_zones": zone_scores,
                    "specular_zonal_weighted": int(
                        round(spec_payload.get("aggregate", {}).get("weighted_score", 0.0) * 100)
                    ),
                    "style_weights": result.style_weights,
                    "style_contributions": result.style_contributions,
                    "pose_metrics": result.pose_metrics.to_payload(),
                    "composition": result.composition_metrics.to_payload(),
                    "notes": notes,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows_db.append(
                    (
                        str(p),
                        t,
                        nsfw100,
                        style100,
                        cs100,
                        sp100,
                        ib100,
                        notes,
                        SCORES_SCHEMA_VERSION,
                    )
                )
                embedding = None
                if result.embedding is not None:
                    embedding = np.asarray(result.embedding, dtype=np.float32).copy()

                out_images.append(
                    ScoredImage(
                        path=p,
                        nsfw=nsfw100,
                        style=style100,
                        clip_style=cs100,
                        specular=sp100,
                        illu_bias=ib100,
                        embedding=embedding,
                        style_raw=float(result.style),
                        clip_style_raw=float(result.clip_style),
                        specular_raw=float(result.specular),
                        illu_bias_raw=float(result.illu_bias),
                        style_contributions=dict(result.style_contributions),
                        style_weights=dict(result.style_weights),
                        pose_class=result.pose_metrics.pose_class,
                        pose_confidence=int(round(result.pose_metrics.pose_confidence * 100)),
                        body_curve=int(round(result.pose_metrics.body_curve * 100)),
                        gaze_direction=result.pose_metrics.gaze_direction,
                        gaze_confidence=int(round(result.pose_metrics.gaze_confidence * 100)),
                        coverage=int(round(result.pose_metrics.coverage * 100)),
                        skin_ratio=int(round(result.pose_metrics.skin_ratio * 100)),
                        cropping_tightness=int(round(result.composition_metrics.cropping_tightness * 100)),
                        thirds_alignment=int(round(result.composition_metrics.thirds_alignment * 100)),
                        negative_space=int(round(result.composition_metrics.negative_space * 100)),
                        composition_raw=result.composition_metrics.to_payload(),
                    )
                )

        if rows_db:
            _insert_db(self.db, rows_db)

        # немного освобождаем память GPU между батчами
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return out_images
