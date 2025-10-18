# scorer.py
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import opennsfw2 as n2
import torch
from PIL import Image
from skimage import color, filters
import open_clip

from imagen_lab.scoring import ClipTextHeadsConfig, load_clip_text_heads

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
W_CLIP, W_SPEC, W_ILLU = 0.55, 0.35, 0.10


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
      notes TEXT
    )
    """)
    conn.commit()
    conn.close()


def _insert_db(db_path: Path, rows: List[Tuple]):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO scores(path, ts, nsfw, style, clip_style, specular, illu_bias, notes)
    VALUES(?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()


# =========================
# Specular / Wetness index
# =========================

def specular_index(img: Image.Image) -> float:
    """
    Индекс «влажности/бликов» (0..1):
      - Берём яркостный канал V из HSV.
      - Порог по верхнему квантили (p=0.97) → маска бликов.
      - Доля маски + локальная резкость по Лапласу (на бликах).
    Эмпирические нормировки подобраны для постеров ~1–4K.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = color.rgb2hsv(arr)
    V = hsv[..., 2]
    thr = np.quantile(V, 0.97)
    mask = (V >= thr).astype(np.float32)

    # Доля пикселей бликов
    ratio = float(mask.mean())  # обычно ~0.0..0.05

    # Резкость бликов
    lap = filters.laplace(V, ksize=3)
    sharp = float((np.abs(lap) * mask).mean())

    # Нормализация
    ratio_n = min(ratio / 0.04, 1.0)   # saturate на 4% площади бликов
    sharp_n = min(sharp * 12.0, 1.0)   # эмпирика
    score = 0.6 * ratio_n + 0.4 * sharp_n
    return float(max(0.0, min(1.0, score)))


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
) -> Dict[str, Dict[str, float]]:
    """
    Рассчитывает вероятности для всех текстовых голов, определённых в конфиге.
    Возвращает словарь {head_key: {label: prob}}.
    """
    im = _image_emb(clip, img)  # (1, d)
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
        weights: Dict[str, float] | None = None,
        tau: Optional[float] = None,
        cal_style: Optional[Tuple[float, float]] = None,
        cal_illu: Optional[Tuple[float, float]] = None,
        auto_weights: Optional[Dict[str, float]] | None = None,
    ):
        # device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch = max(1, int(batch))

        # backend
        self.clip = _load_clip(self.device)

        # storage
        self.db = Path(db_path)
        self.jsonl = Path(jsonl_path)
        _ensure_db(self.db)

        # weights & params
        self.w = self._normalize_weights(weights or {"clip": W_CLIP, "spec": W_SPEC, "illu": W_ILLU})
        self.tau = float(tau) if tau is not None else DEFAULT_TAU
        self.cal_style = cal_style if cal_style is not None else DEFAULT_CAL_STYLE
        self.cal_illu = cal_illu if cal_illu is not None else DEFAULT_CAL_ILLU
        self._base_w = dict(self.w)
        self._auto = AutoWeightsSettings.from_dict(auto_weights)
        self._ema = {k: self._auto.initial_level for k in ("clip", "spec", "illu")}
        if self._auto.enabled:
            self._apply_weight_bounds()

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            return {"clip": W_CLIP, "spec": W_SPEC, "illu": W_ILLU}
        normed = {k: max(v, 0.0) / total for k, v in weights.items()}
        return {k: normed.get(k, 0.0) for k in ("clip", "spec", "illu")}

    def _apply_weight_bounds(self) -> None:
        if not self._auto.enabled:
            return
        for key in ("clip", "spec", "illu"):
            self.w[key] = max(self._auto.min_weight, min(self._auto.max_weight, self.w[key]))
        total = sum(self.w.values())
        if total > 0:
            for key in self.w:
                self.w[key] /= total

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
        v = self.w["clip"] * clip_style + self.w["spec"] * specular + self.w["illu"] * illu_bias
        return float(max(0.0, min(1.0, v)))

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

        head_probs = clip_head_probabilities(
            self.clip,
            img,
            tau=self.tau,
            calibration_overrides=calibration_overrides,
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

        spec = specular_index(img)
        nsfw = nsfw_score(path)
        style = self.style_from_components(clip_style, spec, illu_bias)
        self._update_auto_weights(clip_style, spec, illu_bias)

        clip_head_payload: Dict[str, Dict[str, Any]] = {}
        for key, head in self.clip.heads.items():
            clip_head_payload[key] = {
                "display_name": head.display_name or key,
                "primary": head.primary,
                "probabilities": dict(head_probs.get(key, {})),
            }

        return ScoreResult(
            nsfw=nsfw,
            style=style,
            clip_style=clip_style,
            specular=spec,
            illu_bias=illu_bias,
            clip_heads=clip_head_payload,
        )

    # ---- batch & save ----
    def score_and_save(self, paths: List[Path], notes: str = "") -> List[Tuple[Path, int, int]]:
        """
        Считает метрики для списка путей, пишет в SQLite и JSONL,
        возвращает список (path, nsfw100, style100).
        """
        if not paths:
            return []

        rows_db: List[Tuple] = []
        out_triplets: List[Tuple[Path, int, int]] = []
        t = _ts()

        # простая последовательная обработка (GPU/CPU-агностично)
        with self.jsonl.open("a", encoding="utf-8") as jf:
            for p in paths:
                try:
                    result = self.score_one(p)
                except Exception:
                    # если картинка битая/не читается
                    result = ScoreResult(0.0, 0.0, 0.0, 0.0, 0.0, {})

                nsfw100 = int(round(result.nsfw * 100))
                style100 = int(round(result.style * 100))
                cs100 = int(round(result.clip_style * 100))
                sp100 = int(round(result.specular * 100))
                ib100 = int(round(result.illu_bias * 100))

                rec = {
                    "path": str(p),
                    "ts": t,
                    "nsfw": nsfw100,
                    "style": style100,
                    "clip_style": cs100,
                    "specular": sp100,
                    "illu_bias": ib100,
                    "clip_heads": result.clip_heads,
                    "notes": notes,
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows_db.append((str(p), t, nsfw100, style100, cs100, sp100, ib100, notes))
                out_triplets.append((p, nsfw100, style100))

        if rows_db:
            _insert_db(self.db, rows_db)

        # немного освобождаем память GPU между батчами
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return out_triplets
