# scorer.py
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import opennsfw2 as n2
import torch
from PIL import Image
from skimage import color, filters
import open_clip


# =========================
# Config & Text Anchors
# =========================

# Позитивные описания целевого стиля
STYLE_POS = [
    "2D watercolor pin-up illustration, glossy highlights, wet sheen, soft paper texture, pastel palette",
    "glossy watercolor poster art, dewy reflections, rim light, smooth gradients, paper grain",
    "stylized pin-up drawing with wet shine and jellylike speculars, elegant composition",
    "soft pastel watercolor character art, luminous highlights, dewy wet gloss on surfaces",
    "retro pin-up poster, watercolor wash, glossy finish, paper texture visible",
    "anime-style watercolor illustration with wet speculars and pastel tones"
]

# Антикласс (от чего отталкиваемся)
STYLE_NEG = [
    "photorealistic photo of a person, camera shot, skin pores, natural lighting",
    "3D render, physically based rendering, CGI",
    "oil painting, thick impasto, visible brush strokes",
    "hard-ink comic with flat cel shading and heavy outlines",
    "noisy scan, low quality snapshot, candid photo",
    "sketch pencil drawing, rough lines",
    "vector flat infographic, icon style"
]

# Иллюстрация vs Фото (для иллюстративного сдвига)
ILLU_TEXTS = [
    "clean 2D illustration, stylized drawing, graphic lines",
    "digital painting, character illustration",
    "anime illustration, toon shading"
]
PHOTO_TEXTS = [
    "realistic photograph of a person",
    "studio photo, DSLR, lens bokeh",
    "street candid photo, natural skin texture"
]

# Модель CLIP по умолчанию
_CLIP_ARCH = ("ViT-B-32", "openai")  # компактная и дружелюбная к VRAM

# Температура softmax для повышения контраста классов (ниже — контрастнее)
TAU: float = 0.07

# Калибровка распределений (опционально). Можно проставить (p20, p80) после быстрых измерений.
CAL_STYLE: Optional[Tuple[float, float]] = None  # например (0.35, 0.75)
CAL_ILLU: Optional[Tuple[float, float]] = None  # например (0.45, 0.85)

# Веса итоговой style-метрики (композиция из трёх компонент)
W_CLIP, W_SPEC, W_ILLU = 0.55, 0.35, 0.10


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
class ClipBackend:
    device: str
    model: any
    preprocess: any
    tok: any
    style_pos: torch.Tensor
    style_neg: torch.Tensor
    illu_pos: torch.Tensor
    illu_neg: torch.Tensor


def _load_clip(device: str = "auto") -> ClipBackend:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    name, pre = _CLIP_ARCH
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=pre, device=device
    )
    tok = open_clip.get_tokenizer(name)
    model.eval()

    return ClipBackend(
        device=device,
        model=model,
        preprocess=preprocess,
        tok=tok,
        style_pos=_encode_texts(model, tok, device, STYLE_POS),
        style_neg=_encode_texts(model, tok, device, STYLE_NEG),
        illu_pos=_encode_texts(model, tok, device, ILLU_TEXTS),
        illu_neg=_encode_texts(model, tok, device, PHOTO_TEXTS),
    )


@torch.inference_mode()
def _image_emb(clip: ClipBackend, img: Image.Image) -> torch.Tensor:
    x = clip.preprocess(img.convert("RGB")).unsqueeze(0).to(clip.device)
    with torch.amp.autocast('cuda', enabled=True, cache_enabled=True):
        im = clip.model.encode_image(x)
        im = im / im.norm(dim=-1, keepdim=True)
    return im  # (1, d)


def _softmax_prob(sim_pos: torch.Tensor, sim_neg: torch.Tensor, tau: float) -> float:
    """
    Вероятность класса 'pos' против 'neg' через softmax c температурой.
    sim_*: (1, K) — косинусные сходства с каждым анкором в классе.
    Берём max по классу, чтобы «лучший матч» доминировал.
    """
    # max по каждому классу
    p = sim_pos.max(dim=1).values  # (1,)
    n = sim_neg.max(dim=1).values  # (1,)
    logits = torch.cat([p, n], dim=0) / max(tau, 1e-6)
    probs = torch.softmax(logits, dim=0)
    return float(probs[0].item())  # P(pos)


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


@torch.inference_mode()
def clip_style_and_illu(clip: ClipBackend, img: Image.Image) -> Tuple[float, float]:
    """
    Возвращает:
      clip_style ∈ [0..1] — вероятность «watercolor glossy pin-up» против анти-анкоров,
      illu_bias  ∈ [0..1] — вероятность «illustration» против «photo».
    """
    im = _image_emb(clip, img)                 # (1, d)
    with torch.amp.autocast('cuda', enabled=True, cache_enabled=True):
        sp = (im @ clip.style_pos.T)           # (1, Ks)
        sn = (im @ clip.style_neg.T)           # (1, Kn)
        cs = _softmax_prob(sp, sn, TAU)
        cs = _calibrate(cs, CAL_STYLE)

        ip = (im @ clip.illu_pos.T)            # (1, Ki)
        ineg = (im @ clip.illu_neg.T)          # (1, Kj)
        ib = _softmax_prob(ip, ineg, TAU)
        ib = _calibrate(ib, CAL_ILLU)

    return cs, ib


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
        global TAU, CAL_STYLE, CAL_ILLU
        self.w = weights or {"clip": W_CLIP, "spec": W_SPEC, "illu": W_ILLU}
        self.tau = float(tau) if tau is not None else TAU
        # локальная переустановка глобалок (если заданы)
        TAU = self.tau
        CAL_STYLE = cal_style if cal_style is not None else CAL_STYLE
        CAL_ILLU = cal_illu if cal_illu is not None else CAL_ILLU

    # ---- composition ----
    def style_from_components(self, clip_style: float, specular: float, illu_bias: float) -> float:
        v = self.w["clip"] * clip_style + self.w["spec"] * specular + self.w["illu"] * illu_bias
        return float(max(0.0, min(1.0, v)))

    # ---- single image ----
    def score_one(self, path: Path) -> Tuple[float, float, float, float, float]:
        """
        Возвращает tuple:
          nsfw, style, clip_style, specular, illu_bias   (все — 0..1)
        """
        img = Image.open(path).convert("RGB")
        clip_style, illu_bias = clip_style_and_illu(self.clip, img)
        spec = specular_index(img)
        nsfw = nsfw_score(path)
        style = self.style_from_components(clip_style, spec, illu_bias)
        return nsfw, style, clip_style, spec, illu_bias

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
                    nsfw, style, clip_s, spec, illu = self.score_one(p)
                except Exception:
                    # если картинка битая/не читается
                    nsfw = style = clip_s = spec = illu = 0.0

                nsfw100 = int(round(nsfw * 100))
                style100 = int(round(style * 100))
                cs100 = int(round(clip_s * 100))
                sp100 = int(round(spec * 100))
                ib100 = int(round(illu * 100))

                rec = {
                    "path": str(p),
                    "ts": t,
                    "nsfw": nsfw100,
                    "style": style100,
                    "clip_style": cs100,
                    "specular": sp100,
                    "illu_bias": ib100,
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
