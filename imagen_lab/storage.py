from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .scene_builder import SceneStruct, short_readable
from .embeddings import EmbeddingCache
from .metrics import history_distance, intra_batch_distance

try:  # pragma: no cover - typing support
    from scorer import ScoredImage
except Exception:  # pragma: no cover - runtime import fallback
    ScoredImage = None  # type: ignore


@dataclass
class ScoredBatch:
    images: List["ScoredImage"]
    metrics: Dict[str, Any]

    def best(self, w_style: float, w_nsfw: float) -> Optional["ScoredImage"]:
        if not self.images:
            return None
        return max(self.images, key=lambda item: item.fitness(w_style, w_nsfw))

    def is_empty(self) -> bool:
        return not self.images

try:  # pragma: no cover - optional dependency
    import piexif
except ImportError:  # pragma: no cover - optional dependency
    piexif = None


@dataclass
class PromptLogRow:
    path: str
    ts: int
    gen: int
    indiv: int
    prompt: str
    params: Dict[str, object]
    sfw: float
    temperature: float
    w_style: float
    w_nsfw: float
    fitness: float
    parents: Optional[Sequence[str]]
    op: str


def _xp_utf16le(text: str) -> bytes:
    return text.encode("utf-16le")


def write_exif_text(jpg: Path, text: str) -> None:
    if piexif is None:
        return
    exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    try:
        exif = piexif.load(str(jpg))
    except Exception:  # pragma: no cover - best-effort
        pass
    exif.setdefault("0th", {})[piexif.ImageIFD.ImageDescription] = text.encode("utf-8", "ignore")
    exif["0th"][piexif.ImageIFD.XPComment] = _xp_utf16le(text)
    piexif.insert(piexif.dump(exif), str(jpg))


class PromptLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs(
                session_id TEXT PRIMARY KEY,
                started_ts INTEGER,
                mode TEXT,
                cfg_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts(
                path TEXT PRIMARY KEY,
                ts INTEGER,
                gen INTEGER,
                indiv INTEGER,
                prompt TEXT,
                params TEXT,
                sfw REAL,
                temperature REAL,
                weights_style REAL,
                weights_nsfw REAL,
                fitness REAL,
                parents TEXT,
                op TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def log_run(self, session_id: str, mode: str, cfg: Dict[str, object]) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs(session_id, started_ts, mode, cfg_json) VALUES(?,?,?,?)",
            (session_id, int(time.time()), mode, json.dumps(cfg, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()

    def log_prompt(self, row: PromptLogRow) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO prompts(
                path, ts, gen, indiv, prompt, params,
                sfw, temperature, weights_style, weights_nsfw, fitness, parents, op
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row.path,
                row.ts,
                row.gen,
                row.indiv,
                row.prompt,
                json.dumps(row.params, ensure_ascii=False),
                row.sfw,
                row.temperature,
                row.w_style,
                row.w_nsfw,
                row.fitness,
                json.dumps(list(row.parents) if row.parents else []),
                row.op,
            ),
        )
        conn.commit()
        conn.close()


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_variants(
        self,
        meta_base: Dict[str, object],
        scene: SceneStruct,
        final_prompt: str,
        images: Iterable[Tuple[int, bytes]],
    ) -> List[Path]:
        saved: List[Path] = []
        for variant_index, image_bytes in images:
            path = self.output_dir / f"{meta_base['id']}_{variant_index}.jpg"
            json_sidecar = path.with_suffix(".json")
            txt_sidecar = path.with_suffix(".txt")

            with open(path, "wb") as handle:
                handle.write(image_bytes)
            exif_line = f"{short_readable(scene)} | prompt={final_prompt}"
            write_exif_text(path, exif_line)

            print(f"[imagen-lab] saved image {path.name} -> {path.parent}")

            meta = dict(meta_base)
            meta["id"] = f"{meta_base['id']}_{variant_index}"
            meta["final_prompt"] = final_prompt
            meta["parameters"] = scene.to_dict()

            json_sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            txt_sidecar.write_text(final_prompt + "\n", encoding="utf-8")

            saved.append(path)
        return saved


def extract_image_bytes(response) -> Iterable[Tuple[int, bytes]]:
    for idx, generated in enumerate(getattr(response, "generated_images", []), start=1):
        img = getattr(generated, "image", None)
        if img is None:
            continue
        blob = getattr(img, "image_bytes", None) or getattr(img, "imageBytes", None)
        if not blob:
            continue
        yield idx, blob


def save_and_score(
    response,
    writer: ArtifactWriter,
    logger: PromptLogger,
    scorer,
    meta_base: Dict[str, object],
    final_prompt: str,
    scene: SceneStruct,
    session_id: str,
    gen: Optional[int],
    indiv: Optional[int],
    w_style: float,
    w_nsfw: float,
) -> ScoredBatch:
    if not response or not response.generated_images:
        return ScoredBatch(images=[], metrics={})
    images = list(extract_image_bytes(response))
    if not images:
        return ScoredBatch(images=[], metrics={})
    saved_paths = writer.write_variants(meta_base, scene, final_prompt, images)
    scored_images: List[ScoredImage] = scorer.score_and_save(saved_paths, notes="imagen jelly pin-up")

    valid_indices: List[int] = []
    valid_embeddings: List[np.ndarray] = []
    for idx, scored in enumerate(scored_images):
        if getattr(scored, "embedding", None) is not None:
            valid_indices.append(idx)
            valid_embeddings.append(np.asarray(scored.embedding, dtype=np.float32))

    batch_metrics = intra_batch_distance(valid_embeddings)
    history_metrics = history_distance(valid_embeddings, getattr(scorer, "embedding_cache", None))

    per_image_batch = [None] * len(scored_images)
    if batch_metrics.get("available"):
        per_vals = batch_metrics.get("per_item_min", [])
        for offset, idx in enumerate(valid_indices):
            if offset < len(per_vals):
                per_image_batch[idx] = float(per_vals[offset])

    per_image_history = [None] * len(scored_images)
    if history_metrics.get("available"):
        per_vals = history_metrics.get("per_item_min", [])
        for offset, idx in enumerate(valid_indices):
            if offset < len(per_vals):
                per_image_history[idx] = float(per_vals[offset])

    cache = getattr(scorer, "embedding_cache", None)
    if isinstance(cache, EmbeddingCache):
        cache.extend(valid_embeddings)

    metrics: Dict[str, Any] = {
        "batch": batch_metrics,
        "history": {**history_metrics, "size": len(cache) if isinstance(cache, EmbeddingCache) else 0},
    }

    if scored_images:
        denom = float(len(scored_images))
        mean_total = sum(float(img.style_raw) for img in scored_images) / denom
        mean_components = {
            "clip": sum(float(img.clip_style_raw) for img in scored_images) / denom,
            "spec": sum(float(img.specular_raw) for img in scored_images) / denom,
            "illu": sum(float(img.illu_bias_raw) for img in scored_images) / denom,
        }
        contrib_totals: Dict[str, float] = {}
        for img in scored_images:
            for key, value in img.style_contributions.items():
                contrib_totals[key] = contrib_totals.get(key, 0.0) + float(value)
        mean_contributions = {k: v / denom for k, v in contrib_totals.items()}
        style_weights = scored_images[0].style_weights if scored_images[0].style_weights else {}
        if not style_weights and hasattr(scorer, "current_weights"):
            try:
                style_weights = dict(scorer.current_weights())  # type: ignore[call-arg]
            except Exception:
                style_weights = {}
        comp_totals = {
            "cropping_tightness": 0.0,
            "thirds_alignment": 0.0,
            "negative_space": 0.0,
        }
        comp_counts = 0
        for img in scored_images:
            comp = getattr(img, "composition_raw", {}) or {}
            if comp and (comp.get("num_detections", 0) or comp.get("primary_bbox")):
                comp_counts += 1
                for key in comp_totals:
                    comp_totals[key] += float(comp.get(key, 0.0))
        comp_means = {k: (comp_totals[k] / comp_counts) for k in comp_totals} if comp_counts else {}
        metrics["style"] = {
            "mean_total": mean_total,
            "mean_components": mean_components,
            "mean_contributions": mean_contributions,
            "weights": style_weights,
        }
        if comp_means:
            metrics["composition"] = {
                "mean_cropping_tightness": comp_means["cropping_tightness"],
                "mean_thirds_alignment": comp_means["thirds_alignment"],
                "mean_negative_space": comp_means["negative_space"],
            }

    for idx, scored in enumerate(scored_images):
        image_path = scored.path
        nsfw100 = scored.nsfw
        style100 = scored.style
        sidecar = image_path.with_suffix(".json")
        try:
            side_meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            side_meta = {}
        fitness = round(w_style * style100 + w_nsfw * nsfw100, 2)
        side_meta.update({
            "nsfw_score": nsfw100,
            "style_score": style100,
            "fitness": fitness,
            "embedding_metrics": {
                "batch": {
                    "pairwise_min": batch_metrics.get("pairwise_min"),
                    "pairwise_mean": batch_metrics.get("pairwise_mean"),
                    "self_min_distance": per_image_batch[idx],
                },
                "history": {
                    "min_distance": history_metrics.get("min_distance"),
                    "mean_distance": history_metrics.get("mean_distance"),
                    "self_min_distance": per_image_history[idx],
                    "history_size": len(cache) if isinstance(cache, EmbeddingCache) else 0,
                },
            },
            "style_breakdown": {
                "aggregate": float(scored.style_raw),
                "components": {
                    "clip": float(scored.clip_style_raw),
                    "spec": float(scored.specular_raw),
                    "illu": float(scored.illu_bias_raw),
                },
                "contributions": {k: float(v) for k, v in scored.style_contributions.items()},
                "weights": scored.style_weights,
            },
            "composition_metrics": scored.composition_raw,
        })
        sidecar.write_text(json.dumps(side_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.log_prompt(
            PromptLogRow(
                path=str(image_path),
                ts=int(time.time()),
                gen=int(gen) if gen is not None else -1,
                indiv=int(indiv) if indiv is not None else -1,
                prompt=final_prompt,
                params={
                    "session": session_id,
                    "meta": meta_base,
                    "struct": scene.to_dict(),
                    "embedding_metrics": metrics,
                },
                sfw=float(meta_base.get("ollama", {}).get("sfw_level", 0.0)),
                temperature=float(meta_base.get("ollama", {}).get("temperature", 0.0)),
                w_style=float(w_style),
                w_nsfw=float(w_nsfw),
                fitness=fitness,
                parents=meta_base.get("ga_parents"),
                op=str(meta_base.get("ga_op", "plain")),
            )
        )
    scored_images.sort(key=lambda item: item.fitness(w_style, w_nsfw), reverse=True)
    return ScoredBatch(images=scored_images, metrics=metrics)
