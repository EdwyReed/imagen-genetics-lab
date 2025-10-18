from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .scene_builder import SceneStruct, short_readable

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
) -> List[Tuple[Path, int, int]]:
    images = list(extract_image_bytes(response))
    if not images:
        return []
    saved_paths = writer.write_variants(meta_base, scene, final_prompt, images)
    triplets = scorer.score_and_save(saved_paths, notes="imagen jelly pin-up")
    for image_path, nsfw100, style100 in triplets:
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
    triplets.sort(key=lambda x: w_style * x[2] + w_nsfw * x[1], reverse=True)
    return triplets
