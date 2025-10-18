from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .catalog import Catalog
from .randomization import WeightedSelector, maybe

GeneSet = Dict[str, Optional[str]]


def _gene_sequence(catalog: Catalog, key: str) -> List[dict]:
    camera = catalog.raw.get("camera", {})
    mapping = {
        "palette": catalog.section("palettes"),
        "lighting": catalog.section("lighting_presets"),
        "background": catalog.section("backgrounds"),
        "camera_angle": camera.get("angles", []),
        "camera_framing": camera.get("framing", []),
        "lens": camera.get("lenses", []),
        "depth": camera.get("depth", []),
        "mood": catalog.section("moods"),
        "pose": catalog.section("poses"),
        "action": catalog.section("actions"),
    }
    return list(mapping.get(key, []))


def mutate_gene(
    catalog: Catalog,
    key: str,
    current_id: Optional[str],
    sfw_level: float,
    temperature: float,
) -> Optional[str]:
    seq = _gene_sequence(catalog, key)
    if not seq:
        return current_id
    selector = WeightedSelector(sfw_level=sfw_level, temperature=temperature)
    choice = selector.pick(seq)
    return choice.get("id")


def crossover_genes(a: GeneSet, b: GeneSet) -> GeneSet:
    child: GeneSet = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        if maybe(0.5):
            child[key] = a.get(key)
        else:
            child[key] = b.get(key, a.get(key))
    return child


def load_best_gene_sets(db_path: Path, k: int, session_id: Optional[str] = None) -> List[GeneSet]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    limit = max(1, k * 3)
    if session_id:
        cur.execute(
            """
            SELECT params FROM prompts
            WHERE fitness IS NOT NULL
              AND params LIKE '%"struct"%'
              AND params LIKE ?
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (f"%{session_id}%", limit),
        )
    else:
        cur.execute(
            """
            SELECT params FROM prompts
            WHERE fitness IS NOT NULL
              AND params LIKE '%"struct"%'
            ORDER BY fitness DESC
            LIMIT ?
            """,
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()

    best: List[GeneSet] = []
    seen: set[str] = set()
    for (params_json,) in rows:
        try:
            params = json.loads(params_json)
            genes = params.get("struct", {}).get("gene_ids")
            if not isinstance(genes, dict):
                continue
            key = json.dumps(genes, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            best.append(genes)
            if len(best) >= k:
                break
        except Exception:
            continue
    return best
