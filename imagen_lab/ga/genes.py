from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from imagen_lab.catalog import Catalog
from imagen_lab.randomization import WeightedSelector, maybe

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
    query = (
        """
        SELECT params_json, gene_choices_json
        FROM prompts
        WHERE fitness IS NOT NULL
        """
    )
    params: tuple = ()
    if session_id:
        query += " AND params_json LIKE ?"
        params = (f"%{session_id}%",)
    query += " ORDER BY fitness DESC LIMIT ?"
    params += (limit,)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    best: List[GeneSet] = []
    seen: set[str] = set()
    for params_json, gene_json in rows:
        try:
            genes: Optional[Dict[str, Optional[str]]] = None
            if gene_json:
                payload = json.loads(gene_json)
                if isinstance(payload, dict):
                    choices = payload.get("choices")
                    if isinstance(choices, dict):
                        genes = {str(slot): value.get("id") if isinstance(value, dict) else value for slot, value in choices.items()}
            if genes is None:
                params = json.loads(params_json) if params_json else {}
                struct = params.get("struct") if isinstance(params, dict) else {}
                if isinstance(struct, dict):
                    raw_genes = struct.get("gene_ids")
                    if isinstance(raw_genes, dict):
                        genes = {str(k): v for k, v in raw_genes.items()}
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
