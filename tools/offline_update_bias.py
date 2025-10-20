#!/usr/bin/env python3
"""Offline bias updater applying simple EMA on palette/pose combinations."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("bias.update")

MIN_WEIGHT = 0.8
MAX_WEIGHT = 1.5
ALPHA = 0.3


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_bias(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"weights": {}, "locks": {}}


def load_runs(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed line: %s", line[:100])


def aggregate_scores(runs: Iterable[dict]) -> Dict[Tuple[str, str], float]:
    aggregates: Dict[Tuple[str, str], list] = defaultdict(list)
    for entry in runs:
        gene_ids = entry.get("gene_ids", {})
        palette = gene_ids.get("palette")
        pose = gene_ids.get("pose")
        if not palette or not pose:
            continue
        scores = entry.get("scores", {})
        on_style = float(scores.get("on_style", 0.0))
        violations = float(scores.get("violations", 0.0))
        uplift = on_style - violations
        aggregates[(palette, pose)].append(uplift)
    return {key: sum(values) / len(values) for key, values in aggregates.items() if values}


def update_bias(bias: dict, aggregates: Dict[Tuple[str, str], float]) -> dict:
    weights = bias.setdefault("weights", {})
    changes = []
    for (palette, pose), score in aggregates.items():
        target = clamp(1.0 + score / 100.0, MIN_WEIGHT, MAX_WEIGHT)
        for group, asset_id in (("palettes", palette), ("poses", pose)):
            key = f"{group}:{asset_id}"
            current = float(weights.get(key, 1.0))
            updated = clamp(current * (1 - ALPHA) + target * ALPHA, MIN_WEIGHT, MAX_WEIGHT)
            weights[key] = round(updated, 4)
            changes.append((key, current, updated))
    for key, old, new in changes:
        if abs(old - new) > 1e-6:
            LOGGER.info("weight %s: %.4f -> %.4f", key, old, new)
    return bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline bias weight updater")
    parser.add_argument("--runs", required=True, type=Path, help="Path to runs.jsonl")
    parser.add_argument("--bias", required=True, type=Path, help="Path to bias.json")
    parser.add_argument("--output", type=Path, help="Optional output path for updated bias")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bias = load_bias(args.bias)
    aggregates = aggregate_scores(load_runs(args.runs))
    if not aggregates:
        LOGGER.warning("No valid palette/pose scores found; nothing to update")
        return
    updated = update_bias(bias, aggregates)
    target_path = args.output or args.bias
    target_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Updated bias written to %s", target_path)


if __name__ == "__main__":  # pragma: no cover
    main()
