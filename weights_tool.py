#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for managing DualScorer weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from imagen_lab.config import load_config
from scorer import DualScorer, W_CLIP, W_SPEC, W_ILLU

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _collect_images(directory: Path) -> List[Path]:
    paths: List[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTS:
            paths.append(entry)
    return paths


def _write_config(config_path: Path, data: Dict) -> None:
    config_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml(path: Path) -> Dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a mapping at the top level")
    return raw


def reset_weights(path: Path) -> None:
    data = _load_yaml(path)
    scoring = data.setdefault("scoring", {})
    scoring["weights"] = {"clip": W_CLIP, "spec": W_SPEC, "illu": W_ILLU}
    _write_config(path, data)
    print(json.dumps({"status": "reset", "weights": scoring["weights"]}, ensure_ascii=False, indent=2))


def calibrate_from_directory(
    config_path: Path,
    reference_dir: Path,
    target: float,
    min_weight: float,
    dry_run: bool,
) -> None:
    config = load_config(config_path)
    scorer = DualScorer(
        device=config.scoring.device,
        batch=config.scoring.batch_size,
        db_path=config.paths.database,
        jsonl_path=config.paths.scores_jsonl,
        weights=config.scoring.weights,
        tau=config.scoring.tau,
        cal_style=config.scoring.cal_style,
        cal_illu=config.scoring.cal_illu,
        auto_weights={"enabled": False},
    )

    images = _collect_images(reference_dir)
    if not images:
        raise SystemExit(f"No reference images found in {reference_dir}")

    components: List[List[float]] = []
    for img_path in images:
        result = scorer.score_one(img_path)
        components.append([result.clip_style, result.specular, result.illu_bias])

    matrix = np.asarray(components, dtype=float)
    target_vec = np.full((matrix.shape[0],), target, dtype=float)

    try:
        solution, *_ = np.linalg.lstsq(matrix, target_vec, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise SystemExit(f"Failed to solve for weights: {exc}")

    clipped = np.clip(solution, min_weight, None)
    if clipped.sum() <= 0:
        raise SystemExit("Computed weights are degenerate; try a different target or dataset")
    normalized = clipped / clipped.sum()

    weights = {
        "clip": float(normalized[0]),
        "spec": float(normalized[1]),
        "illu": float(normalized[2]),
    }

    if dry_run:
        print(json.dumps({"status": "calculated", "weights": weights}, ensure_ascii=False, indent=2))
        return

    data = _load_yaml(config_path)
    scoring = data.setdefault("scoring", {})
    scoring["weights"] = weights
    _write_config(config_path, data)
    print(json.dumps({"status": "updated", "weights": weights}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DualScorer weight management")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--reset", action="store_true", help="Reset weights to factory defaults")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Directory with reference images used to calibrate weights",
    )
    parser.add_argument("--target", type=float, default=0.9, help="Desired style score for reference set")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Lower bound applied to solved weights before normalisation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Calculate weights without touching the config file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.reset and args.reference_dir:
        parser.error("--reset cannot be combined with --reference-dir")

    if args.reset:
        reset_weights(config_path)
        return

    if not args.reference_dir:
        parser.error("Provide --reference-dir or use --reset")

    calibrate_from_directory(
        config_path,
        args.reference_dir,
        target=args.target,
        min_weight=args.min_weight,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

