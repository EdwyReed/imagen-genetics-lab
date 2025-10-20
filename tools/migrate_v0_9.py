#!/usr/bin/env python3
"""Skeleton migrator for bringing legacy assets into the v0.9 layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def migrate_legacy_asset(source: Path, target_dir: Path) -> None:
    """Copy a legacy JSON file into the new per-group directory."""
    data = json.loads(source.read_text(encoding="utf-8"))
    asset_id = data.get("id") or source.stem
    group = data.get("group")
    if not group:
        raise ValueError(f"Legacy asset {source} must declare a group")
    target = target_dir / group / f"{asset_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy assets to v0.9 structure")
    parser.add_argument("source", type=Path, help="Path to legacy asset directory")
    parser.add_argument("target", type=Path, help="Destination directory (e.g. data/packs/base)")
    args = parser.parse_args()
    for path in args.source.rglob("*.json"):
        migrate_legacy_asset(path, args.target)


if __name__ == "__main__":  # pragma: no cover
    main()
