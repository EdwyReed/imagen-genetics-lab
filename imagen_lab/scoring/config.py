from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ClipTextHeadClass:
    """Configuration for a single semantic class of a CLIP text head."""

    label: str
    prompts: List[str]


@dataclass(frozen=True)
class ClipTextHead:
    """Configuration for a CLIP softmax head defined by text prompts."""

    key: str
    groups: List[ClipTextHeadClass]
    display_name: Optional[str] = None
    primary: Optional[str] = None
    calibration: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipTextHeadsConfig:
    """Collection of CLIP text heads alongside model selection metadata."""

    clip_model: Tuple[str, str]
    heads: List[ClipTextHead]


_DEFAULT_CONFIG = Path(__file__).with_name("clip_text_heads.yaml")


def _ensure_prompts(prompts: Iterable[str]) -> List[str]:
    values = [str(p).strip() for p in prompts if str(p).strip()]
    if not values:
        raise ValueError("Each head class must define at least one non-empty prompt")
    return values


def load_clip_text_heads(path: Optional[Path | str] = None) -> ClipTextHeadsConfig:
    """Load CLIP text head configuration from YAML/JSON."""

    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Clip text head config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    model_cfg = raw.get("model", {})
    clip_model = (
        str(model_cfg.get("name", "ViT-B-32")),
        str(model_cfg.get("pretrained", "openai")),
    )

    heads: List[ClipTextHead] = []
    for item in raw.get("heads", []):
        key = str(item.get("key")) if item.get("key") is not None else None
        if not key:
            raise ValueError("Each head entry must define a non-empty 'key'")

        groups_data = item.get("groups", [])
        if not groups_data:
            raise ValueError(f"Head '{key}' must define at least one group")

        groups: List[ClipTextHeadClass] = []
        for group in groups_data:
            label = str(group.get("label")) if group.get("label") is not None else None
            if not label:
                raise ValueError(f"Head '{key}' has a group without a 'label'")
            prompts = _ensure_prompts(group.get("prompts", []))
            groups.append(ClipTextHeadClass(label=label, prompts=prompts))

        calibration: Dict[str, Tuple[float, float]] = {}
        for label, cal in (item.get("calibration") or {}).items():
            if cal is None:
                continue
            try:
                lo, hi = cal
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Calibration for head '{key}' label '{label}' must be a sequence of two floats"
                ) from exc
            calibration[str(label)] = (float(lo), float(hi))

        heads.append(
            ClipTextHead(
                key=key,
                groups=groups,
                display_name=str(item.get("display_name")) if item.get("display_name") else None,
                primary=str(item.get("primary")) if item.get("primary") else None,
                calibration=calibration,
            )
        )

    if not heads:
        raise ValueError("Configuration must define at least one clip text head")

    return ClipTextHeadsConfig(clip_model=clip_model, heads=heads)
