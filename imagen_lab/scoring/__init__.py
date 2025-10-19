"""Utilities for CLIP scoring configuration."""

from .core import ScoreInputs, ScoreReport, compute_score_report
from .weights_table import (
    DEFAULT_STYLE_WEIGHTS,
    STYLE_COMPONENT_KEYS,
    StyleComposition,
    StyleMixer,
    WeightProfileTable,
    normalize_weights,
)

try:  # pragma: no cover - optional dependency guard
    from .config import (
        ClipTextHead,
        ClipTextHeadClass,
        ClipTextHeadsConfig,
        load_clip_text_heads,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - missing yaml
    if exc.name != "yaml":
        raise

    class ClipTextHead:  # type: ignore[empty-body]
        """Placeholder exported when PyYAML is unavailable."""

    class ClipTextHeadClass:  # type: ignore[empty-body]
        """Placeholder exported when PyYAML is unavailable."""

    class ClipTextHeadsConfig:  # type: ignore[empty-body]
        """Placeholder exported when PyYAML is unavailable."""

    def load_clip_text_heads(*_: object, **__: object) -> None:
        raise ModuleNotFoundError("PyYAML is required to load clip text head configuration") from exc

__all__ = [
    "ClipTextHead",
    "ClipTextHeadClass",
    "ClipTextHeadsConfig",
    "DEFAULT_STYLE_WEIGHTS",
    "STYLE_COMPONENT_KEYS",
    "StyleComposition",
    "StyleMixer",
    "WeightProfileTable",
    "load_clip_text_heads",
    "normalize_weights",
    "ScoreInputs",
    "ScoreReport",
    "compute_score_report",
]
