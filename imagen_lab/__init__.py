from __future__ import annotations

"""High-level helpers for the Imagen Genetics pipeline."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from .config import PipelineConfig, load_config
    from .pipeline import run_evolve, run_plain

__all__ = ["PipelineConfig", "load_config", "run_plain", "run_evolve"]


def __getattr__(name: str) -> Any:  # pragma: no cover - dispatch helper
    if name in {"PipelineConfig", "load_config"}:
        module = import_module(".config", __name__)
    elif name in {"run_plain", "run_evolve"}:
        module = import_module(".pipeline", __name__)
    else:
        raise AttributeError(name)

    value = getattr(module, name)
    globals()[name] = value
    return value
