from __future__ import annotations

from typing import Optional

from .metrics import ScoreReport


def format_metrics(report: ScoreReport | None) -> Optional[str]:
    if report is None:
        return None
    meso = report.meso
    parts: list[str] = []

    def _append(label: str, key: str, source: dict[str, float]) -> None:
        value = source.get(key)
        if value is None:
            return
        try:
            parts.append(f"{label}={float(value):.2f}")
        except Exception:  # pragma: no cover - defensive
            return

    _append("visual", "fitness_visual", meso)
    _append("style", "fitness_style", meso)
    _append("coverage", "fitness_coverage", meso)
    _append("alignment", "fitness_alignment", meso)
    _append("cleanliness", "fitness_cleanliness", meso)
    _append("era", "fitness_era_match", meso)
    _append("novelty", "fitness_novelty", meso)

    if not parts:
        return None
    return " ".join(parts)
