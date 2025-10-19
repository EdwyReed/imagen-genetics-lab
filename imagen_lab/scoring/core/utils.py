from __future__ import annotations

from typing import Dict, List, Optional


def format_metrics(metrics: Dict[str, object]) -> Optional[str]:
    if not metrics:
        return None
    pieces: List[str] = []
    batch = metrics.get("batch", {}) if isinstance(metrics, dict) else {}
    history = metrics.get("history", {}) if isinstance(metrics, dict) else {}
    style = metrics.get("style", {}) if isinstance(metrics, dict) else {}
    composition = metrics.get("composition", {}) if isinstance(metrics, dict) else {}

    try:
        pairwise_mean = batch.get("pairwise_mean")  # type: ignore[assignment]
        if pairwise_mean is not None:
            pieces.append(f"batch_mean={float(pairwise_mean):.3f}")
    except Exception:
        pass
    try:
        pairwise_min = batch.get("pairwise_min")  # type: ignore[assignment]
        if pairwise_min is not None:
            pieces.append(f"batch_min={float(pairwise_min):.3f}")
    except Exception:
        pass
    try:
        history_mean = history.get("mean_distance")  # type: ignore[assignment]
        if history_mean is not None:
            pieces.append(f"history_mean={float(history_mean):.3f}")
    except Exception:
        pass
    try:
        history_min = history.get("min_distance")  # type: ignore[assignment]
        if history_min is not None:
            pieces.append(f"history_min={float(history_min):.3f}")
    except Exception:
        pass
    size = history.get("size") if isinstance(history, dict) else None
    if size:
        try:
            pieces.append(f"history_size={int(size)}")
        except Exception:
            pass
    try:
        style_mean = style.get("mean_total")  # type: ignore[assignment]
        if style_mean is not None:
            pieces.append(f"style_mean={float(style_mean):.3f}")
    except Exception:
        pass
    contributions = style.get("mean_contributions") if isinstance(style, dict) else None
    if isinstance(contributions, dict) and contributions:
        try:
            contrib_bits = [
                f"{key}:{float(val):.3f}" for key, val in sorted(contributions.items())
            ]
            pieces.append(f"style_contribs={'/'.join(contrib_bits)}")
        except Exception:
            pass
    if isinstance(composition, dict) and composition:
        try:
            crop = composition.get("mean_cropping_tightness")
            thirds = composition.get("mean_thirds_alignment")
            neg = composition.get("mean_negative_space")
            comp_bits = []
            if crop is not None:
                comp_bits.append(f"crop={float(crop):.3f}")
            if thirds is not None:
                comp_bits.append(f"thirds={float(thirds):.3f}")
            if neg is not None:
                comp_bits.append(f"neg_space={float(neg):.3f}")
            if comp_bits:
                pieces.append(f"composition({' '.join(comp_bits)})")
        except Exception:
            pass
    if not pieces:
        return None
    return " ".join(pieces)
