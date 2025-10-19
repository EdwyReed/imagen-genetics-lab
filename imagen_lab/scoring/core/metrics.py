"""Scoring helpers that compute micro metrics and meso aggregates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from imagen_lab.config import TemperatureConfig

_DEFAULT_TAU = 0.07


@dataclass(frozen=True)
class ScoreInputs:
    """Normalized view of the artefacts required to compute metrics."""

    image: Any
    caption: str
    scene: Any


@dataclass(frozen=True)
class ScoreReport:
    """Container with both micro metrics and meso aggregates."""

    micro: Dict[str, float]
    meso: Dict[str, float]
    temperatures: Dict[str, float]

    def public_payload(self) -> Dict[str, Any]:
        """Return the payload intended for UI consumption (no micro metrics)."""

        return {
            "aggregates": dict(self.meso),
            "temperatures": dict(self.temperatures),
        }


def compute_score_report(
    inputs: ScoreInputs,
    temperatures: TemperatureConfig | Mapping[str, float] | float | None = None,
) -> ScoreReport:
    """Compute micro metrics and meso aggregates for the provided artefacts."""

    temp_cfg = _resolve_temperatures(temperatures)
    image = _ensure_image(inputs.image)
    caption = (inputs.caption or "").strip()
    scene = _ensure_scene_mapping(inputs.scene)

    micro = _compute_micro_metrics(image, caption, scene, temp_cfg)
    meso = _compute_meso_aggregates(micro)
    return ScoreReport(micro=micro, meso=meso, temperatures=temp_cfg.as_dict())


# ---------------------------------------------------------------------------
# Micro metrics helpers
# ---------------------------------------------------------------------------


def _compute_micro_metrics(
    image: np.ndarray,
    caption: str,
    scene: Mapping[str, Any],
    temperatures: TemperatureConfig,
) -> Dict[str, float]:
    micro: Dict[str, float] = {}

    style_core = _style_core_score(image)
    gloss_intensity = _gloss_intensity(image)
    softness_blur = _softness_blur(image)

    chest_focus = _keyword_scene_score(
        caption,
        scene,
        "chest_focus",
        ("chest", "cleavage", "torso", "bust"),
    )
    thigh_focus = _keyword_scene_score(
        caption,
        scene,
        "thigh_focus",
        ("thigh", "leg", "legs", "hips"),
    )
    pose_suggestiveness = _keyword_scene_score(
        caption,
        scene,
        "pose_suggestiveness",
        ("flirtatious", "seductive", "alluring", "playful", "suggestive"),
    )

    coverage_ratio = _coverage_ratio(scene, image)
    skin_exposure = _skin_exposure(scene, image)
    coverage_target_alignment = _coverage_alignment(scene, coverage_ratio)

    clip_prompt_alignment = temperatures.apply(
        "clip_prompt_alignment", _caption_alignment(caption, scene)
    )
    identity_consistency = temperatures.apply(
        "identity_consistency", _identity_consistency(caption, scene)
    )
    pose_coherence = _pose_coherence(caption, scene)
    ai_artifacts = temperatures.apply("ai_artifacts", _ai_artifacts_score(image))
    visual_noise_level = temperatures.apply(
        "visual_noise_level", _visual_noise_level(image)
    )

    palette_era_match = _era_match(scene, ("palette", "palette_primary"))
    wardrobe_era_match = _era_match(scene, ("wardrobe", "wardrobe_main"))
    environment_era_match = _era_match(scene, ("background", "location"))

    novelty_palette = _novelty_score(scene, ("palette", "palette_primary", "color"))
    novelty_pose = _novelty_score(scene, ("pose", "gesture"))
    novelty_props = _novelty_score(scene, ("props", "prop"))

    micro.update(
        {
            "style_core": style_core,
            "gloss_intensity": gloss_intensity,
            "softness_blur": softness_blur,
            "chest_focus": chest_focus,
            "thigh_focus": thigh_focus,
            "pose_suggestiveness": pose_suggestiveness,
            "coverage_ratio": coverage_ratio,
            "skin_exposure": skin_exposure,
            "coverage_target_alignment": coverage_target_alignment,
            "clip_prompt_alignment": clip_prompt_alignment,
            "identity_consistency": identity_consistency,
            "pose_coherence": pose_coherence,
            "ai_artifacts": ai_artifacts,
            "visual_noise_level": visual_noise_level,
            "palette_era_match": palette_era_match,
            "wardrobe_era_match": wardrobe_era_match,
            "environment_era_match": environment_era_match,
            "novelty_palette": novelty_palette,
            "novelty_pose": novelty_pose,
            "novelty_props": novelty_props,
        }
    )
    return micro


def _compute_meso_aggregates(micro: Mapping[str, float]) -> Dict[str, float]:
    def _v(key: str) -> float:
        return float(micro.get(key, 0.0))

    fitness_style = _clamp(
        0.5 * _v("style_core")
        + 0.3 * _v("gloss_intensity")
        + 0.2 * _v("softness_blur")
    )
    fitness_body_focus = _clamp(
        0.4 * _v("chest_focus")
        + 0.4 * _v("thigh_focus")
        + 0.2 * _v("pose_suggestiveness")
    )
    fitness_coverage = _clamp(
        0.5 * _v("coverage_ratio")
        + 0.3 * (1.0 - _v("skin_exposure"))
        + 0.2 * _v("coverage_target_alignment")
    )
    fitness_alignment = _clamp(
        0.45 * _v("clip_prompt_alignment")
        + 0.35 * _v("identity_consistency")
        + 0.20 * _v("pose_coherence")
    )
    fitness_cleanliness = _clamp(
        0.6 * (1.0 - _v("ai_artifacts")) + 0.4 * (1.0 - _v("visual_noise_level"))
    )
    fitness_era_match = _clamp(
        0.4 * _v("palette_era_match")
        + 0.35 * _v("wardrobe_era_match")
        + 0.25 * _v("environment_era_match")
    )
    fitness_novelty = _clamp(
        0.34 * _v("novelty_palette")
        + 0.33 * _v("novelty_pose")
        + 0.33 * _v("novelty_props")
    )
    fitness_visual = _clamp(
        0.35 * fitness_style
        + 0.25 * fitness_alignment
        + 0.20 * fitness_cleanliness
        + 0.20 * fitness_era_match
    )

    return {
        "fitness_style": fitness_style,
        "fitness_body_focus": fitness_body_focus,
        "fitness_coverage": fitness_coverage,
        "fitness_alignment": fitness_alignment,
        "fitness_cleanliness": fitness_cleanliness,
        "fitness_era_match": fitness_era_match,
        "fitness_novelty": fitness_novelty,
        "fitness_visual": fitness_visual,
    }


# ---------------------------------------------------------------------------
# Individual metric estimators
# ---------------------------------------------------------------------------


def _ensure_image(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.size == 0:
        arr = np.zeros((1, 1, 3), dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return arr


def _ensure_scene_mapping(scene: Any) -> Mapping[str, Any]:
    if scene is None:
        return {}
    if isinstance(scene, Mapping):
        return scene
    to_dict = getattr(scene, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, Mapping):
                return result
        except Exception:  # noqa: BLE001
            pass
    return {}


def _style_core_score(image: np.ndarray) -> float:
    saturation = _saturation(image).mean()
    contrast = float(np.std(image)) * 1.5
    return _clamp(0.65 * saturation + 0.35 * _clamp(contrast))


def _gloss_intensity(image: np.ndarray) -> float:
    value = image.max(axis=2)
    highlight_ratio = float((value > 0.82).mean())
    return _clamp(0.7 * highlight_ratio + 0.3 * float(value.mean()))


def _softness_blur(image: np.ndarray) -> float:
    gray = image.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((dx.mean() + dy.mean()) * 0.5)
    sharpness = _clamp(edge_strength * 4.0)
    return _clamp(1.0 - sharpness)


def _keyword_scene_score(
    caption: str,
    scene: Mapping[str, Any],
    key: str,
    keywords: Sequence[str],
) -> float:
    base = _keyword_score(caption, keywords)
    hint = _scene_number(scene, key)
    if hint is None:
        return base
    return _clamp(0.5 * base + 0.5 * hint)


def _coverage_ratio(scene: Mapping[str, Any], image: np.ndarray) -> float:
    coverage = _scene_number(scene, "coverage")
    if coverage is not None:
        return _clamp(coverage)
    pose_coverage = _scene_number(scene, "coverage_ratio")
    if pose_coverage is not None:
        return _clamp(pose_coverage)
    return _clamp(1.0 - _estimate_skin_ratio(image))


def _skin_exposure(scene: Mapping[str, Any], image: np.ndarray) -> float:
    hint = _scene_number(scene, "skin_ratio")
    if hint is not None:
        return _clamp(hint)
    return _clamp(_estimate_skin_ratio(image))


def _coverage_alignment(scene: Mapping[str, Any], coverage_ratio: float) -> float:
    target = _scene_number(scene, "coverage_target")
    if target is None:
        return _clamp(1.0 - abs(coverage_ratio - 0.55))
    return _clamp(1.0 - abs(coverage_ratio - _clamp(target)))


def _caption_alignment(caption: str, scene: Mapping[str, Any]) -> float:
    if not caption:
        return 0.0
    caption_tokens = set(_tokenize(caption))
    if not caption_tokens:
        return 0.0
    expected_words = set()
    summary = _scene_string(scene, "summary") or _scene_string(scene, "scene_summary")
    if summary:
        expected_words.update(_tokenize(summary))
    for key in ("pose", "lighting", "palette", "background", "wardrobe_main"):
        label = _scene_label(scene, key)
        if label:
            expected_words.update(_tokenize(label))
    if not expected_words:
        return 0.5
    overlap = expected_words & caption_tokens
    return _clamp(len(overlap) / max(len(expected_words), 1))


def _identity_consistency(caption: str, scene: Mapping[str, Any]) -> float:
    character = _scene_mapping(scene, "character")
    if not character:
        return 0.5
    name = character.get("name")
    if not isinstance(name, str) or not name.strip():
        return 0.5
    lowered = caption.lower()
    if name.lower() in lowered:
        return 1.0
    aliases = character.get("aliases")
    if isinstance(aliases, Sequence):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip() and alias.lower() in lowered:
                return 0.9
    return 0.4


def _pose_coherence(caption: str, scene: Mapping[str, Any]) -> float:
    pose_label = _scene_label(scene, "pose")
    if not pose_label:
        return 0.5
    tokens = set(_tokenize(caption))
    hits = len(tokens & set(_tokenize(pose_label)))
    if hits == 0:
        return 0.45
    confidence = _scene_number(scene, "pose_confidence")
    if confidence is None:
        confidence = 0.7
    return _clamp(0.6 + 0.4 * min(1.0, hits / 3.0) * confidence)


def _ai_artifacts_score(image: np.ndarray) -> float:
    gray = image.mean(axis=2)
    blurred = _box_blur(gray)
    residual = np.abs(gray - blurred)
    return _clamp(float(residual.mean() * 6.0))


def _visual_noise_level(image: np.ndarray) -> float:
    gray = image.mean(axis=2)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    gradients = np.sqrt(np.square(dx[:-1, :]) + np.square(dy[:, :-1]))
    return _clamp(float(np.std(gradients) * 2.0))


def _era_match(scene: Mapping[str, Any], keys: Sequence[str]) -> float:
    target = _scene_number(scene, "era_target")
    hint = None
    for key in keys:
        hint = _scene_number(scene, f"{key}_era_score")
        if hint is not None:
            break
        label = _scene_label(scene, key)
        if label:
            hint = _era_label_to_score(label)
            if hint is not None:
                break
    if target is None and hint is None:
        return 0.5
    if target is None:
        target = 0.7
    if hint is None:
        hint = target
    return _clamp(1.0 - abs(_clamp(target) - _clamp(hint)))


def _novelty_score(scene: Mapping[str, Any], prefixes: Sequence[str]) -> float:
    gene_ids = _scene_mapping(scene, "gene_ids")
    if not gene_ids:
        return 0.5
    option_prob = _scene_mapping(scene, "option_probabilities")
    scores: list[float] = []
    for slot, selected in gene_ids.items():
        if not isinstance(slot, str):
            continue
        if not _slot_matches(slot, prefixes):
            continue
        prob = _selected_probability(option_prob, slot, selected)
        if prob is None:
            continue
        scores.append(1.0 - _clamp(prob))
    if not scores:
        return 0.5
    return _clamp(float(sum(scores) / len(scores)))


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _resolve_temperatures(
    raw: TemperatureConfig | Mapping[str, float] | float | None,
) -> TemperatureConfig:
    if isinstance(raw, TemperatureConfig):
        return TemperatureConfig.from_mapping(raw)
    if raw is None:
        return TemperatureConfig(default=_DEFAULT_TAU)
    if isinstance(raw, Mapping):
        return TemperatureConfig.from_mapping(raw)
    return TemperatureConfig(default=float(raw))


def _scene_number(scene: Mapping[str, Any], key: str) -> float | None:
    value = _search_scene(scene, key, lambda item: isinstance(item, (int, float)))
    if value is None:
        return None
    return float(value)


def _scene_string(scene: Mapping[str, Any], key: str) -> str | None:
    value = _search_scene(scene, key, lambda item: isinstance(item, str))
    if value is None:
        return None
    return str(value)


def _scene_mapping(scene: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _search_scene(
        scene, key, lambda item: isinstance(item, Mapping) and not isinstance(item, (str, bytes))
    )
    if isinstance(value, Mapping):
        return value
    return {}


def _scene_label(scene: Mapping[str, Any], key: str) -> str | None:
    mapping = _scene_mapping(scene, key)
    if mapping:
        for field in ("label", "name", "summary", "option"):
            val = mapping.get(field)
            if isinstance(val, str) and val.strip():
                return val
    raw = _scene_string(scene, key)
    if raw:
        return raw
    return None


def _search_scene(
    scene: Mapping[str, Any],
    key: str,
    predicate,
) -> Any:
    stack: list[Any] = [scene]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, Mapping):
            if key in current:
                candidate = current[key]
                if predicate(candidate):
                    return candidate
            for value in current.values():
                if isinstance(value, Mapping):
                    stack.append(value)
                elif isinstance(value, (list, tuple)):
                    stack.extend(v for v in value if isinstance(v, Mapping))
    return None


def _slot_matches(slot: str, prefixes: Sequence[str]) -> bool:
    lowered = slot.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes)


def _selected_probability(
    option_prob: Mapping[str, Any], slot: str, selected: Any
) -> float | None:
    if not option_prob:
        return None
    options = option_prob.get(slot)
    if not options:
        return None
    if isinstance(selected, Mapping):
        selected_id = str(selected.get("id"))
    else:
        selected_id = str(selected)
    for option in _iter_options(options):
        opt_id = option.get("id")
        if opt_id is None:
            continue
        if str(opt_id) == selected_id:
            probability = option.get("probability")
            if isinstance(probability, (int, float)):
                return float(probability)
    return None


def _iter_options(options: Any) -> Iterable[MutableMapping[str, Any]]:
    if isinstance(options, Mapping):
        yield options  # type: ignore[misc]
        return
    if isinstance(options, (list, tuple)):
        for item in options:
            if isinstance(item, Mapping):
                yield item  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Low level numeric helpers
# ---------------------------------------------------------------------------


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    current = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _keyword_score(text: str, keywords: Sequence[str]) -> float:
    if not text:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    hits = sum(tokens.count(keyword.lower()) for keyword in keywords)
    if hits <= 0:
        return 0.0
    return _clamp(1.0 - math.exp(-hits / 2.0))


def _saturation(image: np.ndarray) -> np.ndarray:
    maxc = image.max(axis=2)
    minc = image.min(axis=2)
    delta = maxc - minc
    denom = np.where(maxc == 0, 1.0, maxc)
    return np.where(denom > 0, delta / denom, 0.0)


def _estimate_skin_ratio(image: np.ndarray) -> float:
    hsv = _rgb_to_hsv(image)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (h > 0.0) & (h < 0.15) & (s > 0.23) & (s < 0.68) & (v > 0.35) & (v < 0.95)
    return float(mask.mean())


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mask & (maxc == r), ((g - b) / (delta + 1e-6)) % 6, h)
    h = np.where(mask & (maxc == g), (b - r) / (delta + 1e-6) + 2, h)
    h = np.where(mask & (maxc == b), (r - g) / (delta + 1e-6) + 4, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _box_blur(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    result = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return result


def _era_label_to_score(label: str) -> float | None:
    normalized = label.strip().lower()
    mapping = {
        "modern": 0.9,
        "contemporary": 0.9,
        "futuristic": 1.0,
        "future": 1.0,
        "retro": 0.7,
        "vintage": 0.65,
        "classic": 0.6,
        "80s": 0.8,
        "1980s": 0.8,
        "70s": 0.7,
        "1970s": 0.7,
        "90s": 0.85,
        "1990s": 0.85,
        "mid-century": 0.55,
    }
    if normalized in mapping:
        return mapping[normalized]
    return None

