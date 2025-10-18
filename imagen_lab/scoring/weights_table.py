from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, Mapping, Optional, Tuple

import yaml

DEFAULT_STYLE_WEIGHTS: Dict[str, float] = {
    "clip": 0.20,
    "spec": 0.15,
    "illu": 0.08,
    "retro": 0.10,
    "medium": 0.10,
    "sensual": 0.10,
    "pose": 0.08,
    "camera": 0.05,
    "color": 0.05,
    "accessories": 0.04,
    "composition": 0.03,
    "skin_glow": 0.02,
}
STYLE_COMPONENT_KEYS: Tuple[str, ...] = tuple(DEFAULT_STYLE_WEIGHTS.keys())


def normalize_weights(
    weights: Mapping[str, float],
    *,
    keys: Optional[Iterable[str]] = None,
    defaults: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Normalize a mapping of weights while clamping negatives to zero.

    Parameters
    ----------
    weights:
        Raw weight values.
    keys:
        The component keys that must be present in the output mapping.
    defaults:
        Fallback weights used when the provided mapping is degenerate.
    """

    if keys is None:
        if defaults:
            keys = tuple(defaults.keys())
        elif weights:
            keys = tuple(weights.keys())
        else:
            keys = STYLE_COMPONENT_KEYS
    keys = tuple(keys)

    total = 0.0
    cleaned: Dict[str, float] = {}
    for key in keys:
        value = float(weights.get(key, 0.0))
        if value > 0:
            cleaned[key] = value
            total += value
        else:
            cleaned[key] = 0.0

    if total <= 0 and defaults:
        return {
            key: float(defaults.get(key, 0.0))
            for key in keys
        }

    if total <= 0:
        uniform = 1.0 / float(len(tuple(keys))) if keys else 0.0
        return {key: uniform for key in keys}

    return {key: cleaned.get(key, 0.0) / total for key in keys}


@dataclass(frozen=True)
class StyleComposition:
    total: float
    components: Dict[str, float]
    contributions: Dict[str, float]
    weights: Dict[str, float]


class WeightProfileTable:
    """Mutable table of style-weight profiles stored in YAML."""

    def __init__(
        self,
        profiles: Mapping[str, Mapping[str, float]],
        *,
        path: Optional[Path] = None,
        defaults: Optional[Mapping[str, float]] = None,
        default_profile: str = "default",
    ) -> None:
        self._lock = RLock()
        self._defaults = dict(defaults or DEFAULT_STYLE_WEIGHTS)
        self._path = Path(path) if path is not None else None
        self._default_profile = default_profile
        self._profiles: Dict[str, Dict[str, float]] = {}
        for name, weights in profiles.items():
            self._profiles[name] = normalize_weights(weights, defaults=self._defaults)
        if default_profile not in self._profiles and self._defaults:
            self._profiles[default_profile] = dict(self._defaults)

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def default_profile(self) -> str:
        return self._default_profile

    def profile_names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._profiles.keys()))

    def get_profile(
        self,
        name: Optional[str] = None,
        *,
        fallback: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        profile = name or self._default_profile
        with self._lock:
            if profile in self._profiles:
                return dict(self._profiles[profile])
        base = fallback or self._defaults
        return normalize_weights(base, defaults=self._defaults)

    def ensure_profile(self, name: str) -> Dict[str, float]:
        with self._lock:
            if name not in self._profiles:
                self._profiles[name] = dict(self._defaults)
            return dict(self._profiles[name])

    def update_profile(
        self,
        name: str,
        weights: Mapping[str, float],
        *,
        persist: bool = False,
    ) -> Dict[str, float]:
        normalized = normalize_weights(weights, defaults=self._defaults)
        with self._lock:
            self._profiles[name] = dict(normalized)
        if persist:
            self.persist()
        return dict(normalized)

    def persist(self) -> None:
        if self._path is None:
            return
        with self._lock:
            payload = {"profiles": self._profiles, "default": self._default_profile}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        defaults: Optional[Mapping[str, float]] = None,
        default_profile: str = "default",
        create: bool = False,
    ) -> "WeightProfileTable":
        path = Path(path)
        if not path.exists():
            if not create:
                raise FileNotFoundError(path)
            table = cls({}, path=path, defaults=defaults, default_profile=default_profile)
            table.persist()
            return table

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Weight profile file must contain a mapping")
        profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else None
        if profiles is None:
            # legacy structure without explicit grouping
            profiles = {k: v for k, v in data.items() if isinstance(v, Mapping)}
        default_name = data.get("default", default_profile)
        return cls(
            profiles,
            path=path,
            defaults=defaults,
            default_profile=str(default_name),
        )


class StyleMixer:
    """Helper responsible for combining clip/spec/illu components."""

    def __init__(
        self,
        *,
        weights: Optional[Mapping[str, float]] = None,
        defaults: Optional[Mapping[str, float]] = None,
        weight_table: Optional[WeightProfileTable] = None,
        profile: str = "default",
        persist_updates: bool = False,
    ) -> None:
        self._defaults = dict(defaults or DEFAULT_STYLE_WEIGHTS)
        self._table = weight_table
        self._profile = profile
        self._persist_updates = persist_updates

        initial = weights
        if initial is None and self._table is not None:
            initial = self._table.get_profile(profile, fallback=self._defaults)
        if initial is None:
            initial = self._defaults

        self._weights = normalize_weights(initial, defaults=self._defaults)
        if self._table is not None:
            self._table.update_profile(self._profile, self._weights, persist=False)

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    @property
    def profile(self) -> str:
        return self._profile

    def set_weights(self, weights: Mapping[str, float], *, persist: Optional[bool] = None) -> Dict[str, float]:
        normalized = normalize_weights(weights, defaults=self._defaults)
        self._weights = dict(normalized)
        if self._table is not None:
            should_persist = self._persist_updates if persist is None else persist
            self._table.update_profile(self._profile, normalized, persist=should_persist)
        return dict(self._weights)

    def compose(
        self,
        components: Mapping[str, float] | None = None,
        **kwargs: float,
    ) -> StyleComposition:
        raw_components: Dict[str, float] = {}
        if components:
            raw_components.update({str(k): float(v) for k, v in components.items()})
        if kwargs:
            raw_components.update({str(k): float(v) for k, v in kwargs.items()})

        component_keys = set(self._weights.keys()) | set(raw_components.keys())
        normalized_components: Dict[str, float] = {}
        for key in component_keys:
            value = raw_components.get(key, 0.0)
            normalized_components[str(key)] = max(0.0, min(1.0, float(value)))

        contributions: Dict[str, float] = {}
        for key, weight in self._weights.items():
            contributions[key] = normalized_components.get(key, 0.0) * weight

        total = sum(contributions.values())
        total = max(0.0, min(1.0, total))
        return StyleComposition(
            total=total,
            components=normalized_components,
            contributions=contributions,
            weights=dict(self._weights),
        )
