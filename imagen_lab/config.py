from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import math
import json
import re
import yaml


DEFAULT_WEIGHT_PROFILE_PATH = Path(__file__).with_name("scoring").joinpath("weight_profiles.yaml")
DEFAULT_INLINE_WEIGHTS: Dict[str, float] = {
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


@dataclass
class PathsConfig:
    catalog: Path
    database: Path
    scores_jsonl: Path
    output_dir: Path
    options_catalog: Path | None = None
    character_catalog: Path | None = None
    profiles_dir: Path | None = None
    profile_template: Path | None = None


@dataclass
class PresetConfig:
    profile_id: str | None = None
    style_preset: str | None = None
    character_preset: str | None = None


@dataclass
class BiasConfig:
    macro_weights: Dict[str, float] = field(default_factory=dict)
    meso_aggregates: Dict[str, float] = field(default_factory=dict)
    rules: tuple[str, ...] = field(default_factory=tuple)

    def combined(self) -> Dict[str, float]:
        merged = dict(self.macro_weights)
        for key, value in self.meso_aggregates.items():
            merged[key] = value
        return merged


@dataclass
class PromptConfig:
    required_terms: list[str]
    template_ids: list[str]


@dataclass
class OllamaConfig:
    url: str
    model: str
    temperature: float
    top_p: float
    manual_mode: bool = True


@dataclass
class ImagenConfig:
    model: str
    person_mode: str
    guidance_scale: float = 0.5


@dataclass
class AutoWeightsConfig:
    enabled: bool = False
    ema_alpha: float = 0.25
    momentum: float = 0.35
    target: float = 0.85
    min_component: float = 0.05
    min_weight: float = 0.05
    max_weight: float = 0.9
    min_gain: float = 0.4
    max_gain: float = 2.5
    initial_level: float = 0.7

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any] | None) -> "AutoWeightsConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            ema_alpha=float(raw.get("ema_alpha", 0.25)),
            momentum=float(raw.get("momentum", 0.35)),
            target=float(raw.get("target", 0.85)),
            min_component=float(raw.get("min_component", 0.05)),
            min_weight=float(raw.get("min_weight", 0.05)),
            max_weight=float(raw.get("max_weight", 0.9)),
            min_gain=float(raw.get("min_gain", 0.4)),
            max_gain=float(raw.get("max_gain", 2.5)),
            initial_level=float(raw.get("initial_level", 0.7)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ema_alpha": self.ema_alpha,
            "momentum": self.momentum,
            "target": self.target,
            "min_component": self.min_component,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "min_gain": self.min_gain,
            "max_gain": self.max_gain,
            "initial_level": self.initial_level,
        }


@dataclass
class TemperatureConfig:
    """Temperature parameters for contrastive metric adjustments."""

    default: float = 0.07
    overrides: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default = float(self.default)
        cleaned: Dict[str, float] = {}
        for key, value in self.overrides.items():
            if key == "default":
                continue
            cleaned[str(key)] = float(value)
        self.overrides = cleaned

    @classmethod
    def from_mapping(
        cls, raw: Optional[Mapping[str, object] | float | "TemperatureConfig"]
    ) -> "TemperatureConfig":
        if raw is None:
            return cls()
        if isinstance(raw, TemperatureConfig):
            return cls(default=raw.default, overrides=dict(raw.overrides))
        if isinstance(raw, Mapping):
            default = float(raw.get("default", 0.07))
            overrides = {
                str(key): float(value)
                for key, value in raw.items()
                if key != "default"
            }
            return cls(default=default, overrides=overrides)
        return cls(default=float(raw))

    def resolve(self, metric: str) -> float:
        return float(self.overrides.get(metric, self.default))

    def apply(self, metric: str, value: float) -> float:
        tau = self.resolve(metric)
        return _apply_temperature(value, tau)

    def as_dict(self) -> Dict[str, float]:
        data = {"default": float(self.default)}
        data.update({key: float(val) for key, val in self.overrides.items()})
        return data


def _apply_temperature(value: float, tau: float) -> float:
    clamped = max(0.0, min(1.0, float(value)))
    if tau <= 0:
        return clamped
    strength = 1.0 + max(0.0, float(tau))
    if clamped >= 0.5:
        adjusted = 0.5 + (clamped - 0.5) * strength
        return min(1.0, adjusted)
    adjusted = 0.5 - (0.5 - clamped) / strength
    return max(0.0, adjusted)


@dataclass
class ScoringConfig:
    device: str = "auto"
    batch_size: int = 4
    tau: float = 0.07
    cal_style: Optional[tuple[float, float]] = None
    cal_illu: Optional[tuple[float, float]] = None
    weights: Dict[str, float] | None = field(default_factory=lambda: dict(DEFAULT_INLINE_WEIGHTS))
    auto_weights: "AutoWeightsConfig" = field(default_factory=lambda: AutoWeightsConfig())
    weight_profiles_path: Optional[Path] = DEFAULT_WEIGHT_PROFILE_PATH
    weight_profile: str = "default"
    persist_profile_updates: bool = False
    composition_metrics: bool = True
    temperatures: TemperatureConfig = field(default_factory=TemperatureConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.temperatures, TemperatureConfig):
            self.temperatures = TemperatureConfig.from_mapping(self.temperatures)
        if not math.isclose(self.tau, self.temperatures.default):
            self.tau = float(self.temperatures.default)


@dataclass
class HistoryConfig:
    enabled: bool = True
    max_embeddings: int = 512


@dataclass
class FitnessWeights:
    style: float = 0.7
    nsfw: float = 0.3


@dataclass
class RunDefaults:
    sfw_level: float = 0.6
    temperature: float = 0.55
    per_cycle: int = 2
    cycles: int = 10
    sleep_s: float = 1.0
    seed: Optional[int] = None


@dataclass
class GAConfig:
    pop: int = 16
    gens: int = 4
    keep: float = 0.25
    mut: float = 0.15
    xover: float = 0.30
    resume_best: bool = False
    resume_k: int = 0
    resume_session: Optional[str] = None
    resume_mix: float = 0.10


@dataclass
class FeedbackConfig:
    enabled: bool = True
    component_alpha: float = 0.25
    gene_alpha: float = 0.35
    bias_floor: float = 0.7
    bias_ceiling: float = 1.6
    component_margin: float = 0.12
    history: int = 6
    top_k: int = 3

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any] | None) -> "FeedbackConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            component_alpha=float(raw.get("component_alpha", 0.25)),
            gene_alpha=float(raw.get("gene_alpha", 0.35)),
            bias_floor=float(raw.get("bias_floor", 0.7)),
            bias_ceiling=float(raw.get("bias_ceiling", 1.6)),
            component_margin=float(raw.get("component_margin", 0.12)),
            history=int(raw.get("history", 6)),
            top_k=int(raw.get("top_k", 3)),
        )


@dataclass
class PipelineConfig:
    schema_version: int
    paths: PathsConfig
    presets: PresetConfig
    bias: BiasConfig
    prompting: PromptConfig
    ollama: OllamaConfig
    imagen: ImagenConfig
    scoring: ScoringConfig
    history: HistoryConfig
    fitness: FitnessWeights
    defaults: RunDefaults
    ga: GAConfig
    feedback: FeedbackConfig

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        schema_version = int(raw.get("schema_version", 2))

        storage_data = raw.get("storage", {})
        paths_data = raw.get("paths", {})
        prompting_data = _mapping_merge(
            raw.get("prompting", {}),
            _nested_mapping(raw.get("runtime", {}), "prompting"),
        )
        runtime_block = raw.get("runtime", {})
        ollama_data = runtime_block.get("ollama", raw.get("ollama", {}))
        imagen_data = runtime_block.get("imagen", raw.get("imagen", {}))
        scoring_data = raw.get("scoring", {})
        history_block = runtime_block.get("history", raw.get("history", {}))
        history_data = _nested_mapping(history_block, "embeddings") or history_block
        fitness_data = raw.get("fitness", {})
        defaults_data = runtime_block.get("defaults", raw.get("defaults", {}))
        ga_data = runtime_block.get("ga", runtime_block.get("genetic", raw.get("ga", {})))
        feedback_block = runtime_block.get("feedback", raw.get("feedback", {}))
        feedback_data = _nested_mapping(feedback_block, "style") or feedback_block
        presets_data = raw.get("presets", {})

        macro_weights = _float_map(raw.get("macro_weights"))
        meso_weights = _float_map(raw.get("meso_aggregates"))
        bias_rules = tuple(
            rule.strip()
            for rule in (raw.get("bias_rules") or [])
            if isinstance(rule, str) and rule.strip()
        )

        catalogs_data = storage_data.get("catalogs", {})
        profiles_data = storage_data.get("profiles", {})

        catalog_path = (
            catalogs_data.get("primary")
            or paths_data.get("catalog")
            or "catalogs/all-together.json"
        )
        options_catalog_value = (
            catalogs_data.get("options")
            or paths_data.get("options_catalog")
        )
        character_catalog_value = (
            catalogs_data.get("characters")
            or paths_data.get("character_catalog")
        )

        options_catalog = _optional_path(options_catalog_value)
        character_catalog = _optional_path(character_catalog_value)

        profiles_dir = _optional_path(
            profiles_data.get("directory") or paths_data.get("profiles_dir")
        )
        profile_template = _optional_path(
            profiles_data.get("active")
            or profiles_data.get("profile")
            or paths_data.get("profile_template")
        )

        database_path = storage_data.get("database", paths_data.get("database", "scores.sqlite"))
        scores_log = storage_data.get("scores_log", paths_data.get("scores_jsonl", "scores.jsonl"))
        output_dir = storage_data.get("artifacts_dir", paths_data.get("output_dir", "output"))

        paths = PathsConfig(
            catalog=Path(str(catalog_path)),
            database=Path(str(database_path)),
            scores_jsonl=Path(str(scores_log)),
            output_dir=Path(str(output_dir)),
            options_catalog=options_catalog,
            character_catalog=character_catalog,
            profiles_dir=profiles_dir,
            profile_template=profile_template,
        )

        prompting = PromptConfig(
            required_terms=list(prompting_data.get("required_terms", [])),
            template_ids=list(
                prompting_data.get(
                    "template_ids",
                    ["caption_v1", "caption_v2", "caption_v3", "caption_v4"],
                )
            ),
        )

        ollama = OllamaConfig(
            url=str(ollama_data.get("url", "http://localhost:11434")),
            model=str(ollama_data.get("model", "qwen2.5:3b")),
            temperature=float(ollama_data.get("temperature", 0.55)),
            top_p=float(ollama_data.get("top_p", 0.9)),
            manual_mode=bool(ollama_data.get("manual_mode", True)),
        )

        imagen = ImagenConfig(
            model=str(imagen_data.get("model", "imagen-3.0-generate-002")),
            person_mode=str(imagen_data.get("person_mode", "allow_adult")),
            guidance_scale=float(imagen_data.get("guidance_scale", 0.5)),
        )

        weights_section = scoring_data.get("weights", {}) if isinstance(scoring_data.get("weights"), Mapping) else {}
        inline_weights = (
            scoring_data.get("weights")
            if isinstance(scoring_data.get("weights"), Mapping) and not weights_section
            else weights_section.get("inline")
        )
        weights = dict(inline_weights) if isinstance(inline_weights, Mapping) else None
        if weights is None and inline_weights not in (None, "", False):
            weights = dict(DEFAULT_INLINE_WEIGHTS)

        profiles_section = weights_section.get("profiles", {}) if isinstance(weights_section, Mapping) else {}
        weight_profiles_specified = False
        weight_profiles_raw: Any
        if isinstance(profiles_section, Mapping) and "table" in profiles_section:
            weight_profiles_specified = True
            weight_profiles_raw = profiles_section.get("table")
        else:
            weight_profiles_raw = (
                scoring_data.get("weight_profiles_path")
                or scoring_data.get("weight_profiles")
            )
            if "weight_profiles_path" in scoring_data or "weight_profiles" in scoring_data:
                weight_profiles_specified = True

        if weight_profiles_raw in ("", False):
            weight_profiles_path = None
        elif weight_profiles_raw is None:
            weight_profiles_path = (
                None if weight_profiles_specified else DEFAULT_WEIGHT_PROFILE_PATH
            )
        else:
            weight_profiles_path = Path(str(weight_profiles_raw))

        temperature_source = (
            scoring_data.get("temperatures")
            or scoring_data.get("tau")
            or 0.07
        )
        temperature_cfg = TemperatureConfig.from_mapping(temperature_source)

        weight_profile_name = (
            profiles_section.get("profile")
            or scoring_data.get("weight_profile")
            or "default"
        )

        persist_updates = bool(
            profiles_section.get("persist_updates", scoring_data.get("persist_profile_updates", False))
        )

        scoring = ScoringConfig(
            device=str(scoring_data.get("device", "auto")),
            batch_size=int(scoring_data.get("batch_size", 4)),
            tau=float(temperature_cfg.default),
            cal_style=_tuple_or_none(
                scoring_data.get("cal_style")
                or _nested_tuple(scoring_data.get("calibration", {}), "style")
            ),
            cal_illu=_tuple_or_none(
                scoring_data.get("cal_illu")
                or _nested_tuple(scoring_data.get("calibration", {}), "illustration")
            ),
            weights=weights,
            auto_weights=AutoWeightsConfig.from_mapping(scoring_data.get("auto_weights")),
            weight_profiles_path=weight_profiles_path,
            weight_profile=str(weight_profile_name),
            persist_profile_updates=persist_updates,
            composition_metrics=bool(scoring_data.get("composition_metrics", True)),
            temperatures=temperature_cfg,
        )

        history = HistoryConfig(
            enabled=bool(history_data.get("enabled", True)),
            max_embeddings=int(history_data.get("max_embeddings", history_data.get("limit", 512))),
        )

        fitness = FitnessWeights(
            style=float(fitness_data.get("style", 0.7)),
            nsfw=float(fitness_data.get("nsfw", 0.3)),
        )

        defaults = RunDefaults(
            sfw_level=float(defaults_data.get("sfw_level", 0.6)),
            temperature=float(defaults_data.get("temperature", 0.55)),
            per_cycle=int(defaults_data.get("per_cycle", 2)),
            cycles=int(defaults_data.get("cycles", 10)),
            sleep_s=float(defaults_data.get("sleep_s", 1.0)),
            seed=defaults_data.get("seed"),
        )

        ga = GAConfig(
            pop=int(ga_data.get("pop", ga_data.get("population", 16))),
            gens=int(ga_data.get("gens", ga_data.get("generations", 4))),
            keep=float(ga_data.get("keep", ga_data.get("keep_fraction", 0.25))),
            mut=float(ga_data.get("mut", ga_data.get("mutation", 0.15))),
            xover=float(ga_data.get("xover", ga_data.get("crossover", 0.30))),
            resume_best=bool(ga_data.get("resume_best", False)),
            resume_k=int(ga_data.get("resume_k", ga_data.get("resume_top_k", 0))),
            resume_session=ga_data.get("resume_session"),
            resume_mix=float(ga_data.get("resume_mix", ga_data.get("resume_mix_ratio", 0.10))),
        )

        feedback = FeedbackConfig.from_mapping(feedback_data)

        presets = PresetConfig(
            profile_id=_optional_str(presets_data.get("profile_id")),
            style_preset=_optional_str(presets_data.get("style_preset")),
            character_preset=_optional_str(presets_data.get("character_preset")),
        )

        bias = BiasConfig(
            macro_weights=macro_weights,
            meso_aggregates=meso_weights,
            rules=bias_rules,
        )

        return cls(
            schema_version=schema_version,
            paths=paths,
            presets=presets,
            bias=bias,
            prompting=prompting,
            ollama=ollama,
            imagen=imagen,
            scoring=scoring,
            history=history,
            fitness=fitness,
            defaults=defaults,
            ga=ga,
            feedback=feedback,
        )


def _tuple_or_none(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def load_config(path: Path) -> PipelineConfig:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonc"}:
        data = json.loads(_strip_jsonc(text))
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level")
    return PipelineConfig.from_dict(data)


def _strip_jsonc(payload: str) -> str:
    result: list[str] = []
    length = len(payload)
    i = 0
    in_string = False
    escape = False
    while i < length:
        ch = payload[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue

        if ch == '/' and i + 1 < length:
            nxt = payload[i + 1]
            if nxt == '/':
                i += 2
                while i < length and payload[i] not in "\r\n":
                    i += 1
                continue
            if nxt == '*':
                i += 2
                while i < length - 1:
                    if payload[i] == '*' and payload[i + 1] == '/':
                        i += 2
                        break
                    i += 1
                continue

        result.append(ch)
        i += 1
    return "".join(result)


def _optional_path(value: Any) -> Path | None:
    if value in (None, "", False):
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _mapping_merge(parent: Mapping[str, Any], child: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(parent, Mapping):
        result.update(parent)
    if isinstance(child, Mapping):
        result.update(child)
    return result


def _nested_mapping(source: Any, key: str) -> Dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    value = source.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_tuple(source: Any, key: str) -> Optional[tuple[float, float]]:
    if not isinstance(source, Mapping):
        return None
    value = source.get(key)
    return _tuple_or_none(value)


def _float_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result
