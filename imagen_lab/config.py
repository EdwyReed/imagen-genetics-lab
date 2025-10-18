from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_WEIGHT_PROFILE_PATH = Path(__file__).with_name("scoring").joinpath("weight_profiles.yaml")
DEFAULT_INLINE_WEIGHTS: Dict[str, float] = {"clip": 0.55, "spec": 0.35, "illu": 0.10}


@dataclass
class PathsConfig:
    catalog: Path
    database: Path
    scores_jsonl: Path
    output_dir: Path


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
class PipelineConfig:
    paths: PathsConfig
    prompting: PromptConfig
    ollama: OllamaConfig
    imagen: ImagenConfig
    scoring: ScoringConfig
    history: HistoryConfig
    fitness: FitnessWeights
    defaults: RunDefaults
    ga: GAConfig

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        paths_data = raw.get("paths", {})
        prompting_data = raw.get("prompting", {})
        ollama_data = raw.get("ollama", {})
        imagen_data = raw.get("imagen", {})
        scoring_data = raw.get("scoring", {})
        history_data = raw.get("history", {})
        fitness_data = raw.get("fitness", {})
        defaults_data = raw.get("defaults", {})
        ga_data = raw.get("ga", {})

        paths = PathsConfig(
            catalog=Path(paths_data.get("catalog", "jelly-pin-up.json")),
            database=Path(paths_data.get("database", "scores.sqlite")),
            scores_jsonl=Path(paths_data.get("scores_jsonl", "scores.jsonl")),
            output_dir=Path(paths_data.get("output_dir", "output")),
        )

        prompting = PromptConfig(
            required_terms=list(prompting_data.get("required_terms", [])),
            template_ids=list(prompting_data.get("template_ids", ["caption_v1", "caption_v2", "caption_v3", "caption_v4"])),
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

        weight_profiles_raw = scoring_data.get("weight_profiles_path", scoring_data.get("weight_profiles"))
        if "weight_profiles_path" in scoring_data or "weight_profiles" in scoring_data:
            if weight_profiles_raw in (None, "", False):
                weight_profiles_path = None
            else:
                weight_profiles_path = Path(str(weight_profiles_raw))
        else:
            weight_profiles_path = DEFAULT_WEIGHT_PROFILE_PATH

        weights_value = scoring_data.get("weights")
        weights = dict(weights_value) if isinstance(weights_value, dict) else None
        if weights is None and weights_value not in (None, "", False):
            weights = dict(DEFAULT_INLINE_WEIGHTS)

        scoring = ScoringConfig(
            device=str(scoring_data.get("device", "auto")),
            batch_size=int(scoring_data.get("batch_size", 4)),
            tau=float(scoring_data.get("tau", 0.07)),
            cal_style=_tuple_or_none(scoring_data.get("cal_style")),
            cal_illu=_tuple_or_none(scoring_data.get("cal_illu")),
            weights=weights,
            auto_weights=AutoWeightsConfig.from_mapping(scoring_data.get("auto_weights")),
            weight_profiles_path=weight_profiles_path,
            weight_profile=str(scoring_data.get("weight_profile", "default")),
            persist_profile_updates=bool(scoring_data.get("persist_profile_updates", False)),
            composition_metrics=bool(scoring_data.get("composition_metrics", True)),
        )

        history = HistoryConfig(
            enabled=bool(history_data.get("enabled", True)),
            max_embeddings=int(history_data.get("max_embeddings", 512)),
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
            pop=int(ga_data.get("pop", 16)),
            gens=int(ga_data.get("gens", 4)),
            keep=float(ga_data.get("keep", 0.25)),
            mut=float(ga_data.get("mut", 0.15)),
            xover=float(ga_data.get("xover", 0.30)),
            resume_best=bool(ga_data.get("resume_best", False)),
            resume_k=int(ga_data.get("resume_k", 0)),
            resume_session=ga_data.get("resume_session"),
            resume_mix=float(ga_data.get("resume_mix", 0.10)),
        )

        return cls(
            paths=paths,
            prompting=prompting,
            ollama=ollama,
            imagen=imagen,
            scoring=scoring,
            history=history,
            fitness=fitness,
            defaults=defaults,
            ga=ga,
        )


def _tuple_or_none(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def load_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level")
    return PipelineConfig.from_dict(data)
