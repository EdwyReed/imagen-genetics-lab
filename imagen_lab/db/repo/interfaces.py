from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RunRecord:
    session_id: str
    mode: str
    payload: Mapping[str, Any]
    macro_snapshot: Mapping[str, Any] | None = None
    meso_snapshot: Mapping[str, Any] | None = None
    profile_id: str | None = None
    seed: int | None = None
    conflicts: Mapping[str, Any] | Sequence[str] | None = None
    created_at: int | None = None


@dataclass(frozen=True)
class PromptRecord:
    path: str
    session_id: str
    prompt: str
    params: Mapping[str, Any]
    gene_choices: Mapping[str, Any] | None = None
    option_probabilities: Mapping[str, Any] | None = None
    caption: str | None = None
    imagen_version: str | None = None
    fitness: float = 0.0
    parents: Sequence[str] | None = None
    op: str | None = None
    gen: int | None = None
    indiv: int | None = None
    created_at: int | None = None
    status: str = "ok"


@dataclass(frozen=True)
class ScoreRecord:
    prompt_path: str
    micro_metrics: Mapping[str, Any]
    meso_metrics: Mapping[str, Any]
    fitness_visual: float | None = None
    fitness_body_focus: float | None = None
    fitness_alignment: float | None = None
    fitness_cleanliness: float | None = None
    fitness_era_match: float | None = None
    fitness_novelty: float | None = None
    clip_alignment: float | None = None
    ai_artifacts: float | None = None
    created_at: int | None = None


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    vector: Mapping[str, Any]
    updated_at: int | None = None


@dataclass(frozen=True)
class GeneStatRecord:
    gene_id: str
    ema_fitness: float
    count: int
    last_seen_ts: int
    confidence: float | None = None
    updated_at: int | None = None


class RepositoryProtocol(Protocol):
    def log_run(self, record: RunRecord) -> None:
        ...

    def get_run(self, session_id: str) -> Mapping[str, Any] | None:
        ...

    def log_prompt(self, record: PromptRecord) -> None:
        ...

    def get_prompt(self, path: str) -> Mapping[str, Any] | None:
        ...

    def delete_prompt(self, path: str) -> None:
        ...

    def log_score(self, record: ScoreRecord) -> None:
        ...

    def get_score(self, prompt_path: str) -> Mapping[str, Any] | None:
        ...

    def record_cycle(self, *, prompt: PromptRecord, score: ScoreRecord | None = None) -> None:
        ...

    def upsert_profile(self, record: ProfileRecord) -> None:
        ...

    def get_profile(self, profile_id: str) -> Mapping[str, Any] | None:
        ...

    def upsert_gene_stats(self, records: Iterable[GeneStatRecord]) -> None:
        ...

    def get_gene_stat(self, gene_id: str) -> Mapping[str, Any] | None:
        ...

    def iter_gene_stats(self) -> Iterable[Mapping[str, Any]]:
        ...
