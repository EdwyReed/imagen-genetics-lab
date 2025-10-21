from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for Python <3.11
    import tomli as tomllib  # type: ignore


@dataclass(slots=True)
class RunConfig:
    count: int = 1
    seed: Optional[int] = None
    out_dir: Path = Path("out")


@dataclass(slots=True)
class NamingConfig:
    zero_pad: int = 6


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    to_file: bool = True


@dataclass(slots=True)
class SelectionConfig:
    character_ids: List[str] = field(default_factory=list)
    pose_ids: List[str] = field(default_factory=list)
    action_ids: List[str] = field(default_factory=list)
    style_ids: List[str] = field(default_factory=list)
    clothes_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3:8b-instruct"
    system_prompt_path: Path = Path("conf/ollama_system_prompt_caption.txt")
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: int = 30


@dataclass(slots=True)
class EnforceConfig:
    enabled: bool = True
    required_terms: List[str] = field(default_factory=list)
    max_rewrite: int = 1


@dataclass(slots=True)
class ImagenConfig:
    model: str = "imagen-3.0-generate-002"
    variants: int = 1
    person_mode: str = "allow_adult"
    resolution: str = "1024x1024"
    safety_filter: str = "block_low_and_above"


@dataclass(slots=True)
class ScoringConfig:
    enabled: bool = False


@dataclass(slots=True)
class CLIConfig:
    default_session_prefix: str = "session"


@dataclass(slots=True)
class Config:
    path: Path
    run: RunConfig
    naming: NamingConfig
    logging: LoggingConfig
    select: SelectionConfig
    ollama: OllamaConfig
    enforce: EnforceConfig
    imagen: ImagenConfig
    scoring: ScoringConfig
    cli: CLIConfig

    def resolve_paths(self) -> None:
        base = self.path.parent
        project_root = base.parent if base.parent != base else base

        out_path = Path(self.run.out_dir)
        if not out_path.is_absolute():
            self.run.out_dir = (project_root / out_path).resolve()
        else:
            self.run.out_dir = out_path

        prompt_path = Path(self.ollama.system_prompt_path)
        if prompt_path.is_absolute():
            self.ollama.system_prompt_path = prompt_path
        else:
            candidate = (project_root / prompt_path).resolve()
            if candidate.exists():
                self.ollama.system_prompt_path = candidate
            else:
                self.ollama.system_prompt_path = (base / prompt_path).resolve()

    def apply_overrides(
        self,
        count: Optional[int] = None,
        seed: Optional[int] = None,
        out_dir: Optional[Path] = None,
    ) -> None:
        if count is not None:
            self.run.count = int(count)
        if seed is not None:
            self.run.seed = int(seed)
        if out_dir is not None:
            self.run.out_dir = out_dir.resolve()


def _section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    return dict(raw.get(key, {}))


def _load_config_dict(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: Path) -> Config:
    raw = _load_config_dict(path)
    cfg = Config(
        path=path.resolve(),
        run=RunConfig(**_section(raw, "run")),
        naming=NamingConfig(**_section(raw, "naming")),
        logging=LoggingConfig(**_section(raw, "logging")),
        select=SelectionConfig(**_section(raw, "select")),
        ollama=OllamaConfig(**_section(raw, "ollama")),
        enforce=EnforceConfig(**_section(raw, "enforce")),
        imagen=ImagenConfig(**_section(raw, "imagen")),
        scoring=ScoringConfig(**_section(raw, "scoring")),
        cli=CLIConfig(**_section(raw, "cli")),
    )
    cfg.resolve_paths()
    return cfg


__all__ = [
    "Config",
    "RunConfig",
    "NamingConfig",
    "LoggingConfig",
    "SelectionConfig",
    "OllamaConfig",
    "EnforceConfig",
    "ImagenConfig",
    "ScoringConfig",
    "CLIConfig",
    "load_config",
]
