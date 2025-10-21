from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

try:  # pragma: no cover - optional dependency detection
    from rich.console import Console
    from rich.theme import Theme
except ModuleNotFoundError:  # pragma: no cover - fallback when rich is unavailable
    Console = None  # type: ignore
    Theme = None  # type: ignore

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


def _normalize_level(level: str) -> str:
    return level.upper().strip()


def _should_emit(configured: str, requested: str) -> bool:
    return _LOG_LEVELS.get(requested, 100) >= _LOG_LEVELS.get(configured, 20)


@dataclass(slots=True)
class RunLogger:
    console: Optional[Console]
    level: str = "INFO"
    logfile: Optional[Path] = None
    _plain_file: Optional[object] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.level = _normalize_level(self.level)
        if self.logfile:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)
            self._plain_file = self.logfile.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._plain_file:
            self._plain_file.close()
            self._plain_file = None

    def log(
        self,
        step: str,
        message: str,
        level: str = "INFO",
        elapsed_ms: Optional[float] = None,
    ) -> None:
        level = _normalize_level(level)
        if not _should_emit(self.level, level):
            return
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        step_fmt = step.upper().ljust(7)
        level_fmt = level.ljust(5)
        suffix = f" (ms={elapsed_ms:.0f})" if elapsed_ms is not None else ""
        line = f"[{now}] [{level_fmt}] [{step_fmt}] {message}{suffix}"
        style = {
            "DEBUG": "dim",
            "INFO": "white",
            "WARN": "yellow",
            "ERROR": "red",
        }.get(level, "white")
        if self.console is not None:
            self.console.print(line, style=style, highlight=False, soft_wrap=True)
        else:  # pragma: no cover - console fallback
            print(line)
        if self._plain_file:
            self._plain_file.write(line + "\n")
            self._plain_file.flush()

    def timed(
        self,
        step: str,
        message: Union[str, Callable[[object], str]],
        func: Callable[..., object],
        *args,
        level: str = "INFO",
        **kwargs,
    ) -> object:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.log(step, f"error: {exc}", level="ERROR", elapsed_ms=elapsed)
            raise
        elapsed = (time.perf_counter() - start) * 1000.0
        if callable(message):
            try:
                msg = message(result)
            except Exception:
                msg = "<failed to render message>"
        else:
            msg = message
        self.log(step, msg, level=level, elapsed_ms=elapsed)
        return result


def create_logger(level: str, logfile: Optional[Path]) -> RunLogger:
    console: Optional[Console] = None
    if Console is not None and Theme is not None:
        console = Console(theme=Theme({"repr.number": "cyan"}))
    return RunLogger(console=console, level=level, logfile=logfile)


__all__ = ["RunLogger", "create_logger"]
