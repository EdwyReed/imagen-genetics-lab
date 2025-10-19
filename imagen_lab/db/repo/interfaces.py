from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class RunRecord:
    session_id: str
    mode: str
    payload: Mapping[str, Any]


class RepositoryProtocol(Protocol):
    def log_run(self, record: RunRecord) -> None:
        ...

    def log_message(self, session_id: str, message: str) -> None:
        ...
