from __future__ import annotations

from dataclasses import dataclass

from imagen_lab.storage import PromptLogger

from .interfaces import RepositoryProtocol, RunRecord


@dataclass
class PromptRepository(RepositoryProtocol):
    logger: PromptLogger

    def log_run(self, record: RunRecord) -> None:
        self.logger.log_run(record.session_id, record.mode, dict(record.payload))

    def log_message(self, session_id: str, message: str) -> None:
        self.logger.log_message(session_id, message)
