from __future__ import annotations

from typing import Any, Protocol


class GAEngineProtocol(Protocol):
    def run(self, context: Any) -> None:
        """Execute the evolutionary loop with the provided context object."""
