from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


class JSONRepositoryProtocol(ABC):
    """Contract for components loading JSON catalogs and presets."""

    @abstractmethod
    def load(self, path: Path) -> Mapping[str, Any]:
        """Return a parsed JSON mapping for the given path."""

    @abstractmethod
    def save(self, path: Path, payload: Mapping[str, Any]) -> None:
        """Persist the payload back to the filesystem."""
