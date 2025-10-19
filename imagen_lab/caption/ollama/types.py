from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    final_prompt: str
    enforced: bool
    bounds: Mapping[str, object]
    system_hash: str
