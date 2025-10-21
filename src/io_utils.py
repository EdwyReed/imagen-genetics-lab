from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(slots=True)
class SessionPaths:
    session_id: str
    root: Path
    log_file: Path


@dataclass(slots=True)
class VariantArtifact:
    file_id: str
    image_path: Path
    json_path: Path
    txt_path: Path


def generate_session_id(prefix: str) -> str:
    return f"{prefix}-{datetime.utcnow():%Y%m%d-%H%M%S}"


def create_session(out_dir: Path, session_id: Optional[str], prefix: str = "session") -> SessionPaths:
    sid = session_id or generate_session_id(prefix)
    root = out_dir / sid
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "log.txt"
    return SessionPaths(session_id=sid, root=root, log_file=log_file)


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def hash_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def hash_dictionary_files(paths: Iterable[Path]) -> Dict[str, str]:
    return {p.name: hash_file(p) for p in paths}


def save_variant(
    base_id: int,
    session: SessionPaths,
    index: int,
    variant_number: int,
    caption: str,
    options: Dict[str, object],
    image_bytes: bytes,
    meta_payload: Dict[str, object],
    zero_pad: int,
    seed: Optional[int],
) -> VariantArtifact:
    stem = f"{base_id:0{zero_pad}d}_{variant_number}"
    image_path = session.root / f"{stem}.jpg"
    json_path = image_path.with_suffix(".json")
    txt_path = image_path.with_suffix(".txt")

    with image_path.open("wb") as fh:
        fh.write(image_bytes)

    payload = {
        "id": stem,
        "session_id": session.session_id,
        "seed": seed,
        "index": index,
        "variant": variant_number,
        "caption": caption,
        "options": options,
        "meta": meta_payload,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(caption.strip() + "\n")

    return VariantArtifact(stem, image_path, json_path, txt_path)


def ensure_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"Environment variable {var} is required")
    return value


__all__ = [
    "SessionPaths",
    "VariantArtifact",
    "create_session",
    "generate_session_id",
    "hash_text",
    "hash_file",
    "hash_dictionary_files",
    "save_variant",
    "ensure_env",
]
