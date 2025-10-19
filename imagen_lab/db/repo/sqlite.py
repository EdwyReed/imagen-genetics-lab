from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

from ..schema import apply_migrations
from .interfaces import (
    GeneStatRecord,
    ProfileRecord,
    PromptRecord,
    RepositoryProtocol,
    RunRecord,
    ScoreRecord,
)


def _json_dumps(payload: Mapping[str, Any] | Sequence[Any] | None) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, MutableMapping):
        payload = dict(payload)
    return json.dumps(payload, ensure_ascii=False)


def _json_loads(raw: str | None) -> Any:
    if raw is None:
        return None
    return json.loads(raw)


@dataclass
class SQLiteRepository(RepositoryProtocol):
    """Repository that persists generation data into an SQLite database."""

    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            apply_migrations(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Run records
    def log_run(self, record: RunRecord) -> None:
        payload_json = _json_dumps(record.payload)
        macro_json = _json_dumps(record.macro_snapshot)
        meso_json = _json_dumps(record.meso_snapshot)
        conflicts_json = _json_dumps(record.conflicts)
        created_at = int(record.created_at or time.time())
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs(
                    session_id, created_at, mode, cfg_json,
                    macro_snapshot, meso_snapshot, profile_id, seed, conflicts
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.session_id,
                    created_at,
                    record.mode,
                    payload_json,
                    macro_json,
                    meso_json,
                    record.profile_id,
                    record.seed,
                    conflicts_json,
                ),
            )

    def get_run(self, session_id: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT session_id, created_at, mode, cfg_json, macro_snapshot, meso_snapshot, profile_id, seed, conflicts "
                "FROM runs WHERE session_id=?",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "created_at": row[1],
            "mode": row[2],
            "cfg": _json_loads(row[3]),
            "macro_snapshot": _json_loads(row[4]),
            "meso_snapshot": _json_loads(row[5]),
            "profile_id": row[6],
            "seed": row[7],
            "conflicts": _json_loads(row[8]),
        }

    # ------------------------------------------------------------------
    # Prompts and scores
    def log_prompt(self, record: PromptRecord) -> None:
        fitness = 0.0 if record.status == "no_image" else float(record.fitness)
        created_at = int(record.created_at or time.time())
        params_json = _json_dumps(record.params) or "{}"
        gene_json = _json_dumps(record.gene_choices)
        probs_json = _json_dumps(record.option_probabilities)
        parents_json = _json_dumps(record.parents)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prompts(
                    path, session_id, prompt, params_json, gene_choices_json,
                    option_probs_json, caption, imagen_version, fitness, parents,
                    op, gen, indiv, created_at, status
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.path,
                    record.session_id,
                    record.prompt,
                    params_json,
                    gene_json,
                    probs_json,
                    record.caption,
                    record.imagen_version,
                    fitness,
                    parents_json,
                    record.op,
                    record.gen,
                    record.indiv,
                    created_at,
                    record.status,
                ),
            )

    def get_prompt(self, path: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT path, session_id, prompt, params_json, gene_choices_json,
                       option_probs_json, caption, imagen_version, fitness, parents,
                       op, gen, indiv, created_at, status
                FROM prompts WHERE path=?
                """,
                (path,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "path": row[0],
            "session_id": row[1],
            "prompt": row[2],
            "params": _json_loads(row[3]) or {},
            "gene_choices": _json_loads(row[4]),
            "option_probs": _json_loads(row[5]),
            "caption": row[6],
            "imagen_version": row[7],
            "fitness": row[8],
            "parents": _json_loads(row[9]),
            "op": row[10],
            "gen": row[11],
            "indiv": row[12],
            "created_at": row[13],
            "status": row[14],
        }

    def delete_prompt(self, path: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM prompts WHERE path=?", (path,))

    def log_score(self, record: ScoreRecord) -> None:
        created_at = int(record.created_at or time.time())
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scores(
                    prompt_path, micro_json, meso_json, fitness_visual,
                    fitness_body_focus, fitness_alignment, fitness_cleanliness,
                    fitness_era_match, fitness_novelty, clip_alignment, ai_artifacts,
                    created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.prompt_path,
                    _json_dumps(record.micro_metrics) or "{}",
                    _json_dumps(record.meso_metrics) or "{}",
                    record.fitness_visual,
                    record.fitness_body_focus,
                    record.fitness_alignment,
                    record.fitness_cleanliness,
                    record.fitness_era_match,
                    record.fitness_novelty,
                    record.clip_alignment,
                    record.ai_artifacts,
                    created_at,
                ),
            )

    def get_score(self, prompt_path: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT prompt_path, micro_json, meso_json, fitness_visual,
                       fitness_body_focus, fitness_alignment, fitness_cleanliness,
                       fitness_era_match, fitness_novelty, clip_alignment,
                       ai_artifacts, created_at
                FROM scores WHERE prompt_path=?
                """,
                (prompt_path,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "prompt_path": row[0],
            "micro": _json_loads(row[1]) or {},
            "meso": _json_loads(row[2]) or {},
            "fitness_visual": row[3],
            "fitness_body_focus": row[4],
            "fitness_alignment": row[5],
            "fitness_cleanliness": row[6],
            "fitness_era_match": row[7],
            "fitness_novelty": row[8],
            "clip_alignment": row[9],
            "ai_artifacts": row[10],
            "created_at": row[11],
        }

    def record_cycle(
        self,
        *,
        prompt: PromptRecord,
        score: ScoreRecord | None = None,
    ) -> None:
        with self.transaction() as conn:
            self._insert_prompt(conn, prompt)
            if score is not None:
                self._insert_score(conn, score)

    def _insert_prompt(self, conn: sqlite3.Connection, record: PromptRecord) -> None:
        fitness = 0.0 if record.status == "no_image" else float(record.fitness)
        created_at = int(record.created_at or time.time())
        params_json = _json_dumps(record.params) or "{}"
        gene_json = _json_dumps(record.gene_choices)
        probs_json = _json_dumps(record.option_probabilities)
        parents_json = _json_dumps(record.parents)
        conn.execute(
            """
            INSERT OR REPLACE INTO prompts(
                path, session_id, prompt, params_json, gene_choices_json,
                option_probs_json, caption, imagen_version, fitness, parents,
                op, gen, indiv, created_at, status
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.path,
                record.session_id,
                record.prompt,
                params_json,
                gene_json,
                probs_json,
                record.caption,
                record.imagen_version,
                fitness,
                parents_json,
                record.op,
                record.gen,
                record.indiv,
                created_at,
                record.status,
            ),
        )

    def _insert_score(self, conn: sqlite3.Connection, record: ScoreRecord) -> None:
        created_at = int(record.created_at or time.time())
        conn.execute(
            """
            INSERT OR REPLACE INTO scores(
                prompt_path, micro_json, meso_json, fitness_visual,
                fitness_body_focus, fitness_alignment, fitness_cleanliness,
                fitness_era_match, fitness_novelty, clip_alignment, ai_artifacts,
                created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.prompt_path,
                _json_dumps(record.micro_metrics) or "{}",
                _json_dumps(record.meso_metrics) or "{}",
                record.fitness_visual,
                record.fitness_body_focus,
                record.fitness_alignment,
                record.fitness_cleanliness,
                record.fitness_era_match,
                record.fitness_novelty,
                record.clip_alignment,
                record.ai_artifacts,
                created_at,
            ),
        )

    # ------------------------------------------------------------------
    # Profiles
    def upsert_profile(self, record: ProfileRecord) -> None:
        updated_at = int(record.updated_at or time.time())
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO profiles(profile_id, vector_json, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    vector_json=excluded.vector_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.profile_id,
                    _json_dumps(record.vector) or "{}",
                    updated_at,
                ),
            )

    def get_profile(self, profile_id: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT profile_id, vector_json, updated_at FROM profiles WHERE profile_id=?",
                (profile_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "profile_id": row[0],
            "vector": _json_loads(row[1]) or {},
            "updated_at": row[2],
        }

    # ------------------------------------------------------------------
    # Gene statistics
    def upsert_gene_stats(self, records: Iterable[GeneStatRecord]) -> None:
        rows = [
            (
                rec.gene_id,
                float(rec.ema_fitness),
                int(rec.count),
                int(rec.last_seen_ts),
                rec.confidence,
                int(rec.updated_at or time.time()),
            )
            for rec in records
        ]
        if not rows:
            return
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO gene_stats(gene_id, ema_fitness, count, last_seen_ts, confidence, updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(gene_id) DO UPDATE SET
                    ema_fitness=excluded.ema_fitness,
                    count=excluded.count,
                    last_seen_ts=excluded.last_seen_ts,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def get_gene_stat(self, gene_id: str) -> Mapping[str, Any] | None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT gene_id, ema_fitness, count, last_seen_ts, confidence, updated_at
                FROM gene_stats WHERE gene_id=?
                """,
                (gene_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "gene_id": row[0],
            "ema_fitness": row[1],
            "count": row[2],
            "last_seen_ts": row[3],
            "confidence": row[4],
            "updated_at": row[5],
        }

    def iter_gene_stats(self) -> Iterable[Mapping[str, Any]]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT gene_id, ema_fitness, count, last_seen_ts, confidence, updated_at
                FROM gene_stats
                ORDER BY updated_at DESC
                """
            )
            rows = cursor.fetchall()
        for row in rows:
            yield {
                "gene_id": row[0],
                "ema_fitness": row[1],
                "count": row[2],
                "last_seen_ts": row[3],
                "confidence": row[4],
                "updated_at": row[5],
            }

