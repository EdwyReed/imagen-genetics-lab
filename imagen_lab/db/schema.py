from __future__ import annotations

import sqlite3
from typing import Iterable


MIGRATIONS: list[Iterable[str]] = [
    (
        """
        CREATE TABLE IF NOT EXISTS runs (
            session_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            mode TEXT NOT NULL,
            cfg_json TEXT NOT NULL,
            macro_snapshot TEXT,
            meso_snapshot TEXT,
            profile_id TEXT,
            seed INTEGER,
            conflicts TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS prompts (
            path TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            params_json TEXT NOT NULL,
            gene_choices_json TEXT,
            option_probs_json TEXT,
            caption TEXT,
            imagen_version TEXT,
            fitness REAL NOT NULL,
            parents TEXT,
            op TEXT,
            gen INTEGER,
            indiv INTEGER,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            status TEXT NOT NULL DEFAULT 'ok',
            FOREIGN KEY(session_id) REFERENCES runs(session_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scores (
            prompt_path TEXT PRIMARY KEY,
            micro_json TEXT NOT NULL,
            meso_json TEXT NOT NULL,
            fitness_visual REAL,
            fitness_body_focus REAL,
            fitness_alignment REAL,
            fitness_cleanliness REAL,
            fitness_era_match REAL,
            fitness_novelty REAL,
            clip_alignment REAL,
            ai_artifacts REAL,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            FOREIGN KEY(prompt_path) REFERENCES prompts(path) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS gene_stats (
            gene_id TEXT PRIMARY KEY,
            ema_fitness REAL NOT NULL,
            count INTEGER NOT NULL,
            last_seen_ts INTEGER NOT NULL,
            confidence REAL,
            updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_prompts_session_id ON prompts(session_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_prompts_fitness ON prompts(fitness)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_gene_stats_gene_id ON gene_stats(gene_id)
        """,
    ),
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply the static set of DDL statements to the provided connection."""

    cursor = conn.cursor()
    for migration in MIGRATIONS:
        for statement in migration:
            cursor.execute(statement)
    conn.commit()

