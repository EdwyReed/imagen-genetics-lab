"""Persistence layer encapsulating SQLite operations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ..scene.builder import PrePrompt
from ..scoring.core import ScoreResult

__all__ = ["Repository", "RunRecord"]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config: Mapping[str, Any]
    profile_id: str | None


class Repository:
    """High-level interface for persisting runs, prompts, and scores."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._initialise()

    def _initialise(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT NOT NULL,
                profile_id TEXT
            );
            CREATE TABLE IF NOT EXISTS prompts (
                prompt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                caption TEXT NOT NULL,
                preprompt_json TEXT NOT NULL,
                genes_json TEXT NOT NULL,
                macro_json TEXT NOT NULL,
                meso_json TEXT NOT NULL,
                image_path TEXT NOT NULL,
                fitness REAL,
                generation INTEGER,
                individual INTEGER,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS scores (
                prompt_id TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY(prompt_id, metric),
                FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
            );
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                stability_score REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS gene_stats (
                category TEXT NOT NULL,
                gene_id TEXT NOT NULL,
                total_fitness REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                average_fitness REAL NOT NULL,
                PRIMARY KEY(category, gene_id)
            );
            """
        )
        self._conn.commit()

    def record_run(self, run_id: str, config: Mapping[str, Any], profile_id: Optional[str]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, config_json, profile_id) VALUES (?, ?, ?)",
            (run_id, json.dumps(config), profile_id),
        )
        self._conn.commit()

    def record_prompt(
        self,
        prompt_id: str,
        run_id: str,
        pre_prompt: PrePrompt,
        caption: str,
        image_path: Path,
        score: Optional[ScoreResult],
        generation: int = 0,
        individual: int = 0,
    ) -> None:
        fitness = score.fitness.get("fitness_visual", 0.0) if score else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO prompts(
                prompt_id, run_id, caption, preprompt_json, genes_json, macro_json, meso_json, image_path, fitness, generation, individual
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt_id,
                run_id,
                caption,
                json.dumps(_preprompt_payload(pre_prompt)),
                json.dumps(pre_prompt.selected_genes),
                json.dumps(dict(pre_prompt.macro_controls)),
                json.dumps(dict(pre_prompt.meso_signals)),
                str(image_path),
                fitness,
                generation,
                individual,
            ),
        )
        if score:
            self._store_scores(prompt_id, score)
        self._conn.commit()

    def _store_scores(self, prompt_id: str, score: ScoreResult) -> None:
        records: Iterable[tuple[str, str, float]] = (
            (prompt_id, metric, value)
            for metric_dict in (score.micro_metrics, score.meso_aggregates, score.fitness)
            for metric, value in metric_dict.items()
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO scores(prompt_id, metric, value) VALUES (?, ?, ?)",
            list(records),
        )

    def record_gene_statistics(
        self,
        selected_genes: Mapping[str, str],
        fitness_value: float,
    ) -> None:
        if not selected_genes:
            return
        rows = []
        for category, gene_id in selected_genes.items():
            if not gene_id:
                continue
            rows.append((category, gene_id, float(fitness_value)))
        if not rows:
            return
        cur = self._conn.cursor()
        for category, gene_id, value in rows:
            cur.execute(
                "SELECT total_fitness, sample_count FROM gene_stats WHERE category = ? AND gene_id = ?",
                (category, gene_id),
            )
            row = cur.fetchone()
            if row:
                total = float(row[0]) + value
                count = int(row[1]) + 1
            else:
                total = value
                count = 1
            average = total / count
            cur.execute(
                """
                INSERT OR REPLACE INTO gene_stats(category, gene_id, total_fitness, sample_count, average_fitness)
                VALUES(?, ?, ?, ?, ?)
                """,
                (category, gene_id, total, count, average),
            )
        self._conn.commit()

    def load_gene_biases(self) -> Dict[str, Dict[str, float]]:
        cur = self._conn.execute(
            "SELECT category, gene_id, average_fitness FROM gene_stats"
        )
        biases: Dict[str, Dict[str, float]] = {}
        for category, gene_id, average in cur.fetchall():
            biases.setdefault(str(category), {})[str(gene_id)] = float(average)
        return biases

    def load_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT vector_json FROM profiles WHERE profile_id = ?",
            (profile_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def save_profile(self, profile_id: str, vector: Mapping[str, Any], stability: float = 0.0) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO profiles(profile_id, vector_json, stability_score) VALUES (?, ?, ?)",
            (profile_id, json.dumps(dict(vector)), stability),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _preprompt_payload(pre_prompt: PrePrompt) -> Dict[str, Any]:
    return {
        "style_id": pre_prompt.style_id,
        "character_id": pre_prompt.character_id,
        "scene_summary": pre_prompt.scene_summary,
        "visual_axes": dict(pre_prompt.visual_axes),
    }
