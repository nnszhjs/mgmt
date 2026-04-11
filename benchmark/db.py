# -*- coding: utf-8 -*-
"""
benchmark.db
=============

Thin SQLite wrapper for storing and querying benchmark results.

Schema
------
``runs`` – one row per (dataset, model, round, seed) experiment.

``metrics`` – one row per metric value, foreign-keyed to ``runs``.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset     TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    round       INTEGER NOT NULL,
    seed        INTEGER NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending',   -- pending | running | done | failed
    started_at  TEXT,
    finished_at TEXT,
    error_msg   TEXT,
    UNIQUE(dataset, model, round, seed)
);

CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    phase   TEXT    NOT NULL,   -- 'valid' or 'test'
    name    TEXT    NOT NULL,
    value   REAL    NOT NULL,
    UNIQUE(run_id, phase, name)
);
"""


class BenchmarkDB:
    """Manage a SQLite database for benchmark results."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def get_or_create_run(
        self, dataset: str, model: str, round_idx: int, seed: int
    ) -> int:
        """Return the run id, creating the row if it does not exist."""
        cur = self.conn.execute(
            "SELECT id FROM runs WHERE dataset=? AND model=? AND round=? AND seed=?",
            (dataset, model, round_idx, seed),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO runs(dataset, model, round, seed) VALUES(?,?,?,?)",
            (dataset, model, round_idx, seed),
        )
        self.conn.commit()
        return cur.lastrowid

    def mark_running(self, run_id: int):
        self.conn.execute(
            "UPDATE runs SET status='running', started_at=? WHERE id=?",
            (_now(), run_id),
        )
        self.conn.commit()

    def mark_done(
        self,
        run_id: int,
        valid_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ):
        """Record metrics and mark the run as done."""
        for name, value in valid_metrics.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO metrics(run_id, phase, name, value) "
                "VALUES(?,?,?,?)",
                (run_id, "valid", name, value),
            )
        for name, value in test_metrics.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO metrics(run_id, phase, name, value) "
                "VALUES(?,?,?,?)",
                (run_id, "test", name, value),
            )
        self.conn.execute(
            "UPDATE runs SET status='done', finished_at=? WHERE id=?",
            (_now(), run_id),
        )
        self.conn.commit()

    def mark_failed(self, run_id: int, error_msg: str):
        self.conn.execute(
            "UPDATE runs SET status='failed', finished_at=?, error_msg=? WHERE id=?",
            (_now(), error_msg, run_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def run_status(
        self, dataset: str, model: str, round_idx: int, seed: int
    ) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT status FROM runs "
            "WHERE dataset=? AND model=? AND round=? AND seed=?",
            (dataset, model, round_idx, seed),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def is_done(
        self, dataset: str, model: str, round_idx: int, seed: int
    ) -> bool:
        return self.run_status(dataset, model, round_idx, seed) == "done"

    def get_metrics(
        self,
        dataset: str,
        model: str,
        phase: str = "test",
        round_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int, str, float]]:
        """Return ``(round, seed, metric_name, metric_value)`` rows.

        Optionally filter by round and/or seed.
        """
        query = (
            "SELECT r.round, r.seed, m.name, m.value "
            "FROM metrics m JOIN runs r ON m.run_id = r.id "
            "WHERE r.dataset=? AND r.model=? AND m.phase=? AND r.status='done'"
        )
        params: list = [dataset, model, phase]
        if round_idx is not None:
            query += " AND r.round=?"
            params.append(round_idx)
        if seed is not None:
            query += " AND r.seed=?"
            params.append(seed)
        query += " ORDER BY r.round, r.seed, m.name"
        return self.conn.execute(query, params).fetchall()

    def all_done_models(self, dataset: str) -> List[str]:
        """Return model names that have at least one 'done' run on *dataset*."""
        cur = self.conn.execute(
            "SELECT DISTINCT model FROM runs WHERE dataset=? AND status='done'",
            (dataset,),
        )
        return [row[0] for row in cur]

    def all_done_datasets(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT DISTINCT dataset FROM runs WHERE status='done'"
        )
        return [row[0] for row in cur]

    def summary(self) -> str:
        """Human-readable summary of the database."""
        cur = self.conn.execute(
            "SELECT dataset, model, status, COUNT(*) "
            "FROM runs GROUP BY dataset, model, status ORDER BY dataset, model"
        )
        lines = ["dataset      | model         | status  | count"]
        lines.append("-" * 56)
        for ds, mdl, st, cnt in cur:
            lines.append(f"{ds:<12} | {mdl:<13} | {st:<7} | {cnt}")
        return "\n".join(lines)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
