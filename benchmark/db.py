# -*- coding: utf-8 -*-
"""
benchmark.db
=============

Thin SQLite wrapper for storing and querying benchmark results.

Schema
------
``runs`` – one row per (dataset, model, benchmark_rounds, train_splits,
round_idx, seed) experiment.

``metrics`` – one row per metric value, foreign-keyed to ``runs``.
"""

import os
import sqlite3
from datetime import datetime
from logging import getLogger
from typing import Dict, List, Optional, Tuple

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    window_idx        INTEGER NOT NULL,
    window_size       REAL,
    window_start      REAL,
    window_end        REAL,
    seed              INTEGER NOT NULL,
    status            TEXT    NOT NULL DEFAULT 'pending',   -- pending | running | done | failed
    started_at        TEXT,
    finished_at       TEXT,
    error_msg         TEXT,

    -- Legacy fields for backward compatibility
    benchmark_rounds  INTEGER,
    train_splits      INTEGER,
    round_idx         INTEGER,

    UNIQUE(dataset, model, window_idx, seed)
);

CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    phase   TEXT    NOT NULL,   -- 'valid' or 'test'
    name    TEXT    NOT NULL,
    value   REAL    NOT NULL,
    UNIQUE(run_id, phase, name)
);

CREATE TABLE IF NOT EXISTS losses (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    epoch   INTEGER NOT NULL,
    loss    REAL    NOT NULL,
    UNIQUE(run_id, epoch)
);
"""

# Migrations for databases created by previous schema versions.
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN benchmark_rounds INTEGER",
    "ALTER TABLE runs ADD COLUMN train_splits INTEGER",
    "ALTER TABLE runs ADD COLUMN window_idx INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE runs ADD COLUMN window_size REAL",
    "ALTER TABLE runs ADD COLUMN window_start REAL",
    "ALTER TABLE runs ADD COLUMN window_end REAL",
]


class BenchmarkDB:
    """Manage a SQLite database for benchmark results."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Apply schema migrations for backwards compatibility."""
        for stmt in _MIGRATIONS:
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if (
                    "duplicate column" in msg
                    or "already exists" in msg
                    or "no such column" in msg
                ):
                    pass  # Expected: migration already applied or not applicable
                else:
                    logger.warning("Migration failed: %s — %s", stmt, e)
                    raise

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def get_or_create_run(
        self,
        dataset: str,
        model: str,
        window_idx: int,
        seed: int,
        window_size: Optional[float] = None,
        window_start: Optional[float] = None,
        window_end: Optional[float] = None,
    ) -> int:
        """Return the run id, creating the row if it does not exist.

        New API uses window_idx instead of (benchmark_rounds, train_splits, round_idx).
        """
        cur = self.conn.execute(
            "SELECT id FROM runs WHERE dataset=? AND model=? "
            "AND window_idx=? AND seed=?",
            (dataset, model, window_idx, seed),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO runs(dataset, model, window_idx, seed, "
            "window_size, window_start, window_end) VALUES(?,?,?,?,?,?,?)",
            (dataset, model, window_idx, seed, window_size, window_start, window_end),
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

    def save_losses(self, run_id: int, losses: List[Tuple[int, float]]):
        """Store per-epoch training losses.

        Args:
            run_id: The run to attach losses to.
            losses: List of ``(epoch, loss_value)`` tuples.
        """
        for epoch, loss in losses:
            self.conn.execute(
                "INSERT OR REPLACE INTO losses(run_id, epoch, loss) "
                "VALUES(?,?,?)",
                (run_id, epoch, loss),
            )
        self.conn.commit()

    def get_losses(
        self,
        dataset: str,
        model: str,
        benchmark_rounds: Optional[int] = None,
        train_splits: Optional[int] = None,
        round_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int, int, float]]:
        """Return ``(round_idx, seed, epoch, loss)`` rows."""
        query = (
            "SELECT r.round_idx, r.seed, l.epoch, l.loss "
            "FROM losses l JOIN runs r ON l.run_id = r.id "
            "WHERE r.dataset=? AND r.model=? AND r.status='done'"
        )
        params: list = [dataset, model]
        if benchmark_rounds is not None:
            query += " AND r.benchmark_rounds=?"
            params.append(benchmark_rounds)
        if train_splits is not None:
            query += " AND r.train_splits=?"
            params.append(train_splits)
        if round_idx is not None:
            query += " AND r.round_idx=?"
            params.append(round_idx)
        if seed is not None:
            query += " AND r.seed=?"
            params.append(seed)
        query += " ORDER BY r.round_idx, r.seed, l.epoch"
        return self.conn.execute(query, params).fetchall()

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
        self,
        dataset: str,
        model: str,
        window_idx: int,
        seed: int,
    ) -> Optional[str]:
        """Return the status of a run, or None if it doesn't exist."""
        cur = self.conn.execute(
            "SELECT status FROM runs "
            "WHERE dataset=? AND model=? AND window_idx=? AND seed=?",
            (dataset, model, window_idx, seed),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def is_done(
        self,
        dataset: str,
        model: str,
        window_idx: int,
        seed: int,
    ) -> bool:
        """Check if a run is already done."""
        return self.run_status(dataset, model, window_idx, seed) == "done"

    def get_metrics(
        self,
        dataset: str,
        model: str,
        phase: str = "test",
        benchmark_rounds: Optional[int] = None,
        train_splits: Optional[int] = None,
        round_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int, str, float]]:
        """Return ``(round_idx, seed, metric_name, metric_value)`` rows.

        Optionally filter by benchmark_rounds, train_splits, round_idx, seed.
        """
        query = (
            "SELECT r.round_idx, r.seed, m.name, m.value "
            "FROM metrics m JOIN runs r ON m.run_id = r.id "
            "WHERE r.dataset=? AND r.model=? AND m.phase=? AND r.status='done'"
        )
        params: list = [dataset, model, phase]
        if benchmark_rounds is not None:
            query += " AND r.benchmark_rounds=?"
            params.append(benchmark_rounds)
        if train_splits is not None:
            query += " AND r.train_splits=?"
            params.append(train_splits)
        if round_idx is not None:
            query += " AND r.round_idx=?"
            params.append(round_idx)
        if seed is not None:
            query += " AND r.seed=?"
            params.append(seed)
        query += " ORDER BY r.round_idx, r.seed, m.name"
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
            "SELECT dataset, model, benchmark_rounds, train_splits, status, COUNT(*) "
            "FROM runs GROUP BY dataset, model, benchmark_rounds, train_splits, status "
            "ORDER BY dataset, model"
        )
        lines = ["dataset      | model         | m  | n  | status  | count"]
        lines.append("-" * 64)
        for ds, mdl, br, ts, st, cnt in cur:
            lines.append(f"{ds:<12} | {mdl:<13} | {br:<2} | {ts:<2} | {st:<7} | {cnt}")
        return "\n".join(lines)

    def reset_stale_running(self) -> int:
        """Reset all 'running' runs to 'pending' (from interrupted sessions).

        Returns:
            Number of rows reset.
        """
        cur = self.conn.execute(
            "UPDATE runs SET status='pending', started_at=NULL "
            "WHERE status='running'"
        )
        count = cur.rowcount
        if count > 0:
            logger.info(
                "Reset %d stale 'running' runs to 'pending'.", count
            )
        self.conn.commit()
        return count

    def delete_runs(
        self,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        benchmark_rounds: Optional[int] = None,
        train_splits: Optional[int] = None,
    ) -> int:
        """Delete runs matching the given criteria.

        Args:
            dataset: Dataset name filter (None = all datasets).
            model: Model name filter (None = all models).
            benchmark_rounds: Benchmark rounds filter (None = all).
            train_splits: Train splits filter (None = all).

        Returns:
            Number of runs deleted.
        """
        # Build WHERE clause
        conditions = []
        params = []
        if dataset is not None:
            conditions.append("dataset=?")
            params.append(dataset)
        if model is not None:
            conditions.append("model=?")
            params.append(model)
        if benchmark_rounds is not None:
            conditions.append("benchmark_rounds=?")
            params.append(benchmark_rounds)
        if train_splits is not None:
            conditions.append("train_splits=?")
            params.append(train_splits)

        if not conditions:
            logger.warning("delete_runs called with no filters — refusing to delete all runs.")
            return 0

        where_clause = " AND ".join(conditions)

        # Get run IDs to delete
        cur = self.conn.execute(
            f"SELECT id FROM runs WHERE {where_clause}",
            params,
        )
        run_ids = [row[0] for row in cur.fetchall()]

        if not run_ids:
            return 0

        # Delete related metrics and losses first (foreign key cascade)
        placeholders = ",".join("?" * len(run_ids))
        self.conn.execute(
            f"DELETE FROM metrics WHERE run_id IN ({placeholders})",
            run_ids,
        )
        self.conn.execute(
            f"DELETE FROM losses WHERE run_id IN ({placeholders})",
            run_ids,
        )
        # Delete runs
        self.conn.execute(
            f"DELETE FROM runs WHERE id IN ({placeholders})",
            run_ids,
        )
        self.conn.commit()

        logger.info(
            "Deleted %d runs matching filters: dataset=%s, model=%s, m=%s, n=%s",
            len(run_ids), dataset, model, benchmark_rounds, train_splits,
        )
        return len(run_ids)

    def delete_dataset_runs(self, dataset: str) -> int:
        """Delete all runs for a specific dataset.

        This cascades to metrics and losses via foreign key constraints
        if they were defined with ON DELETE CASCADE. Since our schema
        doesn't have explicit CASCADE, we manually delete from child tables.

        Returns:
            Number of runs deleted.
        """
        # Get run IDs for this dataset
        cur = self.conn.execute(
            "SELECT id FROM runs WHERE dataset=?", (dataset,)
        )
        run_ids = [row[0] for row in cur.fetchall()]

        if not run_ids:
            return 0

        # Delete from child tables
        placeholders = ",".join("?" * len(run_ids))
        self.conn.execute(
            f"DELETE FROM metrics WHERE run_id IN ({placeholders})", run_ids
        )
        self.conn.execute(
            f"DELETE FROM losses WHERE run_id IN ({placeholders})", run_ids
        )

        # Delete from runs table
        self.conn.execute("DELETE FROM runs WHERE dataset=?", (dataset,))
        self.conn.commit()

        logger.info("Deleted %d runs for dataset '%s'.", len(run_ids), dataset)
        return len(run_ids)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
