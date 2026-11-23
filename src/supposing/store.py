"""Where a run's comparisons are kept between steps, so a run can stop and resume.

The engine needs only three operations. ``MemoryStore`` keeps nothing after the process ends;
``SQLiteStore`` keeps a run in one local file. Hosted deployments supply their own.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Protocol

from .ranking import Comparison

_FIELDS = [f.name for f in fields(Comparison)]


class RunStore(Protocol):
    """Where ``run_ranking`` keeps comparisons. Implement these three methods to keep runs elsewhere."""

    def load(self, run_id: str) -> list[Comparison]:
        """Every comparison of the run, in the order they were added. Return copies, not live objects."""
        ...

    def add(self, run_id: str, comparisons: list[Comparison]) -> None:
        """Append newly planned comparisons. Ids already present are ignored, so a replanned round is safe."""
        ...

    def update(self, run_id: str, comparison: Comparison) -> None:
        """Save a comparison's new status, verdict and cost."""
        ...


class MemoryStore:
    """Keeps runs in this process only. The default when no store is given."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Comparison]] = {}

    def load(self, run_id: str) -> list[Comparison]:
        return [Comparison(**asdict(c)) for c in self._runs.get(run_id, {}).values()]

    def add(self, run_id: str, comparisons: list[Comparison]) -> None:
        run = self._runs.setdefault(run_id, {})
        for c in comparisons:
            run.setdefault(c.id, Comparison(**asdict(c)))

    def update(self, run_id: str, comparison: Comparison) -> None:
        self._runs.setdefault(run_id, {})[comparison.id] = Comparison(**asdict(comparison))


class SQLiteStore:
    """Keeps runs in one SQLite file. Each write commits, so a crash loses at most the calls in flight.

    Pass the same path and ``run_id`` again to continue a run. One file can hold many runs.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS comparisons (run_id TEXT NOT NULL, id TEXT NOT NULL, seq INTEGER NOT NULL,"
            " data TEXT NOT NULL, PRIMARY KEY (run_id, id))"
        )

    def load(self, run_id: str) -> list[Comparison]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM comparisons WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [Comparison(**{k: v for k, v in json.loads(r[0]).items() if k in _FIELDS}) for r in rows]

    def add(self, run_id: str, comparisons: list[Comparison]) -> None:
        with self._lock:
            seq = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM comparisons WHERE run_id = ?",
                                     (run_id,)).fetchone()[0]
            self._conn.execute("BEGIN")
            for i, c in enumerate(comparisons, 1):
                self._conn.execute("INSERT OR IGNORE INTO comparisons (run_id, id, seq, data) VALUES (?, ?, ?, ?)",
                                   (run_id, c.id, seq + i, json.dumps(asdict(c))))
            self._conn.execute("COMMIT")

    def update(self, run_id: str, comparison: Comparison) -> None:
        with self._lock:
            self._conn.execute("UPDATE comparisons SET data = ? WHERE run_id = ? AND id = ?",
                               (json.dumps(asdict(comparison)), run_id, comparison.id))

    def close(self) -> None:
        self._conn.close()
