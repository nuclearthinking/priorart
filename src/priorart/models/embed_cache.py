"""Shared embedding cache: exact model inputs mapped to their vectors.

The cache is one SQLite file under the index root, used by every worktree
and profile. Rows are keyed by the embedding space identity, the input role
(document or query) and the hash of the exact prepared input, so vectors of
different spaces never mix. Transactions are short: lookups and stores touch
only their own rows and never block index publication.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    space_id TEXT NOT NULL,
    role TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (space_id, role, input_hash)
) WITHOUT ROWID
"""

_CHUNK = 400


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbedCache:
    """Thread-safe accessor for the shared vector cache file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = self._open()
        except sqlite3.DatabaseError:
            # the cache is a throwaway, rebuildable artifact: a corrupt file
            # must never brick resolve() — recreate it empty instead
            for suffix in ("", "-wal", "-shm"):
                Path(str(self.path) + suffix).unlink(missing_ok=True)
            self._conn = self._open()
        self._mutex = threading.Lock()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(_SCHEMA)
            conn.commit()
        except BaseException:
            conn.close()
            raise
        return conn

    def lookup(self, space_id: str, role: str, texts: list[str]) -> dict[str, bytes]:
        """Return ``{input_hash: vector}`` for the inputs already cached."""
        found: dict[str, bytes] = {}
        unique = {input_hash(text) for text in texts}
        if not unique:
            return found
        keys = sorted(unique)
        with self._mutex:
            for start in range(0, len(keys), _CHUNK):
                chunk = keys[start : start + _CHUNK]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT input_hash, dim, vector FROM vectors "  # noqa: S608 - marks are placeholders only
                    f"WHERE space_id = ? AND role = ? AND input_hash IN ({marks})",
                    (space_id, role, *chunk),
                ).fetchall()
                for input_key, dim, vector in rows:
                    if not vector or len(vector) != dim * 4:
                        continue
                    found[input_key] = vector
        return found

    def store(self, space_id: str, role: str, items: list[tuple[str, bytes]], dim: int) -> None:
        """Persist freshly computed vectors; one short write transaction."""
        if not items:
            return
        now = time.time()
        with self._mutex:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO vectors "
                    "(space_id, role, input_hash, dim, vector, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [(space_id, role, key, dim, vector, now) for key, vector in items],
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._mutex:
            self._conn.close()
