from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    repo TEXT PRIMARY KEY,
    head TEXT,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS files (
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    PRIMARY KEY (repo, path)
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    qualname TEXT NOT NULL,
    kind TEXT NOT NULL,
    lang TEXT NOT NULL,
    line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embed_text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    search_text,
    content='symbols',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
    INSERT INTO symbols_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, search_text)
    VALUES ('delete', old.id, old.search_text);
END;
CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
    INSERT INTO symbols_fts(symbols_fts, rowid, search_text)
    VALUES ('delete', old.id, old.search_text);
    INSERT INTO symbols_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
"""


def connect(db_path: Path, embed_dim: int) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.enable_load_extension(True)  # noqa: FBT003 - sqlite3 positional-only API
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)  # noqa: FBT003 - sqlite3 positional-only API
    try:
        _drop_legacy_files_table(conn)
        conn.executescript(SCHEMA)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'symbols_vec'"
        ).fetchone()
        if row is not None and "partition key" not in row[0]:
            conn.execute("DROP TABLE symbols_vec")
            conn.execute("DELETE FROM files")
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS symbols_vec USING vec0("
            "symbol_id INTEGER PRIMARY KEY, repo TEXT partition key, "
            f"embedding float[{embed_dim}])"
        )
    except sqlite3.OperationalError as err:
        raise RuntimeError(f"failed to initialize priorart database at {db_path}: {err}") from err
    return conn


def _drop_legacy_files_table(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    if columns and "mtime_ns" not in columns:
        conn.execute("DROP TABLE files")
