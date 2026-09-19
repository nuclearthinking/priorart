from __future__ import annotations

import sqlite3
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 4

try:
    APP_VERSION = version("priorart")
except PackageNotFoundError:  # running from a source checkout without installation
    APP_VERSION = "unknown"

SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS repos (
        repo TEXT PRIMARY KEY,
        head TEXT,
        indexed_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS files (
        repo TEXT NOT NULL,
        path TEXT NOT NULL,
        mtime_ns INTEGER NOT NULL,
        size INTEGER NOT NULL,
        PRIMARY KEY (repo, path)
    )""",
    """CREATE TABLE IF NOT EXISTS parse_state (
        repo TEXT NOT NULL,
        path TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT,
        attempted_at REAL NOT NULL,
        PRIMARY KEY (repo, path)
    )""",
    """CREATE TABLE IF NOT EXISTS symbols (
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
        full_signature TEXT NOT NULL,
        docstring TEXT NOT NULL,
        body TEXT NOT NULL,
        search_text TEXT NOT NULL,
        embed_text TEXT NOT NULL
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
        search_text,
        content='symbols',
        content_rowid='id',
        tokenize='porter unicode61'
    )""",
    """CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
        INSERT INTO symbols_fts(rowid, search_text) VALUES (new.id, new.search_text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
        INSERT INTO symbols_fts(symbols_fts, rowid, search_text)
        VALUES ('delete', old.id, old.search_text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
        INSERT INTO symbols_fts(symbols_fts, rowid, search_text)
        VALUES ('delete', old.id, old.search_text);
        INSERT INTO symbols_fts(rowid, search_text) VALUES (new.id, new.search_text);
    END""",
)

_OWNED_TABLES = frozenset(
    {"meta", "repos", "files", "parse_state", "symbols", "symbols_fts", "symbols_vec"}
)


def connect(db_path: Path, embed_dim: int) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.enable_load_extension(True)  # noqa: FBT003 - sqlite3 positional-only API
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)  # noqa: FBT003 - sqlite3 positional-only API
        _initialize(conn, db_path, embed_dim)
    except sqlite3.DatabaseError as err:
        raise RuntimeError(f"failed to initialize priorart database at {db_path}: {err}") from err
    return conn


def _initialize(conn: sqlite3.Connection, db_path: Path, embed_dim: int) -> None:
    if _up_to_date(conn, embed_dim):
        return
    _require_priorart_owned(conn, db_path)
    _reset_schema(conn, embed_dim)


def _require_priorart_owned(conn: sqlite3.Connection, db_path: Path) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version in (0, SCHEMA_VERSION):
        return
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "meta" in tables:
        return
    raise RuntimeError(
        f"priorart database at {db_path} has PRAGMA user_version = {version} and no priorart "
        "metadata; refusing to overwrite a database it does not own"
    )


def _up_to_date(conn: sqlite3.Connection, embed_dim: int) -> bool:
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        return False
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "meta" not in tables:
        return False
    expected = {"app_version": APP_VERSION, "embed_dim": str(embed_dim)}
    stored = dict(
        conn.execute("SELECT key, value FROM meta WHERE key IN ('app_version', 'embed_dim')")
    )
    return stored == expected


def _reset_schema(conn: sqlite3.Connection, embed_dim: int) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _up_to_date(conn, embed_dim):
            conn.execute("COMMIT")
            return
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        for table in tables:
            if table in _OWNED_TABLES:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        for table in tables:
            if table.startswith(("symbols_fts", "symbols_vec")):
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS symbols_vec USING vec0("
            "symbol_id INTEGER PRIMARY KEY, repo TEXT partition key, "
            f"embedding float[{embed_dim}])"
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (("app_version", APP_VERSION), ("embed_dim", str(embed_dim))),
        )
        conn.execute("COMMIT")
    except BaseException:
        _rollback(conn)
        raise


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")
