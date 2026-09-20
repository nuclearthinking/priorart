"""SQLite store: schema, writer initialization and reader opening.

One store file belongs to exactly one (worktree, profile) pair. Writers are
initialized under an inter-process ownership lock and may reset an
incompatible schema (the index is a throwaway artifact, rebuildable by
reindexing). Readers never reset anything: an incompatible or absent store
is reported, not repaired.
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from priorart.core import (
    APP_VERSION,
    INDEX_PROFILE_MISMATCH,
    PriorartError,
)

SCHEMA_VERSION = 7

SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS repos (
        repo TEXT PRIMARY KEY,
        head TEXT,
        indexed_at REAL,
        epoch INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS files (
        repo TEXT NOT NULL,
        path TEXT NOT NULL,
        mtime_ns INTEGER NOT NULL,
        size INTEGER NOT NULL,
        hash TEXT NOT NULL DEFAULT '',
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
        embed_text TEXT NOT NULL,
        embed_key TEXT NOT NULL DEFAULT '',
        source_role TEXT NOT NULL DEFAULT 'unknown'
    )""",
    """CREATE INDEX IF NOT EXISTS symbols_repo_path ON symbols(repo, path)""",
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
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        repo TEXT NOT NULL,
        mode TEXT NOT NULL,
        state TEXT NOT NULL,
        phase TEXT,
        created_at REAL NOT NULL,
        heartbeat_at REAL NOT NULL,
        counters TEXT NOT NULL,
        error TEXT,
        epoch INTEGER NOT NULL DEFAULT 0
    )""",
)

_OWNED_TABLES = frozenset(
    {"meta", "repos", "files", "parse_state", "symbols", "symbols_fts", "symbols_vec", "jobs"}
)


@dataclass(frozen=True)
class StoreProfile:
    """Everything that defines the meaning of stored vectors."""

    embed_dim: int
    embed_model: str = ""
    embed_input_format: str = ""

    @property
    def app_version(self) -> str:
        return APP_VERSION

    def as_meta(self) -> dict[str, str]:
        return {
            "app_version": APP_VERSION,
            "embed_dim": str(self.embed_dim),
            "embed_model": self.embed_model or "",
            "embed_input_format": self.embed_input_format or "",
        }

    def describe(self) -> str:
        meta = self.as_meta()
        return ", ".join(f"{key}={value!r}" for key, value in sorted(meta.items()))


def initialize_writer(store_path: Path, profile: StoreProfile) -> sqlite3.Connection:
    """Open (creating if needed) the store for exclusive index writing.

    An incompatible schema or embedding profile resets the schema: the index
    is fully rebuildable from the repository sources. The reset happens in
    one transaction, so readers see either the old or the new published
    state, never a mix.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    conn: sqlite3.Connection | None = None
    try:
        conn = _connect(store_path)
        _initialize(conn, store_path, profile)
    except sqlite3.DatabaseError as err:
        if conn is not None:
            conn.close()
        raise RuntimeError(
            f"failed to initialize priorart database at {store_path}: {err}"
        ) from err
    except BaseException:
        if conn is not None:
            conn.close()
        raise
    return conn


def open_reader(store_path: Path, profile: StoreProfile) -> sqlite3.Connection | None:
    """Open the published store for reading, or ``None`` when absent.

    Readers never reset the schema and never create or mutate the store: a
    store written by another application version or embedding profile is
    an explicit incompatibility, not something to repair. The read-only
    URI also makes the exists() check race-safe (a deleted store cannot be
    recreated as an empty file by the reader).
    """
    store_path = Path(store_path)
    if not store_path.exists():
        return None
    try:
        # percent-encode the path: a raw ? or # would terminate the URI and
        # silently open the wrong (empty) database
        uri = f"file:{urllib.parse.quote(str(store_path), safe='/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    except sqlite3.OperationalError:
        return None
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.enable_load_extension(True)  # noqa: FBT003 - sqlite3 positional-only API
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)  # noqa: FBT003 - sqlite3 positional-only API
        _require_compatible(conn, store_path, profile)
    except sqlite3.OperationalError:
        # a WAL database in a directory the reader cannot write (needs the
        # -shm file) behaves the same as an unopenable store: absent for
        # this reader, never a crash
        conn.close()
        return None
    except BaseException:
        conn.close()
        raise
    return conn


def _connect(store_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(store_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.enable_load_extension(True)  # noqa: FBT003 - sqlite3 positional-only API
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)  # noqa: FBT003 - sqlite3 positional-only API
    return conn


def _initialize(conn: sqlite3.Connection, store_path: Path, profile: StoreProfile) -> None:
    if _up_to_date(conn, profile):
        return
    _require_priorart_owned(conn, store_path)
    _reset_schema(conn, profile)


def _require_compatible(conn: sqlite3.Connection, store_path: Path, profile: StoreProfile) -> None:
    if not _up_to_date(conn, profile):
        raise PriorartError(
            INDEX_PROFILE_MISMATCH,
            f"the index at {store_path} was built by another priorart version or "
            "embedding profile; run refresh_index to rebuild it",
            stored_profile=_stored_description(conn),
            requested_profile=profile.describe(),
        )


def _stored_description(conn: sqlite3.Connection) -> str:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "meta" not in tables:
        return "unknown (no priorart metadata)"
    stored = dict(conn.execute("SELECT key, value FROM meta"))
    return ", ".join(f"{key}={stored.get(key)!r}" for key in sorted(stored)) or "empty"


def _require_priorart_owned(conn: sqlite3.Connection, store_path: Path) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not row[0].startswith("sqlite_")
    }
    if not tables:
        return  # a fresh database file; user_version is still 0
    if "meta" in tables:
        return  # carries priorart metadata
    if version in (0, SCHEMA_VERSION) and _all_priorart_tables(tables):
        return  # legacy priorart layout from before user_version was recorded
    unknown = ", ".join(sorted(tables - _OWNED_TABLES)) or "no unknown tables"
    raise RuntimeError(
        f"priorart database at {store_path} has PRAGMA user_version = {version} and tables "
        f"it does not own ({unknown}); refusing to overwrite a database it does not own"
    )


def _all_priorart_tables(tables: set[str]) -> bool:
    # fts5/vec0 virtual tables carry shadow tables sharing their prefix
    return all(
        table in _OWNED_TABLES or table.startswith(("symbols_fts", "symbols_vec"))
        for table in tables
    )


def _up_to_date(conn: sqlite3.Connection, profile: StoreProfile) -> bool:
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        return False
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "meta" not in tables:
        return False
    expected = profile.as_meta()
    stored = dict(
        conn.execute("SELECT key, value FROM meta WHERE key IN (?, ?, ?, ?)", tuple(expected))
    )
    return stored == expected


def _reset_schema(conn: sqlite3.Connection, profile: StoreProfile) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _up_to_date(conn, profile):
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
            f"embedding float[{profile.embed_dim}])"
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            tuple(profile.as_meta().items()),
        )
        conn.execute("COMMIT")
    except BaseException:
        _rollback(conn)
        raise


def _rollback(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")
