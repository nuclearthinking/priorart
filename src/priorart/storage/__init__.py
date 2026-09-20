"""Storage layer: per-worktree store layout, writer/reader openings and
inter-process ownership locks.

Public surface: ``worktree_dir``, ``profile_id``, ``store_path``,
``ensure_worktree_dir``, ``known_roots``, ``StoreProfile``,
``initialize_writer``, ``open_reader``, ``FileLock``, ``SCHEMA_VERSION``.
"""

from __future__ import annotations

from .layout import (
    STORE_LAYOUT_VERSION,
    embed_cache_path,
    ensure_worktree_dir,
    known_roots,
    profile_id,
    store_path,
    worktree_dir,
)
from .locks import FileLock
from .store import (
    SCHEMA_VERSION,
    StoreProfile,
    initialize_writer,
    open_reader,
)

__all__ = [
    "SCHEMA_VERSION",
    "STORE_LAYOUT_VERSION",
    "FileLock",
    "StoreProfile",
    "embed_cache_path",
    "ensure_worktree_dir",
    "initialize_writer",
    "known_roots",
    "open_reader",
    "profile_id",
    "store_path",
    "worktree_dir",
]
