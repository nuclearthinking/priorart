"""Indexing layer: parsing, capture, inventory, the refresh pipeline and
the job manager.

Public surface: ``Symbol``, ``ParseResult``, ``parse_source``,
``preflight_parsers``, ``language_of``, ``LANGS``, ``capture_file``,
``list_source_files``, ``repo_languages``, ``head_revision``,
``InventoryError``, ``index_repo``, ``RefreshCancelledError``, ``JobManager``,
``journal_job``,
``start_watcher``.
"""

from __future__ import annotations

from .capture import capture_file
from .inventory import InventoryError, head_revision, list_source_files, repo_languages
from .jobs import JobManager, journal_job
from .parser import (
    LANGS,
    ParseResult,
    Symbol,
    language_of,
    parse_source,
    preflight_parsers,
)
from .pipeline import RefreshCancelledError, index_repo
from .watcher import start_watcher

__all__ = [
    "LANGS",
    "InventoryError",
    "JobManager",
    "ParseResult",
    "RefreshCancelledError",
    "Symbol",
    "capture_file",
    "head_revision",
    "index_repo",
    "journal_job",
    "language_of",
    "list_source_files",
    "parse_source",
    "preflight_parsers",
    "repo_languages",
    "start_watcher",
]
