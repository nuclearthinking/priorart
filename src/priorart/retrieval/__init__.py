"""Retrieval layer: search, status and symbol mapping over a published
index store.

Public surface: ``search``, ``format_report``, ``Candidate``, ``SearchReport``,
``SearchTrace``, ``rerank_document``, ``rerank_positions``,
``valid_rerank_order``, ``rrf``, ``status_summary``, ``status_text``,
``map_symbols_rows``, ``IndexSummary``, ``published_epoch``,
``published_file_state``.
"""

from __future__ import annotations

from .search import (
    CANDIDATE_LIMIT,
    Candidate,
    SearchReport,
    SearchTrace,
    format_report,
    map_symbols_rows,
    rerank_document,
    rerank_positions,
    rrf,
    search,
    valid_rerank_order,
)
from .status import IndexSummary, published_epoch, published_file_state, status_summary, status_text

__all__ = [
    "CANDIDATE_LIMIT",
    "Candidate",
    "IndexSummary",
    "SearchReport",
    "SearchTrace",
    "format_report",
    "map_symbols_rows",
    "published_epoch",
    "published_file_state",
    "rerank_document",
    "rerank_positions",
    "rrf",
    "search",
    "status_summary",
    "status_text",
    "valid_rerank_order",
]
