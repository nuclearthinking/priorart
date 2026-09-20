"""Retrieval layer: search, status and symbol mapping over a published
index store.

Public surface: ``search``, ``format_report``, ``format_status``,
``Candidate``, ``SearchReport``, ``SearchTrace``, ``rerank_document``,
``rerank_positions``, ``valid_rerank_order``, ``rrf``, ``status_summary``,
``status_text``, ``map_symbols_rows``, ``IndexSummary``, ``published_epoch``,
``published_summary``, ``published_file_state``.
"""

from __future__ import annotations

from .search import (
    CANDIDATE_LIMIT,
    DEADLINE_FALLBACK,
    DENSE_INDEX_PARTIAL,
    PARSE_COVERAGE_PARTIAL,
    QUERY_MODEL_UNAVAILABLE,
    RERANK_FALLBACK,
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
from .status import (
    IndexSummary,
    format_status,
    published_epoch,
    published_file_state,
    published_summary,
    status_summary,
    status_text,
)

__all__ = [
    "CANDIDATE_LIMIT",
    "DEADLINE_FALLBACK",
    "DENSE_INDEX_PARTIAL",
    "PARSE_COVERAGE_PARTIAL",
    "QUERY_MODEL_UNAVAILABLE",
    "RERANK_FALLBACK",
    "Candidate",
    "IndexSummary",
    "SearchReport",
    "SearchTrace",
    "format_report",
    "format_status",
    "map_symbols_rows",
    "published_epoch",
    "published_file_state",
    "published_summary",
    "rerank_document",
    "rerank_positions",
    "rrf",
    "search",
    "status_summary",
    "status_text",
    "valid_rerank_order",
]
