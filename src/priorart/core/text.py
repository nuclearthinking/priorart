"""Bounded text helpers shared by the parser and the retrieval pipeline."""

from __future__ import annotations


def bounded_body(body: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(body) <= max_chars:
        return body
    kept = body[:max_chars]
    cut = kept.rfind("\n")
    if cut > 0:
        kept = kept[:cut]
    skipped = len(body.splitlines()) - len(kept.splitlines())
    if skipped > 0:
        return f"{kept}\n… (+{skipped} lines)"
    if len(kept) < len(body):
        return f"{kept}\n…"
    return kept
