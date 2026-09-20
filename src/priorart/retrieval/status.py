"""Index status: cheap publication metadata and opt-in live freshness.

``published_summary`` reads only the index store and is safe for refresh/job
responses. ``status_summary`` adds Git and file observations for an explicit
status request. Dense coverage never changes source freshness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from priorart.core import git_output
from priorart.indexing import list_source_files


@dataclass
class IndexSummary:
    """Published index state of one repository, read in one transaction."""

    repo: str
    state: str  # "ready" or "absent"
    head_at_capture: str | None = None
    head_observed: str | None = None
    epoch: int = 0
    files: int = 0
    symbols: int = 0
    vectorized: int = 0
    indexed_at: float | None = None
    working_tree_dirty: bool = False
    stale_reasons: list[str] = field(default_factory=list)
    parse_counts: dict[str, int] = field(default_factory=dict)
    dense_enabled: bool = True

    @property
    def freshness(self) -> str:
        if self.state == "absent":
            return "absent"
        return "stale" if self.stale_reasons else "fresh"

    def payload(self) -> dict:
        return {
            "repo": self.repo,
            "state": self.state,
            "index_epoch": self.epoch,
            "head_at_capture": self.head_at_capture,
            "head_observed": self.head_observed,
            "files": self.files,
            "symbols": self.symbols,
            "vectorized": self.vectorized,
            "indexed_at": self.indexed_at,
            "parse_counts": dict(self.parse_counts),
            "source_kind": "working_tree",
            "working_tree_dirty": self.working_tree_dirty,
            "freshness": self.freshness,
            "lexical": {"state": "ready" if self.epoch else "absent"},
            "dense": {
                "state": (
                    "disabled"
                    if not self.dense_enabled
                    else "ready"
                    if self.vectorized == self.symbols
                    else "partial"
                ),
                "ready": self.vectorized,
                "eligible": self.symbols,
            },
            "stale_reasons": list(self.stale_reasons),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> IndexSummary:
        """Rebuild a summary from its wire payload (daemon round-trip)."""
        return cls(
            repo=payload["repo"],
            state=payload.get("state", "ready"),
            head_at_capture=payload.get("head_at_capture"),
            head_observed=payload.get("head_observed"),
            epoch=payload.get("index_epoch", 0),
            files=payload.get("files", 0),
            symbols=payload.get("symbols", 0),
            vectorized=payload.get("vectorized", 0),
            indexed_at=payload.get("indexed_at"),
            working_tree_dirty=payload.get("working_tree_dirty", False),
            stale_reasons=payload.get("stale_reasons", []),
            parse_counts=payload.get("parse_counts", {}),
            dense_enabled=payload.get("dense", {}).get("state") != "disabled",
        )


def published_epoch(conn, repo: str) -> bool:
    """True when the repo has a published index epoch (> 0).

    The single schema-aware answer to "is there something to search";
    composers stay SQL-free by using it.
    """
    row = conn.execute("SELECT epoch FROM repos WHERE repo = ?", (repo,)).fetchone()
    return row is not None and row[0] > 0


def published_file_state(conn, repo: str) -> dict[str, tuple[int, int, str]]:
    """File state of the last publication: ``{path: (mtime_ns, size, hash)}``.

    The schema-aware source for drift detection; the registry composes it
    instead of owning files-table SQL.
    """
    rows = conn.execute(
        "SELECT path, mtime_ns, size, hash FROM files WHERE repo = ?", (repo,)
    ).fetchall()
    return {path: (mtime_ns, size, digest) for path, mtime_ns, size, digest in rows}


def published_summary(conn, repo: str, *, dense: bool = True) -> IndexSummary:
    """Read published index metadata without touching Git or source files."""
    summary = IndexSummary(repo=repo, state="ready", dense_enabled=dense)
    conn.execute("BEGIN")
    try:
        head, indexed_at, epoch = repo_meta(conn, repo)
        summary.head_at_capture = head
        summary.indexed_at = indexed_at
        summary.epoch = epoch or 0
        summary.files = conn.execute(
            "SELECT COUNT(*) FROM files WHERE repo = ?", (repo,)
        ).fetchone()[0]
        summary.symbols = conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo = ?", (repo,)
        ).fetchone()[0]
        summary.vectorized = conn.execute(
            "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (repo,)
        ).fetchone()[0]
        summary.parse_counts = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM parse_state WHERE repo = ? GROUP BY status",
                (repo,),
            ).fetchall()
        )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    if not summary.epoch:
        summary.state = "absent"
        summary.stale_reasons.append("not indexed")
    return summary


def status_summary(conn, repo: str, *, dense: bool = True) -> IndexSummary:
    """Add live source freshness observations to published metadata."""
    root = Path(repo)
    summary = published_summary(conn, repo, dense=dense)
    summary.head_observed = git_output(root, "rev-parse", "HEAD")
    summary.working_tree_dirty = git_output(root, "status", "--porcelain") is not None
    if not summary.epoch:
        return summary
    if summary.head_observed and summary.head_at_capture != summary.head_observed:
        summary.stale_reasons.append("HEAD moved since indexing")
    gone, changed = _file_drift(conn, repo, root)
    if gone:
        summary.stale_reasons.append(f"{gone} indexed files no longer present")
    if changed:
        summary.stale_reasons.append(f"{changed} files changed since indexing")
    return summary


def format_status(summary: IndexSummary) -> str:
    """Render one summary as text; no second read, no torn snapshot."""
    lines = [
        f"repo: {summary.repo}",
        f"index_head: {summary.head_at_capture or '-'} | current_head: {summary.head_observed or '-'}",
        f"working_tree_dirty: {'yes' if summary.working_tree_dirty else 'no'}",
        f"stale: {'; '.join(summary.stale_reasons) if summary.stale_reasons else 'no'}",
        (
            f"files: {summary.files}, symbols: {summary.symbols}, "
            f"vectorized: {summary.vectorized}/{summary.symbols}"
        ),
    ]
    if summary.epoch:
        lines.append(f"index_epoch: {summary.epoch}")
    if summary.parse_counts:
        lines.append(
            "parse: "
            + ", ".join(
                f"{count} {status}" for status, count in sorted(summary.parse_counts.items())
            )
        )
    if summary.indexed_at:
        lines.append(f"indexed: {age(time.time() - summary.indexed_at)} ago")
    return "\n".join(lines)


def status_text(conn, repo: str, *, dense: bool = True) -> str:
    return format_status(status_summary(conn, repo, dense=dense))


def repo_meta(conn, repo: str) -> tuple[str | None, float | None, int | None]:
    row = conn.execute(
        "SELECT head, indexed_at, epoch FROM repos WHERE repo = ?", (repo,)
    ).fetchone()
    return (row[0], row[1], row[2]) if row else (None, None, None)


def _file_drift(conn, repo: str, root: Path) -> tuple[int, int]:
    expected = set(list_source_files(root))
    known = {
        path: (mtime_ns, size)
        for path, mtime_ns, size in conn.execute(
            "SELECT path, mtime_ns, size FROM files WHERE repo = ?", (repo,)
        )
    }
    gone = sum(1 for path in known if path not in expected)
    changed = len(expected - set(known))
    for path, (mtime_ns, size) in known.items():
        if path not in expected:
            continue
        try:
            st = (root / path).stat()
        except OSError:
            changed += 1
            continue
        if (st.st_mtime_ns, st.st_size) != (mtime_ns, size):
            changed += 1
    return gone, changed


def age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
