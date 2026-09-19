from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .indexer import _git, _list_files

CANDIDATE_LIMIT = 50
RRF_K = 60
RERANK_DOCUMENT_FORMAT = "path-qualname-kind-signature-docstring-v1"


@dataclass
class Candidate:
    path: str
    name: str
    qualname: str
    kind: str
    lang: str
    line: int
    end_line: int
    signature: str
    full_signature: str
    docstring: str
    score: float


@dataclass
class SearchTrace:
    queries: list[str]
    stage_seconds: dict[str, float]
    fts_rankings: list[list[int]]
    vec_rankings: list[list[int]]
    fused: list[tuple[int, float]]
    rerank_order: list[int] | None


@dataclass
class SearchReport:
    candidates: list[Candidate]
    warnings: list[str]
    symbol_count: int
    head: str | None
    age_seconds: float | None
    repo: str | None = None
    current_head: str | None = None
    trace: SearchTrace | None = None
    parse_coverage: dict[str, int] | None = None


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, symbol_id in enumerate(ranking):
            scores[symbol_id] = scores.get(symbol_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def fts_search(conn, repo: str, query: str, limit: int = 50) -> list[int]:
    terms = re.findall(r"\w{2,}", query)
    if not terms:
        return []
    match = " OR ".join(f'"{term}"' for term in terms)
    rows = conn.execute(
        "SELECT s.id FROM symbols_fts JOIN symbols s ON s.id = symbols_fts.rowid "
        "WHERE symbols_fts MATCH ? AND s.repo = ? ORDER BY rank LIMIT ?",
        (match, repo, limit),
    ).fetchall()
    return [row[0] for row in rows]


def vec_search(conn, repo: str, vector, limit: int = 50) -> list[int]:
    rows = conn.execute(
        "SELECT symbol_id FROM symbols_vec WHERE repo = ? AND embedding MATCH ? AND k = ?",
        (repo, vector, limit),
    ).fetchall()
    return [row[0] for row in rows]


def search(  # noqa: PLR0913, PLR0917 - retrieval pipeline takes explicit per-stage collaborators
    conn,
    repo: str,
    query: str,
    k: int = 10,
    expand_fn=None,
    embed_fn=None,
    rerank_fn=None,
) -> SearchReport:
    warnings: list[str] = []
    stage_seconds: dict[str, float] = {}
    started = time.perf_counter()
    queries = _expand_queries(query, expand_fn, warnings)
    stage_seconds["expand"] = time.perf_counter() - started
    started = time.perf_counter()
    query_vectors = _embed_queries(queries, embed_fn, warnings)
    stage_seconds["embed"] = time.perf_counter() - started
    if conn.in_transaction:
        conn.rollback()
    conn.execute("BEGIN")
    try:
        started = time.perf_counter()
        fts_rankings = [fts_search(conn, repo, q) for q in queries]
        vec_rankings = [vec_search(conn, repo, vector) for vector in query_vectors]
        scores = rrf([*fts_rankings, *vec_rankings])
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : max(CANDIDATE_LIMIT, k)
        ]
        rows = _fetch(conn, [symbol_id for symbol_id, _ in top], repo)
        head, indexed_at = _repo_meta(conn, repo)
        symbol_count = conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo = ?", (repo,)
        ).fetchone()[0]
        parse_coverage = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM parse_state WHERE repo = ? GROUP BY status",
                (repo,),
            ).fetchall()
        )
        stage_seconds["retrieve"] = time.perf_counter() - started
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    pool: list[tuple[int, Candidate]] = []
    for symbol_id, score in top:
        row = rows.get(symbol_id)
        if row is None:
            continue
        pool.append(
            (
                symbol_id,
                Candidate(
                    path=row[1],
                    name=row[2],
                    qualname=row[3],
                    kind=row[4],
                    lang=row[5],
                    line=row[6],
                    end_line=row[7],
                    signature=row[8],
                    full_signature=row[9],
                    docstring=row[10],
                    score=score,
                ),
            )
        )
    started = time.perf_counter()
    candidates, rerank_indices = _apply_rerank(
        [candidate for _symbol_id, candidate in pool], query, rerank_fn, warnings
    )
    stage_seconds["rerank"] = time.perf_counter() - started
    rerank_order = (
        [pool[index][0] for index in rerank_indices] if rerank_indices is not None else None
    )
    current_head = _git(Path(repo), "rev-parse", "HEAD")
    return SearchReport(
        candidates=candidates[:k],
        warnings=warnings,
        symbol_count=symbol_count,
        head=head,
        age_seconds=time.time() - indexed_at if indexed_at else None,
        repo=repo,
        current_head=current_head,
        trace=SearchTrace(
            queries=queries,
            stage_seconds=stage_seconds,
            fts_rankings=fts_rankings,
            vec_rankings=vec_rankings,
            fused=list(top),
            rerank_order=rerank_order,
        ),
        parse_coverage=parse_coverage,
    )


def _expand_queries(query: str, expand_fn, warnings: list[str]) -> list[str]:
    queries = [query]
    if expand_fn is None:
        return queries
    expanded, warning = expand_fn(query)
    if warning:
        warnings.append(warning)
    for extra in expanded or []:
        if extra and extra not in queries:
            queries.append(extra)
    return queries


def _embed_queries(queries: list[str], embed_fn, warnings: list[str]) -> list[bytes]:
    if embed_fn is None:
        warnings.append("dense search skipped: embedding endpoint is not configured")
        return []
    vectors = []
    for query in queries:
        result, warning = embed_fn([query], query=True)
        if result is None:
            warnings.append(warning or "dense search failed")
            continue
        vectors.append(result[0])
    return vectors


def _rerank_document(candidate: Candidate) -> str:
    header = f"{candidate.path} :: {candidate.qualname} ({candidate.kind})"
    return f"{header}\n{candidate.full_signature}\n{candidate.docstring}"


def _apply_rerank(candidates, query, rerank_fn, warnings) -> tuple[list, list[int] | None]:
    if rerank_fn is None or not candidates:
        return candidates, None
    documents = [_rerank_document(candidate) for candidate in candidates]
    order, warning = rerank_fn(query, documents)
    if warning:
        warnings.append(warning)
    if _valid_rerank(order, len(documents)):
        positions = dict(order)
        ordered = sorted(positions, key=lambda idx: positions[idx], reverse=True)
        return [replace(candidates[idx], score=positions[idx]) for idx in ordered], ordered
    if order:
        warnings.append(
            f"rerank order invalid or incomplete ({len(order)}/{len(documents)} "
            "candidates); kept hybrid order"
        )
    return candidates, None


def _valid_rerank(order, n: int) -> bool:
    if not isinstance(order, list) or len(order) != n:
        return False
    seen: set[int] = set()
    return all(_valid_rerank_item(item, n, seen) for item in order)


def _valid_rerank_item(item, n: int, seen: set[int]) -> bool:
    if not isinstance(item, tuple) or len(item) != 2:
        return False
    idx, score = item
    if not isinstance(idx, int) or isinstance(idx, bool):
        return False
    if not 0 <= idx < n or idx in seen:
        return False
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    seen.add(idx)
    return math.isfinite(score)


def format_report(report: SearchReport) -> str:
    lines = [_index_line(report)]
    lines.extend(f"warning: {warning}" for warning in report.warnings)
    if not report.candidates:
        lines.append(
            "No matches. The index may be stale or empty: run refresh_index or "
            "'priorart index <repo>'."
        )
    for position, candidate in enumerate(report.candidates, 1):
        lines.append(
            f"{position}. {candidate.path}:{candidate.line} {candidate.kind} `{candidate.qualname}`"
        )
        if candidate.signature:
            lines.append(f"   {candidate.signature.strip()[:160]}")
        if candidate.docstring:
            lines.append(f"   {candidate.docstring.splitlines()[0][:160]}")
    lines.append(
        "Results are hints: read the file at the referenced location before reusing or extending."
    )
    return "\n".join(lines)


def status_text(conn, repo: str, *, dense: bool = True) -> str:
    root = Path(repo)
    head, indexed_at = _repo_meta(conn, repo)
    symbol_count = conn.execute("SELECT COUNT(*) FROM symbols WHERE repo = ?", (repo,)).fetchone()[
        0
    ]
    file_count = conn.execute("SELECT COUNT(*) FROM files WHERE repo = ?", (repo,)).fetchone()[0]
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (repo,)
    ).fetchone()[0]
    parse_counts = dict(
        conn.execute(
            "SELECT status, COUNT(*) FROM parse_state WHERE repo = ? GROUP BY status", (repo,)
        ).fetchall()
    )
    current_head = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain") is not None
    reasons: list[str] = []
    if head is None and symbol_count == 0:
        reasons.append("not indexed")
    else:
        if current_head and head != current_head:
            reasons.append("HEAD moved since indexing")
        gone, changed = _file_drift(conn, repo, root)
        if gone:
            reasons.append(f"{gone} indexed files no longer present")
        if changed:
            reasons.append(f"{changed} files changed since indexing")
    if dense and vectorized < symbol_count:
        reasons.append(f"{symbol_count - vectorized} symbols without vectors")
    lines = [
        f"repo: {repo}",
        f"index_head: {head or '-'} | current_head: {current_head or '-'}",
        f"working_tree_dirty: {'yes' if dirty else 'no'}",
        f"stale: {'; '.join(reasons) if reasons else 'no'}",
        f"files: {file_count}, symbols: {symbol_count}, vectorized: {vectorized}/{symbol_count}",
    ]
    if parse_counts:
        lines.append(
            "parse: "
            + ", ".join(f"{count} {status}" for status, count in sorted(parse_counts.items()))
        )
    if indexed_at:
        lines.append(f"indexed: {_age(time.time() - indexed_at)} ago")
    return "\n".join(lines)


def map_symbols_text(conn, repo: str, path_glob: str, limit: int = 200) -> str:
    pattern = (
        path_glob.replace(".", "\\.")
        .replace("**", "\x00")
        .replace("*", "%")
        .replace("\x00", "%")
        .replace("?", "_")
    )
    rows = conn.execute(
        "SELECT path, kind, qualname, line FROM symbols "
        "WHERE repo = ? AND path LIKE ? ESCAPE '\\' ORDER BY path, line LIMIT ?",
        (repo, pattern, limit),
    ).fetchall()
    if not rows:
        return f"No indexed symbols match {path_glob!r}."
    lines = [f"{path}:{line} {kind} {qualname}" for path, kind, qualname, line in rows]
    if len(rows) == limit:
        lines.append(f"(truncated at {limit} rows)")
    return "\n".join(lines)


def _file_drift(conn, repo: str, root: Path) -> tuple[int, int]:
    expected = set(_list_files(root))
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


def _index_line(report: SearchReport) -> str:
    parts = [f"Index: {report.symbol_count} symbols"]
    if report.repo:
        parts.append(f"repo {report.repo}")
    if report.head:
        parts.append(f"HEAD {report.head[:8]}")
    if report.head and report.current_head and report.head != report.current_head:
        parts.append("stale: HEAD moved")
    if report.age_seconds is not None:
        parts.append(f"indexed {_age(report.age_seconds)} ago")
    flagged = {
        status: count
        for status, count in (report.parse_coverage or {}).items()
        if status != "ok" and count
    }
    if flagged:
        parts.append(
            "parse issues: "
            + ", ".join(f"{count} {status}" for status, count in sorted(flagged.items()))
        )
    return ", ".join(parts)


def _age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _repo_meta(conn, repo: str) -> tuple[str | None, float | None]:
    row = conn.execute("SELECT head, indexed_at FROM repos WHERE repo = ?", (repo,)).fetchone()
    return (row[0], row[1]) if row else (None, None)


_COLUMNS = (
    "id, path, name, qualname, kind, lang, line, end_line, signature, full_signature, docstring"
)


def _fetch(conn, ids: list[int], repo: str) -> dict[int, tuple]:
    rows = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT {_COLUMNS} FROM symbols WHERE id IN ({marks}) AND repo = ?",  # noqa: S608 - marks are placeholders only
            (*chunk, repo),
        ):
            rows[row[0]] = row
    return rows
