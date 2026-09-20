"""Retrieval pipeline over the published index.

Contract kept from the previous design: retrieval reads candidates, bodies
and metadata inside one read transaction, materializes the result and closes
the transaction before any model call (rerank) happens. Every caller owns
its connection, so no shared-connection rollback hack is needed or allowed.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from priorart.core import bounded_body, git_output
from priorart.models import input_hash, set_deadline

from .status import age, repo_meta


@contextlib.contextmanager
def _budgeted_stage(stage_seconds: dict, name: str, deadline: float | None):
    """One timed pipeline stage under the shared monotonic deadline.

    The deadline is thread-local state: this wrapper is the only arming
    point, so a stage can never forget to clear it on an exception path.
    """
    started = time.perf_counter()
    set_deadline(deadline)
    try:
        yield
    finally:
        set_deadline(None)
        stage_seconds[name] = time.perf_counter() - started


CANDIDATE_LIMIT = 50
RRF_K = 60
RERANK_DOCUMENT_FORMAT = "path-qualname-kind-signature-docstring-body-v1"
BODY_MAX_CHARS = 3000
EXPANSION_LIMIT = 50
EXPANSION_FILE_QUOTA = 3

DENSE_INDEX_PARTIAL = "DENSE_INDEX_PARTIAL"
PARSE_COVERAGE_PARTIAL = "PARSE_COVERAGE_PARTIAL"
QUERY_MODEL_UNAVAILABLE = "QUERY_MODEL_UNAVAILABLE"
RERANK_FALLBACK = "RERANK_FALLBACK"
DEADLINE_FALLBACK = "DEADLINE_FALLBACK"

# "empty" is a clean parse of a symbol-free file, not a problem; "embed_failed"
# leaves stale symbols behind and must stay visible.
_PARSE_ISSUE_STATUSES = frozenset({"partial", "error", "unsupported", "unreadable"})


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
    body: str
    source_role: str
    score: float

    def payload(self) -> dict:
        """Wire form without the body: adapters never rerank client-side."""
        return {
            "path": self.path,
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "lang": self.lang,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "full_signature": self.full_signature,
            "docstring": self.docstring,
            "source_role": self.source_role,
            "score": self.score,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> Candidate:
        return cls(
            path=payload["path"],
            name=payload["name"],
            qualname=payload["qualname"],
            kind=payload["kind"],
            lang=payload["lang"],
            line=payload["line"],
            end_line=payload["end_line"],
            signature=payload["signature"],
            full_signature=payload["full_signature"],
            docstring=payload["docstring"],
            body="",
            source_role=payload["source_role"],
            score=payload["score"],
        )


_COLUMNS = (
    "id, path, name, qualname, kind, lang, line, end_line, signature, full_signature, "
    "docstring, body, source_role"
)
_ROLE_COLUMN = 12


@dataclass
class SearchTrace:
    """Per-stage record of one search.

    ``pool_expansion`` holds the symbol ids that pool expansion appended
    after the fused top (file-to-owner candidates), not the query-expansion
    variants listed in ``queries``.
    """

    queries: list[str]
    stage_seconds: dict[str, float]
    fts_rankings: list[list[int]]
    vec_rankings: list[list[int]]
    fused: list[tuple[int, float]]
    rerank_order: list[int] | None
    pool_expansion: list[int] | None = None


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
    pool: list[Candidate] | None = None
    epoch: int | None = None
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    stages_used: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        """Wire form for adapters: candidates without bodies, trace timings."""
        return {
            "candidates": [candidate.payload() for candidate in self.candidates],
            "warnings": self.warnings,
            "symbol_count": self.symbol_count,
            "head": self.head,
            "age_seconds": self.age_seconds,
            "repo": self.repo,
            "current_head": self.current_head,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "stages_used": self.stages_used,
            "epoch": self.epoch,
            "parse_coverage": self.parse_coverage,
            "stage_seconds": self.trace.stage_seconds if self.trace else {},
        }

    @classmethod
    def from_payload(cls, payload: dict) -> SearchReport:
        candidates = [Candidate.from_payload(item) for item in payload.get("candidates", [])]
        # the untruncated pool and candidate bodies stay with the producer;
        # None keeps consumers honest instead of handing them k-truncated data
        return cls(
            candidates=candidates,
            pool=None,
            warnings=payload.get("warnings", []),
            symbol_count=payload.get("symbol_count", 0),
            head=payload.get("head"),
            age_seconds=payload.get("age_seconds"),
            repo=payload.get("repo"),
            current_head=payload.get("current_head"),
            trace=SearchTrace(
                queries=[],
                stage_seconds=payload.get("stage_seconds", {}),
                fts_rankings=[],
                vec_rankings=[],
                fused=[],
                rerank_order=None,
                pool_expansion=[],
            ),
            parse_coverage=payload.get("parse_coverage"),
            epoch=payload.get("epoch"),
            degraded=payload.get("degraded", False),
            degradation_reasons=payload.get("degradation_reasons", []),
            stages_used=payload.get("stages_used", []),
        )


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


def search(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 - explicit retrieval stages
    conn,
    repo: str,
    query: str,
    k: int = 10,
    expand_fn=None,
    embed_fn=None,
    rerank_fn=None,
    *,
    pool_expansion: bool = True,
    candidate_limit: int | None = None,
    query_cache=None,
    query_space_id: str = "",
    intent: str = "implementation",
    exact_dispatch: bool = True,
    deadline_seconds: float | None = None,
    dense_expected: bool = True,
) -> SearchReport:
    """Run the retrieval pipeline; ``k`` only truncates the returned candidates.

    The rerank pool is independent of ``k``: fused retrieval always takes
    ``candidate_limit`` symbols (CANDIDATE_LIMIT by default) plus up to
    EXPANSION_LIMIT pool-expansion owners, so the same profile does the same
    model work for any output depth. The untruncated reranked pool is
    available as ``report.pool``.

    ``intent`` selects the source-role contract: ``implementation`` prefers
    production code at comparable relevance, ``tests`` restricts the pool to
    test/fixture sources, ``any`` is neutral. ``exact_dispatch`` answers a
    query that exactly matches a symbol name or qualname without any model
    calls; ``expand_fn`` enables the generative (deep) expansion — without
    it the pipeline embeds only the raw query.
    """
    limit = CANDIDATE_LIMIT if candidate_limit is None else candidate_limit
    warnings: list[str] = []
    degradation_reasons: list[str] = []
    stage_seconds: dict[str, float] = {}
    started = time.perf_counter()
    if exact_dispatch:
        exact = _exact_match(conn, repo, query)
        if intent == "tests":
            # the tests intent must not bypass the role restriction via an
            # exact production-symbol hit
            exact = [row for row in exact if row[_ROLE_COLUMN] in ("test", "fixture")]
        if exact:
            stage_seconds["exact"] = time.perf_counter() - started
            return _exact_report(
                conn, repo, query, k, exact, stage_seconds, warnings, intent=intent
            )
    deadline = None if deadline_seconds is None else time.monotonic() + deadline_seconds
    # the deadline covers generative expansion too, not only embedding and
    # reranking: a slow expander must not overrun the budget
    with _budgeted_stage(stage_seconds, "expand", deadline):
        queries = _expand_queries(query, expand_fn, warnings)
    # without generative expansion only the raw query is embedded; the cheap
    # lexical variants serve the FTS channel alone
    embedded_queries = queries if expand_fn is not None else queries[:1]
    with _budgeted_stage(stage_seconds, "embed", deadline):
        query_vectors = _embed_queries(
            embedded_queries,
            embed_fn,
            warnings,
            query_cache,
            query_space_id,
            dense_expected=dense_expected,
        )
    # A budget timeout can return no vectors, so classification must use the
    # monotonic deadline rather than the result or provider warning text.
    dense_skipped_for_deadline = deadline is not None and time.monotonic() >= deadline
    if dense_skipped_for_deadline:
        if query_vectors:
            warnings.append("search deadline exhausted before retrieval; dense channel skipped")
        degradation_reasons.append(DEADLINE_FALLBACK)
        query_vectors = []
    started = time.perf_counter()
    retrieval = _retrieve(conn, repo, query, queries, query_vectors, pool_expansion, limit)
    stage_seconds["retrieve"] = time.perf_counter() - started
    if query_vectors and retrieval.symbol_count and not retrieval.vectorized_count:
        warnings.append(
            "dense channel is empty: indexed symbols have no vectors "
            "(index built without embeddings or with a different embedding "
            "profile); rerun priorart index with embeddings configured"
        )
    if dense_expected and not query_vectors and not dense_skipped_for_deadline:
        degradation_reasons.append(QUERY_MODEL_UNAVAILABLE)
    if dense_expected and retrieval.vectorized_count < retrieval.symbol_count:
        degradation_reasons.append(DENSE_INDEX_PARTIAL)
    if any(retrieval.parse_coverage.get(status, 0) for status in _PARSE_ISSUE_STATUSES):
        degradation_reasons.append(PARSE_COVERAGE_PARTIAL)
    pool: list[tuple[int, Candidate]] = []
    for symbol_id, score in retrieval.top:
        row = retrieval.rows.get(symbol_id)
        if row is None:
            continue
        pool.append((symbol_id, _candidate_from_row(row, score)))
    pool.extend(
        (symbol_id, _candidate_from_row(row, 0.0)) for symbol_id, row in retrieval.expansion
    )
    if intent == "tests":
        # explicit test intent restricts the pool; production preference
        # is a ranking prior instead of a filter
        pool = [item for item in pool if item[1].source_role in ("test", "fixture")]
    rerank_input = [candidate for _symbol_id, candidate in pool]
    with _budgeted_stage(stage_seconds, "rerank", deadline):
        candidates, rerank_indices = _apply_rerank(rerank_input, query, rerank_fn, warnings)
    rerank_skipped_for_deadline = (
        rerank_fn is not None
        and bool(pool)
        and deadline is not None
        and time.monotonic() >= deadline
    )
    if rerank_skipped_for_deadline:
        warnings.append("search deadline exhausted during reranking; kept deterministic order")
        candidates = _fallback_rank(rerank_input, query)
        rerank_indices = None
        degradation_reasons.append(DEADLINE_FALLBACK)
    if rerank_fn is not None and pool and rerank_indices is None:
        degradation_reasons.append(RERANK_FALLBACK)
    degradation_reasons = list(dict.fromkeys(degradation_reasons))
    if intent == "implementation":
        candidates = _prefer_production(candidates)
    rerank_order = (
        [pool[index][0] for index in rerank_indices] if rerank_indices is not None else None
    )
    current_head = observed_head(repo)
    return SearchReport(
        candidates=candidates[:k],
        warnings=warnings,
        symbol_count=retrieval.symbol_count,
        head=retrieval.head,
        age_seconds=time.time() - retrieval.indexed_at if retrieval.indexed_at else None,
        repo=repo,
        current_head=current_head,
        trace=SearchTrace(
            queries=queries,
            stage_seconds=stage_seconds,
            fts_rankings=retrieval.fts_rankings,
            vec_rankings=retrieval.vec_rankings,
            fused=list(retrieval.top),
            rerank_order=rerank_order,
            pool_expansion=[symbol_id for symbol_id, _ in retrieval.expansion],
        ),
        parse_coverage=retrieval.parse_coverage,
        pool=candidates,
        epoch=retrieval.epoch,
        degraded=bool(degradation_reasons),
        degradation_reasons=degradation_reasons,
        stages_used=["lexical", *(["dense"] if query_vectors else [])],
    )


@dataclass
class _Retrieval:
    """Snapshot read for one search, produced inside a single transaction."""

    top: list[tuple[int, float]]
    rows: dict[int, tuple]
    expansion: list[tuple[int, tuple]]
    fts_rankings: list[list[int]]
    vec_rankings: list[list[int]]
    head: str | None
    indexed_at: float | None
    epoch: int | None
    symbol_count: int
    vectorized_count: int
    parse_coverage: dict[str, int]


def _retrieve(  # noqa: PLR0913, PLR0917 - retrieval pipeline takes explicit per-stage collaborators
    conn, repo, query, queries, query_vectors, pool_expansion, limit
) -> _Retrieval:
    if conn.in_transaction:
        raise RuntimeError("search requires a connection with no open transaction")
    conn.execute("BEGIN")
    try:
        fts_rankings = [fts_search(conn, repo, q, limit) for q in queries]
        vec_rankings = [vec_search(conn, repo, vector, limit) for vector in query_vectors]
        scores = rrf([*fts_rankings, *vec_rankings])
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        rows = _fetch(conn, [symbol_id for symbol_id, _ in top], repo)
        expansion = (
            _expand_pool(conn, repo, query, query_vectors[0] if query_vectors else None, top)
            if pool_expansion
            else []
        )
        head, indexed_at, epoch = repo_meta(conn, repo)
        symbol_count = _symbol_count(conn, repo)
        vectorized_count = conn.execute(
            "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (repo,)
        ).fetchone()[0]
        parse_coverage = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM parse_state WHERE repo = ? GROUP BY status",
                (repo,),
            ).fetchall()
        )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return _Retrieval(
        top=top,
        rows=rows,
        expansion=expansion,
        fts_rankings=fts_rankings,
        vec_rankings=vec_rankings,
        head=head,
        indexed_at=indexed_at,
        epoch=epoch,
        symbol_count=symbol_count,
        vectorized_count=vectorized_count,
        parse_coverage=parse_coverage,
    )


_MAX_EXPANDED_QUERIES = 8


def _expand_queries(query: str, expand_fn, warnings: list[str]) -> list[str]:
    queries = [query]
    for variant in _lexical_variants(query):
        if variant not in queries:
            queries.append(variant)
    if expand_fn is None:
        return queries
    expanded, warning = expand_fn(query)
    if warning:
        warnings.append(warning)
    for extra in expanded or []:
        if extra and extra not in queries:
            queries.append(extra)
    if len(queries) > _MAX_EXPANDED_QUERIES:
        # the expander's "5 to 8" prompt is not a contract; the pipeline
        # enforces the bound itself so latency cannot be driven by output
        # length
        queries = queries[:_MAX_EXPANDED_QUERIES]
    return queries


_CAMEL_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def _lexical_variants(query: str) -> list[str]:
    """Cheap deterministic variants of the raw query for the FTS channel.

    snake_case/CamelCase compounds split into their spoken form so lexical
    retrieval does not depend on exact identifier spelling.
    """
    variants: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query):
        parts = _CAMEL_SPLIT.findall(token.replace("_", " "))
        if len(parts) > 1:
            spoken = " ".join(parts)
            if spoken.lower() != token.lower():
                variants.append(spoken)
    return variants[:8]


def _exact_match(conn, repo: str, query: str) -> list[tuple]:
    """Rows whose name or qualname equals the whole query, bounded.

    A fuzzy intent must not fall into the exact path through one shared word:
    only a full-string identifier match qualifies.
    """
    stripped = query.strip()
    if not stripped or len(stripped) > 200 or not re.fullmatch(r"[\w.]+", stripped):
        return []
    return conn.execute(
        f"SELECT {_COLUMNS} FROM symbols "  # noqa: S608 - no user input in the statement
        "WHERE repo = ? AND (qualname = ? OR name = ?) "
        "ORDER BY length(qualname), path, line LIMIT 10",
        (repo, stripped, stripped),
    ).fetchall()


def _exact_report(  # noqa: PLR0913, PLR0917 - report construction takes the full context
    conn, repo, query, k, rows, stage_seconds, warnings, *, intent
) -> SearchReport:
    candidates = [_candidate_from_row(row, 1.0) for row in rows]
    if intent == "implementation":
        candidates = _prefer_production(candidates)
    head, indexed_at, epoch = repo_meta(conn, repo)
    symbol_count = _symbol_count(conn, repo)
    return SearchReport(
        candidates=candidates[:k],
        pool=candidates,
        warnings=warnings,
        symbol_count=symbol_count,
        head=head,
        age_seconds=time.time() - indexed_at if indexed_at else None,
        repo=repo,
        current_head=observed_head(repo),
        trace=SearchTrace(
            queries=[query],
            stage_seconds=stage_seconds,
            fts_rankings=[],
            vec_rankings=[],
            fused=[],
            rerank_order=None,
            pool_expansion=[],
        ),
        parse_coverage={},
        epoch=epoch,
        degraded=False,
        degradation_reasons=[],
        stages_used=["exact"],
    )


_ROLE_PRIOR = 0.01


def _prefer_production(candidates: list[Candidate]) -> list[Candidate]:
    """Tiny stable production prior: reorders only comparable relevance."""
    return sorted(
        candidates,
        key=lambda candidate: (
            -(candidate.score + (_ROLE_PRIOR if candidate.source_role == "production" else 0.0)),
        ),
    )


def _embed_queries(  # noqa: PLR0913 - pipeline stage takes its explicit collaborators
    queries: list[str],
    embed_fn,
    warnings: list[str],
    query_cache=None,
    query_space_id: str = "",
    *,
    dense_expected: bool = True,
) -> list[bytes]:
    """Embed every query variant through its own cache identity.

    Query variants (raw, expanded) hash to different cache keys, so a failed
    raw embedding can never be answered with another variant's vector.
    """
    if embed_fn is None:
        if dense_expected:
            warnings.append("dense search skipped: embedding endpoint is not configured")
        return []
    cached: dict[str, bytes] = {}
    if query_cache is not None and query_space_id:
        hits = query_cache.lookup(query_space_id, "query", queries)
        cached = {query: hits[input_hash(query)] for query in queries if input_hash(query) in hits}
    misses = [query for query in dict.fromkeys(queries) if query not in cached]
    computed = dict(zip(misses, _embed_batched(misses, embed_fn, warnings), strict=False))
    if query_cache is not None and query_space_id and computed:
        query_cache.store(
            query_space_id,
            "query",
            [(input_hash(query), vector) for query, vector in computed.items()],
            dim=len(next(iter(computed.values()))) // 4,
        )
    return [
        vector
        for query in queries
        if (vector := cached.get(query) or computed.get(query)) is not None
    ]


def _embed_batched(queries: list[str], embed_fn, warnings: list[str]) -> list[bytes]:
    """Embed the uncached queries in order; a failed call stops the batch run.

    Stopping (rather than skipping) keeps every returned vector aligned
    with its query: a skip would silently shift the remaining vectors onto
    the wrong queries and poison the query cache with them.
    """
    vectors: list[bytes] = []
    for start in range(0, len(queries), 32):
        batch = queries[start : start + 32]
        result, warning = embed_fn(batch, query=True)
        if result is None:
            warnings.append(warning or "dense search failed")
            break
        vectors.extend(result)
    return vectors


def rerank_document(candidate: Candidate, *, body_max_chars: int = BODY_MAX_CHARS) -> str:
    """Render one candidate as a reranker document.

    The single source of truth for the current ``RERANK_DOCUMENT_FORMAT``;
    benchmarks replay this exact builder instead of keeping a copy.
    """
    header = f"{candidate.path} :: {candidate.qualname} ({candidate.kind})"
    body = bounded_body(candidate.body, body_max_chars)
    return f"{header}\n{candidate.full_signature}\n{candidate.docstring}\n{body}"


def _candidate_from_row(row: tuple, score: float) -> Candidate:
    return Candidate(
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
        body=row[11],
        source_role=row[12],
        score=score,
    )


def _expand_pool(conn, repo, query, query_vector, top) -> list[tuple[int, tuple]]:
    """Add suitable owners from files the fused ranking already found.

    A file with symbols in the candidate pool is "found"; its remaining
    definitions can still be the queried owner even when they never made any
    per-stage top list. Per file, at most EXPANSION_FILE_QUOTA symbols are
    added, ranked by distinct query terms matched in the body (prefix,
    case-insensitive) with cosine similarity to the query as tie-breaker;
    files are considered in order of their best fused score until
    EXPANSION_LIMIT symbols are added in total.
    """
    pool_ids = {symbol_id for symbol_id, _score in top}
    terms = _query_terms(query)
    files = _file_order(conn, dict(top))
    expanded: list[tuple[int, tuple]] = []
    seen: set[int] = set()
    for path in files:
        if len(expanded) >= EXPANSION_LIMIT:
            break
        quota = min(EXPANSION_FILE_QUOTA, EXPANSION_LIMIT - len(expanded))
        for symbol_id, row in _file_expansions(
            conn, repo, path, pool_ids | seen, terms, query_vector
        )[:quota]:
            expanded.append((symbol_id, row))
            seen.add(symbol_id)
    return expanded


def _chunks(ids: list, size: int = 500) -> Iterator[list]:
    for start in range(0, len(ids), size):
        yield ids[start : start + size]


def _file_order(conn, pool) -> list[str]:
    """Files of the pooled symbols, strongest symbol first."""
    paths: dict[int, str] = {}
    symbol_ids = list(pool)
    for chunk in _chunks(symbol_ids):
        marks = ",".join("?" * len(chunk))
        paths.update(
            dict(
                conn.execute(
                    f"SELECT id, path FROM symbols WHERE id IN ({marks})",  # noqa: S608 - marks are placeholders only
                    chunk,
                )
            )
        )
    best: dict[str, float] = {}
    for symbol_id, score in pool.items():
        path = paths.get(symbol_id)
        if path is not None and score > best.get(path, 0.0):
            best[path] = score
    return [path for path, _score in sorted(best.items(), key=lambda item: item[1], reverse=True)]


def _file_expansions(  # noqa: PLR0913, PLR0917 - same explicit-collaborator shape as search()
    conn, repo, path, exclude, terms, query_vector
) -> list[tuple[int, tuple]]:
    rows = [
        row
        for row in conn.execute(
            f"SELECT {_COLUMNS} FROM symbols WHERE repo = ? AND path = ?",  # noqa: S608 - _COLUMNS is a fixed column list
            (repo, path),
        )
        if row[0] not in exclude
    ]
    if not rows:
        return []
    embeddings = _embeddings(conn, [row[0] for row in rows]) if query_vector is not None else {}
    ranked = sorted(
        rows,
        key=lambda row: (
            _term_hits(row[11], terms),
            _cosine(embeddings.get(row[0]), query_vector),
        ),
        reverse=True,
    )
    return [(row[0], row) for row in ranked]


def _query_terms(query: str) -> list[str]:
    terms: dict[str, None] = {}
    for term in re.findall(r"\w{3,}", query.lower()):
        terms.setdefault(term)
    return list(terms)


def _term_hits(body: str | None, terms: list[str]) -> int:
    if not body or not terms:
        return 0
    low = body.lower()
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}", low))


def _embeddings(conn, symbol_ids: list[int]) -> dict[int, bytes]:
    if not symbol_ids:
        return {}
    marks = ",".join("?" * len(symbol_ids))
    return {
        row[0]: row[1]
        for row in conn.execute(
            f"SELECT symbol_id, embedding FROM symbols_vec WHERE symbol_id IN ({marks})",  # noqa: S608 - marks are placeholders only
            symbol_ids,
        )
    }


def _cosine(embedding: bytes | None, query_vector: bytes | None) -> float:
    if embedding is None or query_vector is None:
        return 0.0
    left = array("f")
    left.frombytes(embedding)
    right = array("f")
    right.frombytes(query_vector)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if not math.isfinite(dot) or not math.isfinite(norm) or norm == 0.0:
        return 0.0
    return dot / norm


def _apply_rerank(candidates, query, rerank_fn, warnings) -> tuple[list, list[int] | None]:
    if not candidates:
        return candidates, None
    if rerank_fn is not None:
        documents = [rerank_document(candidate) for candidate in candidates]
        order, warning = rerank_fn(query, documents)
        if warning:
            warnings.append(warning)
        ordered = rerank_positions(order, len(documents))
        if ordered is not None:
            positions = dict(order)
            return [replace(candidates[idx], score=positions[idx]) for idx in ordered], ordered
        if order:
            warnings.append(
                f"rerank order invalid or incomplete ({len(order)}/{len(documents)} "
                "candidates); kept hybrid order"
            )
    # R03: without a usable reranker the pool is ranked by deterministic
    # evidence, and pool-expansion owners take part on equal terms instead
    # of sitting at the tail with score 0
    return _fallback_rank(candidates, query), None


_QUERY_TERMS = re.compile(r"\w{2,}")


def _fallback_rank(candidates: list[Candidate], query: str) -> list[Candidate]:
    """Deterministic ranking used when the reranker is unavailable."""
    terms = [term.lower() for term in _QUERY_TERMS.findall(query)]
    scored = []
    for candidate in candidates:
        qualname = candidate.qualname.lower()
        path = candidate.path.lower()
        body = candidate.body.lower()
        score = 0.0
        for term in terms:
            if term == qualname:
                score += 3.0
            elif term in qualname:
                score += 2.0
            if term in path:
                score += 0.5
            score += min(body.count(term), 3) * 0.25
        if candidate.source_role == "production":
            score += _ROLE_PRIOR
        scored.append(replace(candidate, score=score))
    return sorted(scored, key=lambda candidate: candidate.score, reverse=True)


def rerank_positions(order, n: int) -> list[int] | None:
    """Validated descending-position view of a rerank order, or ``None``.

    Shared by the live pipeline and benchmark replay so both apply rerank
    results through the same validation and tie-break rules.
    """
    if not valid_rerank_order(order, n):
        return None
    positions = dict(order)
    return sorted(positions, key=lambda idx: positions[idx], reverse=True)


def valid_rerank_order(order, n: int) -> bool:
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


def map_symbols_rows(
    conn, repo: str, path_glob: str, limit: int = 200, offset: int = 0
) -> tuple[list[dict], int | None]:
    """One page of indexed symbols and the cursor for the next page.

    Rows are dicts: the shape survives the daemon wire and every consumer
    by name, not by SELECT-column position.
    """
    pattern = _glob_pattern(path_glob)
    rows = [
        {"path": path, "kind": kind, "qualname": qualname, "line": line}
        for path, kind, qualname, line in conn.execute(
            "SELECT path, kind, qualname, line FROM symbols "
            "WHERE repo = ? AND path LIKE ? ESCAPE '\\' ORDER BY path, line LIMIT ? OFFSET ?",
            (repo, pattern, limit, offset),
        )
    ]
    # an exactly full page is not proof of a next one: probe instead of
    # advertising a phantom cursor
    more = None
    if len(rows) == limit:
        probe = conn.execute(
            "SELECT 1 FROM symbols "
            "WHERE repo = ? AND path LIKE ? ESCAPE '\\' ORDER BY path, line "
            "LIMIT 1 OFFSET ?",
            (repo, pattern, offset + limit),
        ).fetchone()
        more = offset + limit if probe is not None else None
    return rows, more


def _glob_pattern(path_glob: str) -> str:
    # literal %, _ and \ in the input are SQL wildcards to LIKE and must be
    # escaped before the glob wildcards are substituted in
    escaped = path_glob.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped.replace("**", "\x00").replace("*", "%").replace("\x00", "%").replace("?", "_")


def _index_line(report: SearchReport) -> str:
    parts = [f"Index: {report.symbol_count} symbols"]
    if report.repo:
        parts.append(f"repo {report.repo}")
    if report.head:
        parts.append(f"HEAD {report.head[:8]}")
    if report.head and report.current_head and report.head != report.current_head:
        parts.append("stale: HEAD moved")
    if report.age_seconds is not None:
        parts.append(f"indexed {age(report.age_seconds)} ago")
    flagged = {
        status: count
        for status, count in (report.parse_coverage or {}).items()
        if status in _PARSE_ISSUE_STATUSES and count
    }
    if flagged:
        parts.append(
            "parse issues: "
            + ", ".join(f"{count} {status}" for status, count in sorted(flagged.items()))
        )
    return ", ".join(parts)


def _symbol_count(conn, repo: str) -> int:
    return conn.execute("SELECT COUNT(*) FROM symbols WHERE repo = ?", (repo,)).fetchone()[0]


def observed_head(repo: str) -> str | None:
    """Observed HEAD of the repo root, or ``None`` outside git."""
    return git_output(Path(repo), "rev-parse", "HEAD")


def _fetch(conn, ids: list[int], repo: str) -> dict[int, tuple]:
    rows = {}
    for chunk in _chunks(ids):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT {_COLUMNS} FROM symbols WHERE id IN ({marks}) AND repo = ?",  # noqa: S608 - marks are placeholders only
            (*chunk, repo),
        ):
            rows[row[0]] = row
    return rows
