"""Indexing pipeline: staged delta, atomic lexical publication, background
vector attachment.

One lexical refresh runs in five steps:

1. Inventory tracked + non-ignored untracked files (git is the only source
   of truth; a git failure is a diagnosable error).
2. Capture bytes of changed/suspect files, hash them and parse them into an
   in-memory staging area — no writes, no transactions.
3. Publish the whole lexical delta (drops, new symbols, FTS text, file
   state, parse diagnostics, HEAD observation, a new index epoch) in one
   short transaction. Until this commit search sees the previous published
   version; after it, the new one — possibly with incomplete dense coverage.
4. Attach vectors for symbols that lack them in guarded batches, outside
   any long transaction. Identical inputs are embedded once; the shared
   embedding cache answers inputs computed before, so a rebuild or a
   restart reuses earlier inference. Freshly computed vectors are stored
   in the cache immediately, so a failure of a later batch never loses
   earlier computations. A batch result is attached only when the symbol's
   embedding key and the file's content hash still match the capture, so a
   late result never lands on a newer version of the file.
5. Failures of the embedding endpoint degrade dense coverage but never
   block or roll back published lexical symbols.

No SQLite transaction is ever held across git, parsing, model calls or
locks, and no per-file commits exist.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from priorart.models import input_hash as _embed_key

from .capture import capture_file
from .inventory import head_revision, list_source_files
from .parser import language_of, parse_source
from .roles import source_role

EMBED_CHUNK = 64

ProgressFn = Callable[[str, dict], None]
CancelFn = Callable[[], bool]


class RefreshCancelledError(Exception):
    """The job was cancelled; published lexical state stays published."""


@dataclass
class _Staged:
    rel: str
    stat: object | None = None
    content_hash: str | None = None
    status: str = "unreadable"
    detail: str | None = None
    symbols: list = field(default_factory=list)


def index_repo(  # noqa: PLR0913 - explicit collaborator set
    conn,
    root: Path,
    embed_fn=None,
    *,
    rebuild: bool = False,
    paths: list[str] | None = None,
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    embed_cache=None,
    embed_space_id: str = "",
) -> dict:
    """Refresh the lexical index atomically, then attach missing vectors.

    Returns counters for the job journal. Raises ``RefreshCancelledError`` when
    ``should_cancel`` fires between units of work; everything committed
    before that stays published.
    """
    root = Path(root).resolve()
    repo = str(root)
    started = time.time()
    head = head_revision(root)
    _emit(progress, "scanning", {})
    rels = list_source_files(root)
    current = set(rels)
    scoped = _scoped_current(current, root, paths)
    known = {
        path: (mtime_ns, size)
        for path, mtime_ns, size in conn.execute(
            "SELECT path, mtime_ns, size FROM files WHERE repo = ?", (repo,)
        )
    }
    removed = sorted(path for path in known if path not in current)

    staged: list[_Staged] = []
    total = len(scoped) + len(removed)
    scanned = 0
    for rel in sorted(scoped):
        _check_cancelled(should_cancel)
        scanned += 1
        item = _stage_file(root, rel, known, rebuild=rebuild, force=paths is not None)
        if item is not None:
            staged.append(item)
        _emit(progress, "parsing", {"total": total, "scanned": scanned})

    files = 0
    symbols = 0
    warnings: list[str] = []
    _emit(progress, "publishing_lexical", {"total": total, "scanned": scanned})
    conn.execute("BEGIN IMMEDIATE")
    try:
        for rel in removed:
            _drop_file(conn, repo, rel)
        for item in staged:
            published = _publish_staged(conn, repo, item, warnings)
            files += 1 if published else 0
            symbols += len(item.symbols)
        epoch = _publish_epoch(conn, repo, head)
        conn.execute("COMMIT")
    except BaseException:
        _rollback(conn)
        raise

    embedded, cache_hits, embed_failures = _attach_missing_vectors(
        conn, repo, embed_fn, should_cancel, progress, warnings, embed_cache, embed_space_id
    )
    missing_vectors = 0
    if embed_fn is not None:
        missing_vectors = conn.execute(
            "SELECT COUNT(*) FROM symbols s LEFT JOIN symbols_vec v ON v.symbol_id = s.id "
            "WHERE s.repo = ? AND v.symbol_id IS NULL",
            (repo,),
        ).fetchone()[0]
    return {
        "files": files,
        "symbols": symbols,
        "removed": len(removed),
        "warnings": warnings,
        "epoch": epoch,
        "head": head,
        "embedded": embedded,
        "cache_hits": cache_hits,
        "embed_failures": embed_failures,
        "missing_vectors": missing_vectors,
        "elapsed_seconds": time.time() - started,
    }


def _scoped_current(current: set[str], root: Path, paths: list[str] | None) -> set[str]:
    """Restrict a refresh to ``paths`` without bypassing inventory validation."""
    if paths is None:
        return current
    root_posix = root.as_posix()
    wanted: set[str] = set()
    for raw in paths:
        pure = PurePosixPath(raw)
        if pure.is_absolute():
            try:
                rel = str(pure.relative_to(root_posix))
            except ValueError:
                continue
        else:
            rel = str(pure)
        if not rel.startswith(".."):
            wanted.add(rel)
    return current & wanted


def _stage_file(
    root: Path, rel: str, known: dict, *, rebuild: bool, force: bool = False
) -> _Staged | None:
    """Stage one file for publication, or ``None`` when nothing changed.

    The stat shortcut only serves unscoped scans: an explicitly requested
    path is always recaptured, so content changes that preserved size and
    mtime are still detected (F03).
    """
    try:
        st = (root / rel).stat()
    except OSError:
        return _Staged(rel, status="unreadable", detail="missing from the working tree")
    if not rebuild and not force and known.get(rel) == (st.st_mtime_ns, st.st_size):
        return None
    captured = capture_file(root, rel)
    if captured is None:
        return _Staged(
            rel, status="unreadable", detail="unreadable or changed while reading; will retry"
        )
    data, st = captured
    lang = language_of(rel)
    result = parse_source(data, lang, rel) if lang else None
    if result is None:
        return _Staged(rel, stat=st, status="unsupported", detail="unsupported file type")
    return _Staged(
        rel,
        stat=st,
        content_hash=hashlib.sha256(data).hexdigest(),
        status=result.status,
        detail=result.detail,
        symbols=result.symbols,
    )


def _publish_staged(conn, repo: str, item: _Staged, warnings: list[str]) -> bool:
    """Apply one staged file inside the open publish transaction.

    Returns whether the file's captured version became the published one.
    Parse failures keep the previous symbols as an explicitly stale
    fallback (the files row keeps the old stat, so the next refresh retries).
    """
    _record_parse_state(conn, repo, item.rel, item.status, item.detail)
    if item.status in ("unreadable", "unsupported", "error"):
        warnings.append(
            f"{item.rel}: {item.status}"
            + (f" ({item.detail})" if item.detail else "")
            + "; kept previous symbols as a stale fallback"
        )
        return False
    _replace_symbols(conn, repo, item)
    conn.execute(
        "INSERT INTO files (repo, path, mtime_ns, size, hash) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(repo, path) DO UPDATE SET mtime_ns = excluded.mtime_ns, "
        "size = excluded.size, hash = excluded.hash",
        (repo, item.rel, item.stat.st_mtime_ns, item.stat.st_size, item.content_hash),
    )
    if item.status == "partial":
        warnings.append(
            f"{item.rel}: partial parse, {len(item.symbols)} symbols extracted "
            "from source with errors"
        )
    return True


def _replace_symbols(conn, repo: str, item: _Staged) -> None:
    ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, item.rel)
        )
    ]
    _drop_symbols(conn, ids)
    for symbol in item.symbols:
        conn.execute(
            "INSERT INTO symbols (repo, path, name, qualname, kind, lang, line, end_line, "
            "signature, full_signature, docstring, body, search_text, embed_text, embed_key, "
            "source_role) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                repo,
                symbol.path,
                symbol.name,
                symbol.qualname,
                symbol.kind,
                symbol.lang,
                symbol.line,
                symbol.end_line,
                symbol.signature,
                symbol.full_signature,
                symbol.docstring,
                symbol.body,
                symbol.search_text,
                symbol.embed_text,
                _embed_key(symbol.embed_text),
                source_role(symbol.path),
            ),
        )


def _publish_epoch(conn, repo: str, head: str | None) -> int:
    row = conn.execute("SELECT epoch FROM repos WHERE repo = ?", (repo,)).fetchone()
    epoch = (row[0] if row else 0) + 1
    conn.execute(
        "INSERT INTO repos (repo, head, indexed_at, epoch) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(repo) DO UPDATE SET head = excluded.head, "
        "indexed_at = excluded.indexed_at, epoch = excluded.epoch",
        (repo, head, time.time(), epoch),
    )
    return epoch


def _attach_missing_vectors(  # noqa: PLR0913, PLR0917
    conn,
    repo: str,
    embed_fn,
    should_cancel,
    progress,
    warnings: list[str],
    embed_cache=None,
    embed_space_id: str = "",
) -> tuple[int, int, int]:
    """Embed symbols without vectors and attach them in guarded batches.

    Identical inputs are computed once: rows are grouped by their embedding
    key, the shared cache answers inputs computed before, and only the
    remaining unique inputs reach the model. Fresh vectors are stored in
    the cache immediately after each successful batch.
    """
    if embed_fn is None:
        return 0, 0, 0
    missing = conn.execute(
        "SELECT s.id, s.embed_text, s.embed_key, f.hash FROM symbols s "
        "LEFT JOIN symbols_vec v ON v.symbol_id = s.id "
        "JOIN files f ON f.repo = s.repo AND f.path = s.path "
        "WHERE s.repo = ? AND v.symbol_id IS NULL ORDER BY s.id",
        (repo,),
    ).fetchall()
    if not missing:
        return 0, 0, 0
    by_key: dict[str, list] = {}
    texts: dict[str, str] = {}
    for row in missing:
        by_key.setdefault(row[2], []).append(row)
        texts[row[2]] = row[1]

    cache_hits = 0
    pending_keys: list[str] | None = None
    if embed_cache is not None and embed_space_id:
        hits = embed_cache.lookup(embed_space_id, "document", list(texts.values()))
        if hits:
            hit_rows = [row for key in by_key if key in hits for row in by_key[key]]
            cache_hits = _attach_batch(conn, repo, hit_rows, [hits[row[2]] for row in hit_rows])
        pending_keys = [key for key in by_key if key not in hits]
    else:
        pending_keys = list(by_key)

    embedded = 0
    for start in range(0, len(pending_keys), EMBED_CHUNK):
        _check_cancelled(should_cancel)
        batch_keys = pending_keys[start : start + EMBED_CHUNK]
        batch_rows = [row for key in batch_keys for row in by_key[key]]
        _emit(progress, "embedding", {"pending": len(pending_keys) - start})
        vectors, warning = embed_fn([texts[key] for key in batch_keys])
        if vectors is None:
            warnings.append(warning or "embedding failed; dense coverage stays partial")
            return embedded, cache_hits, len(pending_keys) - start
        if embed_cache is not None and embed_space_id:
            # persist before attaching: a crash or later batch failure must
            # not lose this batch's inference
            embed_cache.store(
                embed_space_id,
                "document",
                list(zip(batch_keys, vectors, strict=True)),
                dim=len(vectors[0]) // 4 if vectors else 0,
            )
        # a job cancelled while inference was in flight must not attach its
        # unresolved results
        _check_cancelled(should_cancel)
        _emit(progress, "publishing_vectors", {"pending": len(pending_keys) - start})
        vector_by_key = dict(zip(batch_keys, vectors, strict=True))
        embedded += _attach_batch(
            conn, repo, batch_rows, [vector_by_key[row[2]] for row in batch_rows]
        )
    return embedded, cache_hits, 0


def _attach_batch(conn, repo: str, batch, vectors) -> int:
    """Attach one batch atomically, guarded by the capture identity.

    A vector lands only when the symbol's embedding key and the file's
    content hash still match the staged capture: a late result for a file
    that changed or was recreated must not attach to the new version.
    """
    attached = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row, vector in zip(batch, vectors, strict=False):
            symbol_id, _text, embed_key, content_hash = row
            fresh = conn.execute(
                "SELECT 1 FROM symbols s JOIN files f ON f.repo = s.repo AND f.path = s.path "
                "WHERE s.id = ? AND s.repo = ? AND s.embed_key = ? AND f.hash = ?",
                (symbol_id, repo, embed_key, content_hash),
            ).fetchone()
            if fresh is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO symbols_vec (symbol_id, repo, embedding) VALUES (?, ?, ?)",
                (symbol_id, repo, vector),
            )
            attached += 1
        conn.execute("COMMIT")
    except BaseException:
        _rollback(conn)
        raise
    return attached


def _record_parse_state(conn, repo: str, rel: str, status: str, detail: str | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO parse_state (repo, path, status, detail, attempted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (repo, rel, status, detail, time.time()),
    )


def _drop_file(conn, repo: str, rel: str) -> None:
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, rel))
    ]
    _drop_symbols(conn, ids)
    conn.execute("DELETE FROM files WHERE repo = ? AND path = ?", (repo, rel))
    conn.execute("DELETE FROM parse_state WHERE repo = ? AND path = ?", (repo, rel))


def _drop_symbols(conn, ids: list[int]) -> None:
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM symbols_vec WHERE symbol_id IN ({marks})", ids)  # noqa: S608 - marks are placeholders only
    conn.execute(f"DELETE FROM symbols WHERE id IN ({marks})", ids)  # noqa: S608 - marks are placeholders only


def _emit(progress: ProgressFn | None, phase: str, counters: dict) -> None:
    if progress is not None:
        progress(phase, counters)


def _check_cancelled(should_cancel: CancelFn | None) -> None:
    if should_cancel is not None and should_cancel():
        raise RefreshCancelledError


def _rollback(conn) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")
