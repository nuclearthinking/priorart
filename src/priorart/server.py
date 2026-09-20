"""MCP adapter: the public tool surface.

Every repository-scoped call selects its repository explicitly (``repo``
parameter, configured default, or error — never a cached guess). Responses
carry a short text rendering plus a machine-readable structured envelope;
domain failures are MCP tool errors with ``is_error=true``.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent

from .core import FINAL_STATES, Config, Envelope, PriorartError
from .registry import RuntimeRegistry
from .retrieval import format_report

INSTRUCTIONS = (
    "Codebase search over one indexed repository. Before implementing a new "
    "function, class or module, call search_codebase and prefer importing, "
    "extending or refactoring an existing symbol over writing a duplicate. "
    "Pass the absolute repo of the current task in the repo parameter. "
    "Results are hints: read the referenced file before reusing anything. "
    "When a tool returns INDEX_NOT_READY, call refresh_index once and follow "
    "get_index_job until lexical_ready; do not run a full refresh before "
    "every search. A newly created git worktree or another checkout has "
    "its own separate, initially empty index: call refresh_index once after "
    "creating it."
)


def build_server(
    repo: Path | None = None,
    *,
    config: Config | None = None,
) -> MCPServer:
    """Wire the tool surface over one shared registry.

    With ``daemon_socket`` configured the server becomes a thin client of
    the coordinator process: background jobs and watchers live there and
    survive this server's lifetime.
    """
    config = config or Config()
    if config.daemon_socket:
        from priorart.coordinator import remote_registry

        registry = remote_registry(config, default_repo=repo)
    else:
        registry = RuntimeRegistry(config, default_repos=[repo] if repo else [])
    mcp = MCPServer("priorart", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_codebase(
        query: str,
        repo: str | None = None,
        k: int = 10,
        mode: str = "balanced",
        intent: str = "implementation",
    ) -> CallToolResult:
        """Search the repository for symbols relevant to a feature or task.

        Call this before implementing anything new. Pass the absolute repo
        of the current task. Returns ranked candidates with file, line, kind
        and signature; prefer extending an existing symbol over writing a
        duplicate.
        """
        return _guard(_search, registry, repo, query, k, mode, intent)

    @mcp.tool()
    def map_symbols(
        path_glob: str = "*",
        repo: str | None = None,
        limit: int = 100,
        cursor: int | None = None,
    ) -> CallToolResult:
        """List indexed symbols under a path glob, e.g. 'src/**' or '*gateway*'."""
        return _guard(_map_symbols, registry, repo, path_glob, limit, cursor)

    @mcp.tool()
    def refresh_index(
        repo: str | None = None,
        rebuild: bool = False,  # noqa: FBT001, FBT002
        paths: list[str] | None = None,
    ) -> CallToolResult:
        """Refresh the repository index; returns a job immediately.

        Incremental by default: only changed files are recaptured. rebuild
        recreates the derived index from scratch, reusing vectors already
        computed from the shared embedding cache. Follow the returned job
        with get_index_job; searching stays possible while the job runs.
        """
        return _guard(_refresh_index, registry, repo, rebuild, paths)

    @mcp.tool()
    def status(repo: str | None = None) -> CallToolResult:
        """Show index freshness: repo, published epoch, index vs current
        HEAD, working tree state, staleness with reasons, dense coverage."""
        return _guard(_status, registry, repo)

    @mcp.tool()
    def list_workspaces() -> CallToolResult:
        """List workspace candidates by origin; does not select anything."""
        return _guard(_list_workspaces, registry)

    @mcp.tool()
    def get_index_job(job_id: str, repo: str | None = None) -> CallToolResult:
        """Show one indexing job: state, phase, counters, published epoch."""
        return _guard(_get_index_job, registry, job_id, repo)

    @mcp.tool()
    def cancel_index_job(job_id: str, repo: str | None = None) -> CallToolResult:
        """Request cancellation of one indexing job. Already published
        lexical state stays published."""
        return _guard(_cancel_index_job, registry, job_id, repo)

    return mcp


def _guard(fn, *args) -> CallToolResult:
    """Run one tool body; domain failures become structured tool errors."""
    try:
        return _call(fn(*args))
    except PriorartError as err:
        return _call(Envelope.failure(err))


def _call(envelope: Envelope) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=envelope.text())],
        structured_content=envelope.payload(),
        is_error=not envelope.ok,
    )


def _search(registry, repo, query, k, mode, intent) -> Envelope:  # noqa: PLR0913, PLR0917
    # mode/intent validation lives in RepoHandle.search: one guard for the
    # local and the daemon path; an invalid value raises ValueError, which
    # the SDK reports as an unexpected tool error (see plan note A)
    handle = registry.resolve(_absolute(repo))
    report = handle.search(query, k=k, mode=mode, intent=intent)
    index_block = {
        "repo": str(handle.root),
        "index_epoch": report.epoch,
        "head_at_capture": report.head,
        "head_observed": report.current_head,
        "state": "ready",
        "degraded": report.degraded,
        "stages_used": report.stages_used,
    }
    return Envelope.success(
        repo=str(handle.root),
        data={
            "query": query,
            "mode": mode,
            "intent": intent,
            "k": k,
            "text": format_report(report),
            "candidates": [candidate.payload() for candidate in report.candidates],
        },
        index=index_block,
        warnings=report.warnings,
        timings=report.trace.stage_seconds if report.trace else {},
    )


def _map_symbols(registry, repo, path_glob, limit, cursor) -> Envelope:
    handle = registry.resolve(_absolute(repo))
    rows, next_cursor = handle.map_symbols(path_glob, limit=limit, offset=cursor or 0)
    return Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={
            "path_glob": path_glob,
            "symbols": rows,
            "next_cursor": next_cursor,
            "truncated": next_cursor is not None,
            "text": _map_text(path_glob, rows, next_cursor),
        },
    )


def _map_text(path_glob: str, rows, next_cursor) -> str:
    if not rows:
        return f"No indexed symbols match {path_glob!r}."
    lines = [f"{row['path']}:{row['line']} {row['kind']} {row['qualname']}" for row in rows]
    if next_cursor is not None:
        lines.append("(truncated; pass the returned next_cursor for more rows)")
    return "\n".join(lines)


def _refresh_index(registry, repo, rebuild, paths) -> Envelope:
    handle = registry.resolve(_absolute(repo))
    job = registry.submit_refresh(handle, rebuild=rebuild, paths=paths)
    return Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={
            "job": job.snapshot(),
            "text": _job_text(job, "started"),
        },
    )


def _status(registry, repo) -> Envelope:
    handle = registry.resolve(_absolute(repo))
    summary = handle.status()
    return Envelope.success(
        repo=str(handle.root),
        index=summary.payload(),
        data={"text": handle.status_text()},
    )


def _list_workspaces(registry) -> Envelope:
    workspaces = registry.workspaces()
    lines = ["workspace candidates (selection never changes implicitly):"]
    for origin in ("context", "configured", "known_indexed"):
        entries = workspaces[origin]
        if entries:
            lines.append(f"{origin}: " + ", ".join(entries))
        else:
            lines.append(f"{origin}: (none)")
    lines.append("Pass an explicit repo to any repository-scoped tool to select a workspace.")
    return Envelope.success(repo=None, data={"workspaces": workspaces, "text": "\n".join(lines)})


def _get_index_job(registry, job_id, repo) -> Envelope:
    handle, job = registry.get_job(job_id, _absolute(repo))
    return Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={"job": job.snapshot(), "text": _job_text(job, None)},
    )


def _cancel_index_job(registry, job_id, repo) -> Envelope:
    job = registry.cancel_job(job_id, _absolute(repo))
    return Envelope.success(
        repo=job.repo,
        data={"job": job.snapshot(), "text": _job_text(job, "cancellation requested")},
    )


def _job_text(job, action: str | None) -> str:
    parts = [f"job {job.job_id}: {job.state}"]
    if job.phase:
        parts.append(f"phase {job.phase}")
    counters = " ".join(f"{key}={value}" for key, value in sorted(job.counters.items()))
    if counters:
        parts.append(counters)
    if job.lexical_ready:
        parts.append("lexical_ready")
    if job.dense_ready:
        parts.append("dense_ready")
    if job.epoch:
        parts.append(f"index_epoch={job.epoch}")
    if job.error:
        parts.append(f"error: {job.error}")
    parts.extend(f"warning: {warning}" for warning in job.warnings)
    if action:
        parts.append(f"({action})")
    if job.state in FINAL_STATES:
        parts.append("(final)")
    return ", ".join(parts)


def _absolute(repo: str | None) -> Path | None:
    """Pass the caller's repo through; the resolver rejects relative paths."""
    return Path(repo) if repo else None


__all__ = ["INSTRUCTIONS", "build_server"]
