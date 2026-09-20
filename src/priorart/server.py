"""MCP adapter: the public tool surface.

Every repository-scoped call selects its repository explicitly (``repo``
parameter, configured default, or error — never a cached guess). Each tool
publishes its input contract as schema metadata and its success shape as a
top-level ``outputSchema``; invalid values still reach one domain guard so
every path returns the same structured ``INVALID_ARGUMENT``. Responses carry
a short human rendering in ``content`` and the machine envelope in
``structuredContent``; domain failures are MCP tool errors with
``is_error=true``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from .core import FINAL_STATES, SEARCH_INTENTS, SEARCH_MODES, Config, Envelope, PriorartError
from .registry import RuntimeRegistry
from .retrieval import format_report, format_status

INSTRUCTIONS = (
    "Local code-reuse search for coding agents. Before implementing a new "
    "function, class or module, call search_codebase and prefer importing, "
    "extending or refactoring an existing symbol over writing a duplicate. "
    "Pass the absolute repo of the current task to every repository-scoped tool. "
    "Results are hints: read the referenced file before reusing anything. "
    "When a tool returns INDEX_NOT_READY, call refresh_index once, save the job id, "
    "and poll get_index_job with the same repo until lexical_ready is true; searching "
    "can begin before embeddings finish. Do not refresh before every search. "
    "list_workspaces is discovery-only and never changes the selection. A newly "
    "created git worktree or another checkout has its own separate, initially empty "
    "index and needs one initial refresh. Tool errors and job failures include stable "
    "codes and next_action fields for recovery."
)


class _SearchCandidate(TypedDict):
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
    source_role: str
    score: float


class _SearchData(TypedDict):
    query: str
    k: int
    mode: str
    intent: str
    degradation_reasons: list[str]
    candidates: list[_SearchCandidate]


class _SearchIndex(TypedDict):
    repo: str
    index_epoch: int | None
    head_at_capture: str | None
    head_observed: str | None
    state: str
    degraded: bool
    degradation_reasons: list[str]
    stages_used: list[str]


class SearchSuccessPayload(TypedDict):
    """Success shape of ``search_codebase`` published as its outputSchema."""

    ok: Literal[True]
    repo: str
    index: _SearchIndex
    data: _SearchData
    warnings: list[str]
    timings: dict[str, float]


def build_server(
    repo: Path | None = None,
    *,
    config: Config | None = None,
    config_path: Path | None = None,
    embedded: bool = False,
) -> MCPServer:
    """Wire the tool surface over one shared registry.

    The default server is a thin client of the coordinator process, so jobs
    and watchers survive this stdio process. ``embedded`` is the explicit
    development/test escape hatch.
    """
    config = config or Config()
    if not embedded:
        from priorart.coordinator import remote_registry

        registry = remote_registry(config, default_repo=repo, config_path=config_path)
    else:
        registry = RuntimeRegistry(config, default_repos=[repo] if repo else [])

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield
        finally:
            registry.close()

    mcp = MCPServer("priorart", instructions=INSTRUCTIONS, lifespan=lifespan)

    @mcp.tool()
    def search_codebase(
        query: Annotated[
            str,
            Field(
                description="Natural-language description of the feature, task or symbol to find.",
                examples=["find the retry helper for provider calls"],
            ),
        ],
        repo: Annotated[
            str | None,
            Field(
                description="Absolute path of the repository worktree to search.",
                examples=["/absolute/path/to/repository"],
            ),
        ] = None,
        k: Annotated[
            int,
            Field(
                description="Maximum number of ranked candidates to return.",
                examples=[10],
                json_schema_extra={"minimum": 1},
            ),
        ] = 10,
        mode: Annotated[
            str,
            Field(
                description=(
                    "Retrieval depth: fast skips optional model stages, balanced runs "
                    "hybrid retrieval with configured reranking within one deadline, "
                    "deep adds bounded generative query expansion. Exact symbol or "
                    "qualname matches dispatch automatically in fast and balanced."
                ),
                json_schema_extra={"enum": list(SEARCH_MODES)},
            ),
        ] = "balanced",
        intent: Annotated[
            str,
            Field(
                description=(
                    "Result preference: implementation prefers production symbols over "
                    "comparable test helpers, tests returns only test symbols, any is "
                    "neutral."
                ),
                json_schema_extra={"enum": list(SEARCH_INTENTS)},
            ),
        ] = "implementation",
    ) -> Annotated[CallToolResult, SearchSuccessPayload]:
        """Search the repository for symbols relevant to a feature or task.

        Call this before implementing anything new and pass the absolute repo
        of the current task. Results are ranked symbol discovery hints with
        file, line, kind and signature — not a complete reference or call
        graph: a missing candidate does not prove the code is absent, so
        verify exhaustive callers and references by reading files and text or
        language search. Prefer extending an existing symbol over writing a
        duplicate.
        """
        return _guard_repo(_search, registry, repo, query, k, mode, intent)

    @mcp.tool()
    def map_symbols(
        path_glob: Annotated[
            str, Field(description="Glob filter over indexed symbol paths.")
        ] = "*",
        repo: Annotated[
            str | None, Field(description="Absolute path of the repository worktree.")
        ] = None,
        limit: Annotated[
            int,
            Field(
                description="Maximum rows to return.",
                json_schema_extra={"minimum": 1},
            ),
        ] = 100,
        cursor: Annotated[
            int | None,
            Field(description="Pagination cursor from a previous response; omit to start at zero."),
        ] = None,
    ) -> CallToolResult:
        """List indexed symbols under a path glob, e.g. 'src/**' or '*gateway*'."""
        return _guard_repo(_map_symbols, registry, repo, path_glob, limit, cursor)

    @mcp.tool()
    def refresh_index(
        repo: Annotated[
            str | None, Field(description="Absolute path of the repository worktree.")
        ] = None,
        rebuild: Annotated[  # noqa: FBT002
            bool,
            Field(
                description="Recreate the derived index from scratch instead of refreshing changed files."
            ),
        ] = False,
        paths: Annotated[
            list[str] | None,
            Field(description="Restrict this refresh to these repository-relative paths."),
        ] = None,
    ) -> CallToolResult:
        """Refresh the repository index; returns a job immediately.

        Incremental by default: only changed files are recaptured. rebuild
        recreates the derived index from scratch, reusing vectors already
        computed from the shared embedding cache. Follow the returned job
        with get_index_job; searching stays possible while the job runs.
        """
        return _guard_repo(_refresh_index, registry, repo, rebuild, paths)

    @mcp.tool()
    def get_index_status(
        repo: Annotated[
            str | None, Field(description="Absolute path of the repository worktree.")
        ] = None,
    ) -> CallToolResult:
        """Show index freshness: repo, published epoch, index vs current
        HEAD, working tree state, staleness with reasons, dense coverage."""
        return _guard_repo(_status, registry, repo)

    @mcp.tool()
    def list_workspaces() -> CallToolResult:
        """List workspace candidates by origin; does not select anything."""
        return _guard(_list_workspaces, registry)

    @mcp.tool()
    def get_index_job(
        job_id: Annotated[str, Field(description="Identifier of the indexing job.")],
        repo: Annotated[
            str | None, Field(description="Absolute path of the repository worktree.")
        ] = None,
    ) -> CallToolResult:
        """Show one indexing job: state, phase, counters, published epoch."""
        return _guard_job(_get_index_job, registry, job_id, repo)

    @mcp.tool()
    def cancel_index_job(
        job_id: Annotated[str, Field(description="Identifier of the indexing job.")],
        repo: Annotated[
            str | None, Field(description="Absolute path of the repository worktree.")
        ] = None,
    ) -> CallToolResult:
        """Request cancellation of one indexing job. Already published
        lexical state stays published."""
        return _guard_job(_cancel_index_job, registry, job_id, repo)

    return mcp


def _guard(fn, *args) -> CallToolResult:
    """Run one tool body; domain failures become structured tool errors."""
    try:
        envelope, text = fn(*args)
        return _call(envelope, text)
    except PriorartError as err:
        return _failure(err)


def _guard_repo(fn, registry, repo, *args) -> CallToolResult:
    """Resolve once and preserve that repository on every later error."""
    resolved_repo = None
    try:
        handle = registry.resolve(_absolute(repo))
        resolved_repo = str(handle.root)
        envelope, text = fn(registry, handle, *args)
        return _call(envelope, text)
    except PriorartError as err:
        return _call(Envelope.failure(err, repo=resolved_repo), _error_text(err.payload()))


def _guard_job(fn, *args) -> CallToolResult:
    """Render a job error with the canonical repo resolved by the registry."""
    try:
        envelope, text = fn(*args)
        return _call(envelope, text)
    except PriorartError as err:
        return _failure(err)


def _failure(err: PriorartError) -> CallToolResult:
    envelope = Envelope.failure(err, repo=err.details.get("resolved_repo"))
    return _call(envelope, _error_text(envelope.error))


def _call(envelope: Envelope, text: str) -> CallToolResult:
    """One rendering per channel: human text in content, machine payload in structured."""
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=envelope.payload(),
        is_error=not envelope.ok,
    )


def _error_text(error: dict) -> str:
    """Human rendering derived from the machine error payload."""
    lines = [error["message"]]
    for violation in error.get("violations", ()):
        accepted = ", ".join(str(item) for item in violation["accepted"])
        lines.append(f"{violation['field']}: got {violation['input']!r}; accepted: {accepted}")
    lines.extend(
        f"{key}: {error[key]}" for key in ("candidates", "input", "job_id") if key in error
    )
    lines.append(error["next_action"])
    return "\n".join(lines)


def _success_text(envelope: Envelope, body: str) -> str:
    lines = [f"warning: {warning}" for warning in envelope.warnings]
    lines.append(body.rstrip())
    return "\n".join(lines)


def _search(_registry, handle, query, k, mode, intent) -> tuple[Envelope, str]:
    # mode/intent/k validation lives in RepoHandle.search: one structured
    # guard shared by the embedded and daemon paths
    report = handle.search(query, k=k, mode=mode, intent=intent)
    index_block = {
        "repo": str(handle.root),
        "index_epoch": report.epoch,
        "head_at_capture": report.head,
        "head_observed": report.current_head,
        "state": "ready",
        "degraded": report.degraded,
        "degradation_reasons": report.degradation_reasons,
        "stages_used": report.stages_used,
    }
    envelope = Envelope.success(
        repo=str(handle.root),
        data={
            "query": query,
            "k": k,
            "mode": mode,
            "intent": intent,
            "degradation_reasons": report.degradation_reasons,
            "candidates": [candidate.payload() for candidate in report.candidates],
        },
        index=index_block,
        warnings=report.warnings,
        timings=report.trace.stage_seconds if report.trace else {},
    )
    return envelope, _success_text(envelope, format_report(report))


def _map_symbols(_registry, handle, path_glob, limit, cursor) -> tuple[Envelope, str]:
    rows, next_cursor = handle.map_symbols(path_glob, limit=limit, offset=cursor or 0)
    envelope = Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={
            "path_glob": path_glob,
            "symbols": rows,
            "next_cursor": next_cursor,
            "truncated": next_cursor is not None,
        },
    )
    return envelope, _map_text(path_glob, rows, next_cursor)


def _map_text(path_glob: str, rows, next_cursor) -> str:
    if not rows:
        return f"No indexed symbols match {path_glob!r}."
    lines = [f"{row['path']}:{row['line']} {row['kind']} {row['qualname']}" for row in rows]
    if next_cursor is not None:
        lines.append("(truncated; pass the returned next_cursor for more rows)")
    return "\n".join(lines)


def _refresh_index(registry, handle, rebuild, paths) -> tuple[Envelope, str]:
    job, submission = registry.submit_refresh(
        handle, rebuild=rebuild, paths=paths, include_submission=True
    )
    envelope = Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={"job": job.snapshot(), "submission": submission},
    )
    return envelope, _job_text(job, submission)


def _status(_registry, handle) -> tuple[Envelope, str]:
    # one snapshot feeds both the machine payload and the human text
    summary = handle.status()
    envelope = Envelope.success(repo=str(handle.root), index=summary.payload(), data={})
    return envelope, format_status(summary)


def _list_workspaces(registry) -> tuple[Envelope, str]:
    workspaces = registry.workspaces()
    lines = ["workspace candidates (selection never changes implicitly):"]
    for origin in ("configured", "known_indexed"):
        entries = workspaces[origin]
        if entries:
            lines.append(f"{origin}: " + ", ".join(entries))
        else:
            lines.append(f"{origin}: (none)")
    lines.append("Pass an explicit repo to any repository-scoped tool to select a workspace.")
    envelope = Envelope.success(repo=None, data={"workspaces": workspaces})
    return envelope, "\n".join(lines)


def _get_index_job(registry, job_id, repo) -> tuple[Envelope, str]:
    selected = registry.resolve(None).root if repo is None else _absolute(repo)
    handle, job = registry.get_job(job_id, selected)
    envelope = Envelope.success(
        repo=str(handle.root),
        index=handle.index_metadata(),
        data={"job": job.snapshot()},
    )
    return envelope, _job_text(job, None)


def _cancel_index_job(registry, job_id, repo) -> tuple[Envelope, str]:
    selected = registry.resolve(None).root if repo is None else _absolute(repo)
    job = registry.cancel_job(job_id, selected)
    action = (
        "cancellation requested" if job.state not in FINAL_STATES else "already final; unchanged"
    )
    envelope = Envelope.success(repo=job.repo, data={"job": job.snapshot()})
    return envelope, _job_text(job, action)


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
    if job.failure:
        parts.append(f"failure {job.failure['code']}: {job.failure['message']}")
    parts.extend(f"warning: {warning}" for warning in job.warnings)
    if action:
        parts.append(f"({action})")
    if job.state in FINAL_STATES:
        parts.append("(final)")
    return ", ".join(parts)


def _absolute(repo: str | None) -> Path | None:
    """Pass the caller's repo through; the resolver rejects relative paths."""
    return Path(repo) if repo else None


__all__ = ["INSTRUCTIONS", "SearchSuccessPayload", "build_server"]
