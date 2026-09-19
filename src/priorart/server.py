from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .runtime import Runtime
from .search import format_report

INSTRUCTIONS = (
    "Codebase search over one indexed repository. Before implementing a new "
    "function, class or module, call search_codebase and prefer importing, "
    "extending or refactoring an existing symbol over writing a duplicate. "
    "Results are hints: read the referenced file before reusing anything. "
    "Call refresh_index when the index is stale. A newly created git worktree "
    "or another checkout has its own separate, initially empty index: call "
    "refresh_index once after creating it."
)

NO_REPO = (
    "priorart: no git repository detected at the server working directory. "
    "Start your coding agent (opencode, Codex, Claude Code, ...) inside a git "
    "repository, or pass --repo to 'priorart serve'."
)


def _git_root(start: Path) -> Path | None:
    try:
        out = subprocess.run(  # noqa: S603 - fixed git argv
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out) if out else None


def build_server(repo: Path | None = None) -> MCPServer:
    root = Path(repo) if repo is not None else _git_root(Path.cwd())
    runtime = Runtime(root) if root is not None else None
    mcp = MCPServer("priorart", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_codebase(query: str, k: int = 10) -> str:
        """Search the repository for symbols relevant to a feature or task.

        Call this before implementing anything new. Returns ranked candidates
        with file, line, kind and signature. Prefer extending an existing
        symbol over writing a duplicate.
        """
        if runtime is None:
            return NO_REPO
        return format_report(runtime.search(query, k=k))

    @mcp.tool()
    def map_symbols(path_glob: str = "*") -> str:
        """List indexed symbols under a path glob, for example 'src/**' or '*gateway*'."""
        if runtime is None:
            return NO_REPO
        return runtime.map_symbols(path_glob)

    @mcp.tool()
    def refresh_index(rebuild: bool = False) -> str:  # noqa: FBT001, FBT002
        """Reindex the repository. Incremental by default; rebuild=True re-embeds everything."""
        if runtime is None:
            return NO_REPO
        stats = runtime.reindex(rebuild=rebuild)
        parts = [
            f"indexed {stats['files']} changed files, {stats['symbols']} symbols in them",
        ]
        if stats.get("removed"):
            parts.append(f"removed {stats['removed']} deleted files")
        parts.append(f"index total {runtime.symbol_count()} symbols")
        lines = [", ".join(parts)]
        lines.extend(f"warning: {warning}" for warning in stats.get("warnings", []))
        return "\n".join(lines)

    @mcp.tool()
    def status() -> str:
        """Show index freshness: repo, index vs current HEAD, working tree
        state, staleness with reasons, and vector coverage."""
        if runtime is None:
            return NO_REPO
        return runtime.status()

    return mcp
