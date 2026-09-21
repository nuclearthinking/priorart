"""CLI adapter.

The CLI may keep the convenience of relative paths: it resolves them to
absolute paths before calling the core. ``index`` follows the submitted job
to completion and prints progress events — the same events that
``--json-progress`` emits as one JSON object per line for scripts.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from pathlib import Path
from typing import Annotated

import typer

from .core import FINAL_STATES, Config, PriorartError
from .registry import RuntimeRegistry
from .retrieval import format_report

app = typer.Typer(help="Local agentic code search: find existing symbols before writing new code.")
daemon_app = typer.Typer(
    help="Run, stop or restart the shared coordinator daemon.",
)
app.add_typer(daemon_app, name="daemon")

_POLL_SECONDS = 0.05


@app.command()
def index(
    path: Annotated[Path | None, typer.Argument(help="Repository to index.")] = None,
    *,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help=(
                "Recreate the derived index from scratch. Embedding inputs "
                "already computed are reused from the shared cache; delete "
                "~/.priorart/indexes/v1/embed-cache.db to force re-inference."
            ),
        ),
    ] = False,
    json_progress: Annotated[
        bool, typer.Option("--json-progress", help="Emit job progress as JSON lines.")
    ] = False,
) -> None:
    """Index a repository and follow the job to completion."""
    registry = _registry()
    job = None
    try:
        handle = registry.resolve(_absolute(path))
        job = registry.submit_refresh(handle, rebuild=rebuild)
        job = _follow_job(registry, job.job_id, repo=handle.root, json_progress=json_progress)
    except PriorartError as err:
        _fail(err)
    except KeyboardInterrupt:
        # request cancellation so the worker stops between work units and
        # shutdown does not block on the full join timeout
        typer.echo("interrupted; requesting job cancellation", err=True)
        with contextlib.suppress(Exception):  # best effort on the way out
            if job is not None:
                registry.cancel_job(job.job_id, job.repo)
        raise
    finally:
        registry.close()
    counters = job.counters
    typer.echo(
        f"indexed {counters.get('files', 0)} changed files, "
        f"{counters.get('symbols', 0)} symbols -> {handle.store}"
    )
    if counters.get("removed"):
        typer.echo(f"removed {counters['removed']} deleted files")
    if job.epoch:
        typer.echo(f"index_epoch: {job.epoch}")
    for warning in job.warnings:
        typer.echo(f"warning: {warning}")
    if job.state == "failed":
        typer.echo(f"error: {job.failure['message']}")
        raise typer.Exit(code=1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to implement or find.")],
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1)] = 10,
) -> None:
    registry = _registry()
    try:
        handle = registry.resolve(_absolute(repo))
        report = handle.search(query, k=k)
    except PriorartError as err:
        _fail(err)
    finally:
        registry.close()
    typer.echo(format_report(report))


@app.command()
def status(repo: Annotated[Path | None, typer.Option("--repo")] = None) -> None:
    registry = _registry()
    try:
        handle = registry.resolve(_absolute(repo))
        typer.echo(handle.status_text())
    except PriorartError as err:
        _fail(err)
    finally:
        registry.close()


@app.command()
def workspaces() -> None:
    """List workspace candidates by origin (never selects anything)."""
    registry = _registry()
    for origin, entries in registry.workspaces().items():
        if entries:
            typer.echo(f"{origin}: " + ", ".join(entries))


@app.command()
def serve(
    repo: Annotated[Path | None, typer.Option("--repo", help="Default repository.")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Explicit service config file.")
    ] = None,
    embedded: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--embedded", help="Host jobs in this MCP process (development/tests)."),
    ] = False,
) -> None:
    from .server import build_server

    settings = Config(_env_file=config) if config is not None else None
    build_server(_absolute(repo), config=settings, config_path=config, embedded=embedded).run()


@app.command()
def doctor(
    repo: Annotated[Path | None, typer.Option("--repo", help="Repository to diagnose.")] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Explicit service config file.")
    ] = None,
) -> None:
    """Diagnose the local priorart setup without long inference.

    Reports versions, effective configuration (never secret values), model
    endpoint health and the index state of the selected repository.
    """
    import importlib.metadata
    import sqlite3
    import sys as _sys

    from .core import APP_VERSION as _APP_VERSION
    from .models.probe import probe_endpoints

    settings = Config(_env_file=config) if config is not None else Config()
    typer.echo(f"priorart: {_APP_VERSION}")
    typer.echo(f"python: {_sys.version.split()[0]}")
    try:
        typer.echo(f"mcp sdk: {importlib.metadata.version('mcp')}")
    except importlib.metadata.PackageNotFoundError:
        typer.echo("mcp sdk: not installed")
    typer.echo(f"sqlite: {sqlite3.sqlite_version}")
    typer.echo(f"index_dir: {settings.index_dir}")
    from .coordinator import DEFAULT_SOCKET, profile_fingerprint

    typer.echo("mcp_mode: daemon (default)")
    typer.echo(f"daemon_socket: {Path(settings.daemon_socket or DEFAULT_SOCKET).expanduser()}")
    typer.echo(f"service_profile: {profile_fingerprint(settings)[:12]}")

    from .coordinator import connect as connect_daemon

    try:
        client = connect_daemon(settings, autostart=False)
    except PriorartError as err:
        typer.echo(f"daemon: reachable, but refused this client: {err.message}")
    except OSError:
        typer.echo("daemon: not reachable (priorart daemon restart starts one)")
    else:
        typer.echo("daemon: reachable, current")
        client.close()

    configured = sorted(settings.model_fields_set)
    if configured:
        typer.echo(f"config set: {', '.join(configured)}")

    for name, status_line in probe_endpoints(settings).items():
        typer.echo(f"{name}: {status_line}")

    if repo is None:
        typer.echo("repo: not selected; pass --repo for index diagnostics")
        return
    registry = RuntimeRegistry(settings)
    try:
        handle = registry.resolve(_absolute(repo))
        typer.echo(handle.status_text())
    finally:
        registry.close()


@daemon_app.command("start")
def daemon_start(
    socket_path: Annotated[
        Path | None,
        typer.Option("--socket", help="Unix socket to listen on."),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Explicit service config file.")
    ] = None,
) -> None:
    """Run the shared coordinator in the foreground until interrupted.

    MCP servers are thin front-ends of this process by default; background
    indexing survives client restarts. Correctness never depends on the
    daemon: stores, locks and atomic commits stay valid without it.
    """
    import signal

    from .coordinator import DEFAULT_SOCKET
    from .coordinator import serve as serve_coordinator

    settings = Config(_env_file=config) if config is not None else Config()
    path = socket_path or Path(settings.daemon_socket or DEFAULT_SOCKET)
    if not str(path).strip():
        raise typer.BadParameter("the socket path must not be empty")
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    typer.echo(f"priorart daemon listening on {path}")
    serve_coordinator(settings, path, stop_event=stop_event)


@daemon_app.command()
def stop(
    socket_path: Annotated[
        Path | None,
        typer.Option("--socket", help="Unix socket the daemon listens on."),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Explicit service config file.")
    ] = None,
) -> None:
    """Stop the daemon gracefully: live calls drain, in-flight jobs interrupt."""
    from .coordinator import DEFAULT_SOCKET, stop_daemon

    settings = Config(_env_file=config) if config is not None else Config()
    path = socket_path or Path(settings.daemon_socket or DEFAULT_SOCKET)
    try:
        typer.echo(stop_daemon(path))
    except OSError as err:
        typer.echo(f"could not stop the priorart daemon: {err}", err=True)
        raise typer.Exit(code=1) from err


@daemon_app.command()
def restart(
    socket_path: Annotated[
        Path | None,
        typer.Option("--socket", help="Unix socket the daemon listens on."),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", help="Explicit service config file.")
    ] = None,
) -> None:
    """Stop the daemon and start a fresh detached one on the same socket."""
    from .coordinator import DEFAULT_SOCKET, restart_daemon

    settings = Config(_env_file=config) if config is not None else Config()
    path = socket_path or Path(settings.daemon_socket or DEFAULT_SOCKET)
    try:
        typer.echo(restart_daemon(settings, path, config_path=config))
    except PriorartError as err:
        _fail(err)
    except OSError as err:
        typer.echo(f"could not restart the priorart daemon: {err}", err=True)
        raise typer.Exit(code=1) from err


def _registry() -> RuntimeRegistry:
    return RuntimeRegistry(Config())


def _absolute(path: Path | None) -> Path | None:
    if path is None:
        return None
    if not str(path).strip():
        raise typer.BadParameter("the path must not be empty")
    return Path(path).expanduser().resolve()


def _follow_job(registry: RuntimeRegistry, job_id: str, *, repo: Path, json_progress: bool):
    last = None
    while True:
        _handle, job = registry.get_job(job_id, repo)
        current = (job.state, job.phase, tuple(sorted(job.counters.items())))
        if current != last:
            if json_progress:
                typer.echo(json.dumps(job.snapshot()))
            elif job.phase:
                typer.echo(f"{job.state} ({job.phase})")
            else:
                typer.echo(job.state)
            last = current
        if job.state in FINAL_STATES:
            return job
        time.sleep(_POLL_SECONDS)


def _fail(err: PriorartError) -> None:
    typer.echo(str(err), err=True)
    raise typer.Exit(code=1)


def main() -> None:
    app()
