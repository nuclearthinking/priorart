from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .runtime import Runtime
from .search import format_report

app = typer.Typer(help="Local agentic code search: find existing symbols before writing new code.")


@app.command()
def index(
    path: Annotated[Path | None, typer.Argument(help="Repository to index.")] = None,
    *,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Reindex and re-embed everything.")
    ] = False,
) -> None:
    runtime = Runtime(path or Path.cwd())
    stats = runtime.reindex(rebuild=rebuild)
    typer.echo(
        f"indexed {stats['files']} changed files, {stats['symbols']} symbols "
        f"-> {runtime.config.db_path}"
    )
    if stats.get("removed"):
        typer.echo(f"removed {stats['removed']} deleted files")
    for warning in stats.get("warnings", []):
        typer.echo(f"warning: {warning}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to implement or find.")],
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1)] = 10,
) -> None:
    runtime = Runtime(repo or Path.cwd())
    typer.echo(format_report(runtime.search(query, k=k)))


@app.command()
def status(repo: Annotated[Path | None, typer.Option("--repo")] = None) -> None:
    runtime = Runtime(repo or Path.cwd())
    typer.echo(runtime.status())


@app.command()
def serve(repo: Annotated[Path | None, typer.Option("--repo")] = None) -> None:
    from .server import build_server

    build_server(repo).run()


def main() -> None:
    app()
