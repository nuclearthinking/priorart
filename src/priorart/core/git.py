"""Bounded git subprocess helpers.

Every git interaction goes through here so that timeouts and argv stay in one
place; no layer shells out to git on its own.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GIT_TIMEOUT_SECONDS = 30


def git_output(root: Path, *args: str) -> str | None:
    """Run git in ``root``; return stripped stdout, or ``None`` on failure."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed git argv
            ["git", "-C", str(root), *args],  # noqa: S607 - partial path is fine
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_ls_files(root: Path) -> list[str]:
    """Tracked plus untracked-but-not-ignored files, NUL-delimited.

    The single argv for inventory; raising ``OSError`` on git failure so
    callers translate it into their own domain error.
    """
    proc = subprocess.run(  # noqa: S603 - fixed git argv
        [  # noqa: S607 - partial path is fine
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "no diagnostics"
        raise OSError(f"git ls-files failed in {root}: {detail}")
    return [entry for entry in proc.stdout.split("\0") if entry]


def git_toplevel(path: Path) -> Path | None:
    """Canonical absolute worktree root containing ``path``, or ``None``."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed git argv
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return Path(out).resolve()
    except OSError:
        return None
