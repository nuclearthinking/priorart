"""Safe file capture: read bytes and stat from one open file description.

The capture refuses symlinked path components and detects files that change
while being read; its result is the evidence a snapshot row is built from.
"""

from __future__ import annotations

import os
from pathlib import Path

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _open_beneath(root: Path, rel: str) -> int:
    """Open rel strictly beneath root, refusing symlinked path components."""
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise FileNotFoundError(rel)
    dir_fd = os.open(root, os.O_RDONLY)
    opened: int | None = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | _NOFOLLOW, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        opened = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return opened


def capture_file(root: Path, rel: str) -> tuple[bytes, os.stat_result] | None:
    """Read file bytes and its stat together, from the same open file description."""
    try:
        fd = _open_beneath(root, rel)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            before = os.fstat(fh.fileno())
            data = fh.read()
            after = os.fstat(fh.fileno())
    except OSError:
        return None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None
    if len(data) != before.st_size:
        return None
    return data, before
