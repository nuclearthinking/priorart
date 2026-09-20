"""Inter-process locks.

Python mutexes do not survive multiple processes (CLI plus several MCP
servers), so store ownership uses OS file locks: the kernel releases them
when the owning process dies. Readers never take these locks; they protect
writer ownership only.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Self


class FileLock:
    """Advisory exclusive lock on a lock file, released by the OS on death."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Try to take the lock once; ``False`` means someone else owns it."""
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        if not self.acquire():
            raise RuntimeError(f"another process holds {self.path}")
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
