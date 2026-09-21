"""Application identity: the package version and the loaded-code digest.

One definition point so the MCP handshake, store profiles, the daemon
protocol and doctor all agree without depending on a storage internal.
``APP_VERSION`` pins stores and embedding caches; ``code_identity()`` pins
the daemon handshake — its digest changes with the loaded source, so a
daemon started from older code is refused instead of answering with old
payload shapes, while installed distributions stay stable until an upgrade.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    APP_VERSION = version("priorart")
except PackageNotFoundError:  # running from a source checkout without installation
    APP_VERSION = "unknown"

_CODE_IDENTITY: dict[str, str] = {}


def source_tree_digest(root: Path) -> str:
    """Content digest of every ``*.py`` under ``root``, by sorted path."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_identity() -> str:
    """Identity of the loaded priorart code, computed once per process.

    Edits on disk do not change a running process, so the value is cached
    for the process lifetime: it is the identity the daemon pinned at its
    start, and a client process keeps its own until it restarts.
    """
    if "value" not in _CODE_IDENTITY:
        _CODE_IDENTITY["value"] = source_tree_digest(Path(__file__).resolve().parents[1])
    return _CODE_IDENTITY["value"]
