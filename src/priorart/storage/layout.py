"""Per-worktree store layout.

Every canonical worktree root gets its own directory under the index root;
every embedding profile gets its own SQLite file inside it. Two worktrees of
the same history never share a store, and incompatible embedding profiles
never share vectors.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

STORE_LAYOUT_VERSION = 1

ROOT_MARKER = "root.json"


def _digest(value: str, length: int) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def worktree_dir(index_root: Path, root: Path) -> Path:
    """Stable directory for one canonical worktree root."""
    index_root = Path(index_root)
    root = Path(root).resolve()
    return index_root / f"v{STORE_LAYOUT_VERSION}" / f"{root.name}-{_digest(str(root), 16)}"


def profile_id(embed_model: str, embed_dim: int, embed_input_format: str) -> str:
    """Store filename fragment identifying one embedding space contract."""
    if not embed_model:
        return "lexical"
    slug = "".join(char if char.isalnum() else "-" for char in embed_model).strip("-")
    fingerprint = _digest(f"{embed_model}\0{embed_dim}\0{embed_input_format}", 12)
    return f"{slug}-d{embed_dim}-{fingerprint}"


def store_path(worktree_directory: Path, profile: str) -> Path:
    return Path(worktree_directory) / f"{profile}.db"


def embed_cache_path(index_root: Path) -> Path:
    """Shared embedding cache location: one file for every worktree.

    The cache stores exact model inputs, which do not depend on the
    repository, so all worktrees and profiles read and write the same file
    keyed by their own embedding space identity.
    """
    return Path(index_root) / f"v{STORE_LAYOUT_VERSION}" / "embed-cache.db"


def ensure_worktree_dir(index_root: Path, root: Path) -> Path:
    """Create the worktree directory and its root marker if absent."""
    directory = worktree_dir(index_root, root)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ROOT_MARKER
    if not marker.exists():
        marker.write_text(
            json.dumps({"root": str(root), "created_at": time.time()}), encoding="utf-8"
        )
    return directory


def known_roots(index_root: Path) -> list[Path]:
    """Canonical roots of every worktree this service has indexed."""
    index_root = Path(index_root)
    roots: list[Path] = []
    if not index_root.exists():
        return roots
    for marker in sorted(index_root.glob(f"v*/*/{ROOT_MARKER}")):
        try:
            root = Path(json.loads(marker.read_text(encoding="utf-8"))["root"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if root.exists():
            roots.append(root)
    return roots
