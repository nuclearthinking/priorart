"""Repository inventory: tracked plus non-ignored untracked source files.

Git is the only source of truth: a git failure is a diagnosable error, not
a reason to silently walk an arbitrary directory tree.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from priorart.core import git_ls_files, git_output

from .parser import LANGS, MAX_FILE_BYTES, language_of


class InventoryError(RuntimeError):
    """git could not enumerate the working tree."""


def list_source_files(root: Path) -> list[str]:
    """Relative paths of indexable source files in ``root``.

    Includes tracked files and non-ignored untracked files; ignored files,
    symlinks and oversized files are skipped. Raises ``InventoryError``
    when git itself fails.
    """
    root = Path(root)
    try:
        entries = git_ls_files(root)
    except OSError as err:
        raise InventoryError(str(err)) from err
    seen: dict[str, None] = {}
    for rel in entries:
        if PurePosixPath(rel).suffix in LANGS:
            seen.setdefault(rel)
    out = []
    for rel in seen:
        full = root / rel
        try:
            if full.is_symlink() or full.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(rel)
    return out


def repo_languages(root: Path) -> list[str]:
    """Distinct tree-sitter languages present in the inventory."""
    return sorted(
        {
            lang
            for lang in (language_of(rel) for rel in list_source_files(Path(root)))
            if lang is not None
        }
    )


def head_revision(root: Path) -> str | None:
    """Current HEAD of ``root`` as observed now, or ``None`` outside git."""
    return git_output(Path(root), "rev-parse", "HEAD")
