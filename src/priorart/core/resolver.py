"""Workspace selection: explicit, immutable per call.

Resolution order (plan, "Выбор workspace"):

1. An explicit ``repo`` of the call: validate it, canonicalize to the git
   worktree root. An invalid explicit path is an error; fallback to another
   repository is forbidden.
2. Workspace context actually supplied by the client for this call.
3. Explicitly configured default repositories (``serve --repo`` or the
   configured workspace list). One candidate may be selected; several demand
   an explicit parameter.
4. No source at all: ``REPOSITORY_NOT_SELECTED``. The first cached handle,
   the last used repository, the only indexed repository or the process CWD
   are never used implicitly.

Identity of a worktree is its canonical absolute root; two worktrees of the
same history are different repositories. Symlinked aliases of one root
deduplicate during canonicalization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    AMBIGUOUS_WORKSPACE,
    REPOSITORY_NOT_FOUND,
    REPOSITORY_NOT_SELECTED,
    PriorartError,
)
from .git import git_toplevel


@dataclass(frozen=True)
class ResolvedRepo:
    root: Path
    input: Path | None = None

    def __str__(self) -> str:
        return str(self.root)


def canonical_root(path: Path) -> Path:
    """Canonical git worktree root for ``path`` or ``REPOSITORY_NOT_FOUND``.

    ``path`` may be the checkout root or any directory inside it.
    """
    if not path.is_absolute() or not path.exists():
        raise _invalid(path, "the path must exist and be absolute")
    root = git_toplevel(path)
    if root is None:
        raise _invalid(path, "the path is not inside a git repository")
    return root


def _canonical_candidates(paths: Iterable[Path]) -> list[Path]:
    """Canonical worktree roots of valid repositories among ``paths``.

    A path that is not inside any git repository (for example a parent
    folder above several repositories) contributes nothing; it must not
    silently become a git root of its own.
    """
    roots: dict[str, Path] = {}
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        root = git_toplevel(candidate) if candidate.exists() else None
        if root is not None:
            roots.setdefault(str(root), root)
    return [roots[key] for key in sorted(roots)]


def resolve_repo(
    *,
    explicit: Path | None = None,
    context: Iterable[Path] = (),
    defaults: Iterable[Path] = (),
) -> ResolvedRepo:
    """Select exactly one repository for one call."""
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.is_absolute():
            raise _invalid(explicit, "an explicit repo must be an absolute path")
        if not explicit.exists():
            raise _invalid(explicit, "the path does not exist")
        root = git_toplevel(explicit)
        if root is None:
            raise _invalid(explicit, "the path is not inside a git repository")
        return ResolvedRepo(root=root, input=explicit)
    candidates = _canonical_candidates(context)
    if not candidates:
        candidates = _canonical_candidates(defaults)
    if not candidates:
        raise PriorartError(
            REPOSITORY_NOT_SELECTED,
            "No repository is selected for this call.",
        )
    if len(candidates) > 1:
        raise PriorartError(
            AMBIGUOUS_WORKSPACE,
            "More than one repository is available.",
            candidates=[str(root) for root in candidates],
        )
    return ResolvedRepo(root=candidates[0])


def _invalid(path: Path, reason: str) -> PriorartError:
    return PriorartError(
        REPOSITORY_NOT_FOUND,
        f"Explicit repo {path} is invalid: {reason}.",
        input=str(path),
    )
