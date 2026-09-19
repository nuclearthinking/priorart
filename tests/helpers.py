"""Shared test helpers: git plumbing and runtime configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from priorart.config import Config


def git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        env={
            "PATH": subprocess.os.environ["PATH"],
            "HOME": str(Path.home()),
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


def make_config(tmp_path: Path, **overrides) -> Config:
    fields = {
        "llm_base_url": None,
        "llm_api_key": None,
        "embed_base_url": None,
        "embed_api_key": None,
        "rerank_base_url": None,
        "rerank_api_key": None,
        "embed_model": "",
        "embed_dim": 4,
        "rerank_model": "",
        "llm_model": "",
        "db_path": tmp_path / "runtime.db",
    }
    fields.update(overrides)
    return Config(**fields)
