"""CLI doctor: diagnostics without secrets or long inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.helpers import git, init_repo


def _doctor(env_extra: dict[str, str], *args: str) -> str:
    import os
    import tempfile

    repo = init_repo(Path(tempfile.mkdtemp(prefix="pa-doctor-")) / "repo")
    (repo / "app.py").write_text("def doctor_target():\n    pass\n")
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "init")
    env = {**os.environ, **env_extra, "PRIORART_WATCH_INTERVAL": "0"}
    result = subprocess.run(  # noqa: S603 - fixed priorart argv
        [sys.executable, "-m", "priorart", "doctor", "--repo", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_doctor_reports_versions_and_config_without_secret_values():
    out = _doctor(
        {"PRIORART_API_KEY": "sk-secret-value-do-not-print"},
    )
    assert "priorart:" in out
    assert "python:" in out
    assert "mcp sdk:" in out
    assert "sqlite:" in out
    assert "index_dir:" in out
    assert "api_key" in out  # the field name is listed...
    assert "sk-secret-value-do-not-print" not in out  # ...but never its value
    assert "repo:" in out
    assert "state: absent" in out


def test_doctor_marks_unconfigured_endpoints():
    out = _doctor(
        {
            "PRIORART_BASE_URL": "",
            "PRIORART_EMBED_BASE_URL": "",
            "PRIORART_RERANK_BASE_URL": "",
            "PRIORART_LLM_BASE_URL": "",
        }
    )
    assert "embed: not configured" in out
    assert "rerank: not configured" in out
    assert "llm: not configured" in out
