"""Auto-freshness watcher: polling invalidation with debounce (delivery G).

F01 new untracked file is indexed without a manual refresh, F02 deletions
are reflected and repeated edits coalesce, F03 content changes that
preserve size and mtime are caught by reconciliation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from priorart.core.jobs import FINAL_STATES
from priorart.registry import RuntimeRegistry
from tests.helpers import git, init_repo, make_config

SAMPLE = "def target():\n    pass\n"


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def _watch_config(tmp_path: Path):
    return make_config(
        tmp_path,
        watch_interval=0.05,
        watch_debounce=0.15,
        watch_content_interval=0.1,
    )


def _refresh(registry: RuntimeRegistry, repo: Path):
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _handle, current = registry.get_job(job.job_id)
        if current.state in FINAL_STATES:
            return handle, current
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _wait_for(condition, timeout: float = 20.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"{what} did not happen in {timeout}s")


def _symbols(handle, repo: Path) -> set[str]:
    reader = handle.reader()
    try:
        rows = reader.execute(
            "SELECT qualname FROM symbols WHERE repo = ?", (str(repo.resolve()),)
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        reader.close()


# --- F01: new untracked file -------------------------------------------------


def test_new_untracked_file_is_indexed_automatically(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(_watch_config(tmp_path))
    try:
        handle, first = _refresh(registry, repo)
        assert first.state == "completed"
        assert "target" in _symbols(handle, repo)

        (repo / "helper.py").write_text("def helper():\n    pass\n")
        _wait_for(lambda: "helper" in _symbols(handle, repo), what="untracked file indexed")
    finally:
        registry.close()


# --- F02: delete and repeated-edit coalescing ---------------------------------


def test_deleted_file_disappears_without_manual_refresh(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    (repo / "extra.py").write_text("def extra():\n    pass\n")
    git(repo, "add", "extra.py")
    git(repo, "commit", "-q", "-m", "extra")
    registry = RuntimeRegistry(_watch_config(tmp_path))
    try:
        handle, _first = _refresh(registry, repo)
        assert "extra" in _symbols(handle, repo)

        (repo / "extra.py").unlink()
        _wait_for(lambda: "extra" not in _symbols(handle, repo), what="deleted file removed")
    finally:
        registry.close()


def test_deletion_after_an_auto_refresh_still_submits(tmp_path):
    """A deletion must re-arm the debounce even after a submit happened."""
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(_watch_config(tmp_path))
    try:
        handle, _first = _refresh(registry, repo)

        (repo / "probe.py").write_text("def probe():\n    pass\n")
        _wait_for(lambda: "probe" in _symbols(handle, repo), what="probe auto-indexed")

        (repo / "probe.py").unlink()
        _wait_for(lambda: "probe" not in _symbols(handle, repo), what="probe removal published")
    finally:
        registry.close()


def test_rapid_edits_coalesce_into_one_publication(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    submits: list[list[str] | None] = []
    registry = RuntimeRegistry(_watch_config(tmp_path))
    try:
        _handle, _first = _refresh(registry, repo)
        handle = registry.resolve(repo)

        original_submit = registry.submit_refresh

        def counting_submit(handle_arg, **kwargs):
            submits.append(kwargs.get("paths"))
            return original_submit(handle_arg, **kwargs)

        registry.submit_refresh = counting_submit

        # three edits inside one debounce window: one quiet period, one job
        for index in range(3):
            (repo / "sample.py").write_text(f"def target():\n    pass\n# edit {index}\n")
            time.sleep(0.05)

        _wait_for(
            lambda: (
                submits
                and any("edit 2" in line for line in (repo / "sample.py").read_text().splitlines())
            ),
            what="edits settle",
        )
        _wait_for(lambda: len(submits) >= 1, what="watcher submits a refresh")
        time.sleep(0.6)
        assert len(submits) == 1
        # the coalesced publication carries the final content
        _wait_for(
            lambda: _file_hash_published(handle, repo, "sample.py"),
            what="final edit published",
        )
    finally:
        registry.close()


def _file_hash_published(handle, repo: Path, rel: str) -> bool:
    import hashlib

    reader = handle.reader()
    try:
        row = reader.execute(
            "SELECT hash FROM files WHERE repo = ? AND path = ?", (str(repo.resolve()), rel)
        ).fetchone()
    finally:
        reader.close()
    if row is None:
        return False
    return row[0] == hashlib.sha256((repo / rel).read_bytes()).hexdigest()


# --- F03: content change with preserved stat ---------------------------------


def test_content_change_with_preserved_stat_is_reconciled(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(_watch_config(tmp_path))
    try:
        handle, _first = _refresh(registry, repo)
        assert "target" in _symbols(handle, repo)

        target = repo / "sample.py"
        st = target.stat()
        # same size, different bytes, mtime restored to the captured value
        # same byte length as the captured version, different content
        target.write_text("def mutate():\n    pass\n")
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert target.stat().st_mtime_ns == st.st_mtime_ns
        assert target.stat().st_size == st.st_size

        _wait_for(lambda: "mutate" in _symbols(handle, repo), what="content reconcile")
        assert "target" not in _symbols(handle, repo)
    finally:
        registry.close()


# --- lifecycle ----------------------------------------------------------------


def test_watcher_stops_on_close(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(_watch_config(tmp_path))
    handle, _first = _refresh(registry, repo)
    stop = handle._watcher_stop
    assert stop is not None
    registry.close()
    (repo / "late.py").write_text("def late():\n    pass\n")
    time.sleep(0.4)
    assert "late" not in _symbols(handle, repo)
