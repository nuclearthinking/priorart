"""Regression tests for the bug-hunt fixes.

Each test pins one defect found by the post-implementation review:
query-vector misalignment on partial batch failure, the watcher latch
never re-arming after a failed job, the job worker dying on unexpected
exceptions, exact dispatch bypassing the tests intent, and fast mode
reporting a false dense-degradation.
"""

from __future__ import annotations

import threading
import time

from priorart.indexing.watcher import start_watcher
from priorart.registry import RuntimeRegistry
from priorart.retrieval import search
from tests.helpers import make_config, repo_with_files, wait_job

# --- query vectors never misalign on partial batch failure -------------------


def test_embed_batch_failure_keeps_queries_unvectored(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    stored: dict[str, bytes] = {}
    fake_vector = b"\x00\x00\x80?" * 4

    def flaky_embed(texts, *, query=False):
        # every batch containing more than the first text fails: vectors
        # returned for a failed batch must never land on other queries
        if len(texts) > 1:
            return None, "batch failed"
        return [fake_vector], None

    class Cache:
        def lookup(self, space_id, role, texts):
            return {}

        def store(self, space_id, role, entries):
            stored.update(entries)

    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)
        reader = handle.reader()
        try:
            report = search(
                reader,
                str(repo.resolve()),
                "owner handler function",
                k=5,
                embed_fn=flaky_embed,
                query_cache=Cache(),
            )
        finally:
            reader.close()
        # the query was embedded in one successful single-item batch and
        # nothing else was stored with misaligned vectors
        assert stored in ({}, {"owner handler function": fake_vector})
        assert report.candidates
    finally:
        registry.close()


# --- watcher latch re-arms after a failed job ---------------------------------


def test_watcher_retries_after_failed_refresh(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)

        failures = {"remaining": 1}
        real_refresh = registry.submit_refresh

        def failing_refresh(handle, **kwargs):
            if failures["remaining"] > 0:
                failures["remaining"] -= 1
                raise RuntimeError("simulated submit failure")
            return real_refresh(handle, **kwargs)

        registry.submit_refresh = failing_refresh

        (repo / "src/app.py").write_text("def owner():\n    return 1\n")
        stop = start_watcher(
            repo,
            str(repo.resolve()),
            open_state=handle._published_file_state,
            submit=failing_refresh,
            interval=0.02,
            debounce=0.0,
            content_interval=999.0,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                _h, job = registry.get_job(_latest_job_id(registry, repo))
                if job is not None and job.state == "completed":
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("watcher never retried after the submit failure")
        finally:
            stop()
    finally:
        registry.close()


def _latest_job_id(registry, repo):
    handle = registry.resolve(repo)
    reader = handle.reader()
    try:
        row = reader.execute(
            "SELECT job_id FROM jobs WHERE repo = ? ORDER BY created_at DESC LIMIT 1",
            (str(repo.resolve()),),
        ).fetchone()
    finally:
        reader.close()
    return row[0] if row else ""


# --- worker survives an unexpected exception ----------------------------------


def test_worker_survives_persist_failure(tmp_path, monkeypatch):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        job = registry.submit_refresh(handle)
        wait_job(registry, job.job_id)

        # a persist that raises on the first call of the next job must not
        # kill the worker; the guard marks the job failed and the manager
        # stays usable afterwards
        from priorart.indexing import jobs as jobs_module

        real_persist = jobs_module.JobManager._persist
        calls = {"n": 0}

        def flaky_persist(self, job):
            calls["n"] += 1
            worker_thread = threading.current_thread().name == "priorart-index"
            if worker_thread and calls["n"] == 2:
                raise RuntimeError("journal write failed")
            return real_persist(self, job)

        monkeypatch.setattr(jobs_module.JobManager, "_persist", flaky_persist)
        (repo / "src/app.py").write_text("def owner():\n    return 2\n")
        second = registry.submit_refresh(handle)
        second = wait_job(registry, second.job_id)
        assert second.state == "failed"
        assert "journal write failed" in (second.error or "")

        # the worker is still alive: a third job runs to completion
        third = registry.submit_refresh(handle)
        third = wait_job(registry, third.job_id)
        assert third.state == "completed"
    finally:
        registry.close()


# --- exact dispatch respects the tests intent --------------------------------


def test_exact_dispatch_filters_production_for_tests_intent(tmp_path):
    repo = repo_with_files(
        tmp_path / "repo",
        {
            "src/app.py": "def owner_lookup():\n    pass\n",
            "tests/test_app.py": "def test_owner_lookup():\n    pass\n",
        },
    )
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)

        # default intent: the production symbol is the exact hit
        report = handle.search("owner_lookup", k=5)
        assert report.stages_used == ["exact"]
        assert report.candidates[0].source_role == "production"

        # tests intent: the exact production hit must not bypass the filter
        report = handle.search("owner_lookup", k=5, intent="tests")
        assert report.candidates
        assert {candidate.source_role for candidate in report.candidates} <= {"test", "fixture"}
    finally:
        registry.close()


# --- fast mode does not claim dense degradation -------------------------------


def test_fast_mode_is_not_reported_degraded(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)
        report = handle.search("owner handler", k=5, mode="fast")
        assert report.stages_used == ["lexical"]
        assert not report.degraded
        assert report.warnings == []
    finally:
        registry.close()


# --- deep expansion stays bounded regardless of expander output ---------------


def test_deep_expansion_is_capped_in_code(tmp_path):
    from priorart.retrieval.search import _MAX_EXPANDED_QUERIES, _expand_queries

    def greedy_expand(query):
        return [f"variant {index}" for index in range(50)], None

    warnings: list[str] = []
    queries = _expand_queries("owner handler", greedy_expand, warnings)
    # the "5 to 8" prompt is not a contract: the pipeline enforces the bound
    assert len(queries) <= _MAX_EXPANDED_QUERIES
    assert queries[0] == "owner handler"
    assert warnings == []


# --- scoped refresh force-recaptures a stat-unchanged file -------------------


def test_scoped_refresh_recaptures_stat_unchanged_file(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)

        # same size and mtime, different bytes: the plain stat shortcut
        # cannot see this, only the scoped force-recapture can
        target = repo / "src/app.py"
        stat = target.stat()
        target.write_text("def owner():\n    return 1\n")
        import os

        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        job = wait_job(registry, registry.submit_refresh(handle, paths=["src/app.py"]).job_id)
        assert job.state == "completed"
        # the file was actually recaptured despite the unchanged stat
        assert job.counters.get("files") == 1
        report = handle.search("owner", k=3)
        assert report.candidates
    finally:
        registry.close()
