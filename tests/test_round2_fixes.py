"""Regression tests for the second bug-hunt round.

Pinned defects: the watcher resubmitting unpublishable drift forever, the
missing failure backoff, a restarted daemon bricking a RemoteRegistry, and
submit into a shut-down job manager stranding the job at "queued".
"""

from __future__ import annotations

import time
from pathlib import Path

from priorart.core import PriorartError
from priorart.registry import RuntimeRegistry
from tests.helpers import DaemonFixture, git, make_config, repo_with_files, wait_job

# --- watcher does not loop on unpublishable drift ---------------------------


def test_watcher_does_not_resubmit_unpublishable_drift(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)

        # a tracked file the parser cannot handle: the refresh completes
        # "successfully" but the file never enters the published state
        (repo / "src" / "broken.py").write_text("def broken(:\n")
        git(repo, "add", "src/broken.py")

        submits: list[list[str] | None] = []

        def submit(paths):
            job = registry.submit_refresh(handle, paths=paths)
            submits.append(paths)
            return job

        from priorart.indexing.watcher import start_watcher

        stop = start_watcher(
            repo,
            str(repo.resolve()),
            open_state=handle._published_file_state,
            submit=submit,
            interval=0.02,
            debounce=0.0,
            content_interval=999.0,
        )
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(submits) < 1:
                time.sleep(0.02)
            assert submits, "watcher never submitted the drift"
            first = submits[0]
            time.sleep(0.5)
            # exactly one submission: the parse-broken file was processed
            # and left unpublished; resubmitting it every tick is the bug
            assert submits == [first]
        finally:
            stop()
    finally:
        registry.close()


# --- watcher backs off after a failed job -----------------------------------


def test_watcher_backs_off_after_failed_job(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        wait_job(registry, registry.submit_refresh(handle).job_id)

        (repo / "src/app.py").write_text("def owner():\n    return 1\n")

        # every submitted job fails: the submit callable returns a fake
        # failed job object
        class _Failed:
            state = "failed"

        from priorart.indexing.watcher import start_watcher

        submits = []

        def failing_submit(paths):
            submits.append(time.monotonic())
            return _Failed()

        start = time.monotonic()
        stop = start_watcher(
            repo,
            str(repo.resolve()),
            open_state=handle._published_file_state,
            submit=failing_submit,
            interval=0.02,
            debounce=0.0,
            content_interval=999.0,
        )
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and len(submits) < 3:
                time.sleep(0.01)
            assert len(submits) >= 3
            # exponential spacing: submit #3 must be delayed well past the
            # raw watch interval (2**2 = 4 intervals = 80ms minimum)
            assert submits[2] - start >= 0.08
        finally:
            stop()
    finally:
        registry.close()


# --- a restarted daemon does not brick the remote registry --------------------


def test_remote_registry_reconnects_after_daemon_restart(tmp_path):
    import tempfile

    socket_dir = Path(tempfile.mkdtemp(prefix="pa-daemon2-", dir="/tmp"))
    socket_path = socket_dir / "d.sock"
    try:
        with DaemonFixture(tmp_path, socket=socket_path) as registry:
            repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
            handle = registry.resolve(repo)
            wait_job(registry, registry.submit_refresh(handle).job_id)
            assert handle.search("owner", k=3).candidates

            # daemon dies and comes back: the same registry and handle
            # must transparently reconnect instead of failing forever.
            # The client's connection is closed first so the in-process
            # serve thread can drain and release the singleton lock (a
            # real daemon death frees the flock via the OS immediately).
            registry._client.close()
        with DaemonFixture(tmp_path, socket=socket_path):
            # reconnect through the ORIGINAL registry: its client is dead,
            # the daemon is new; the factory must rebuild the connection
            assert handle.search("owner", k=3).candidates
    finally:
        import shutil

        shutil.rmtree(socket_dir, ignore_errors=True)


# --- submit after shutdown fails loudly --------------------------------------


def test_submit_after_shutdown_raises(tmp_path):
    repo = repo_with_files(tmp_path / "repo", {"src/app.py": "def owner():\n    pass\n"})
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    wait_job(registry, job.job_id)
    handle.close()
    try:
        handle.refresh()
        raise AssertionError("refresh after close must not silently queue")
    except PriorartError as err:
        code = err.code
    assert code == "HANDLE_CLOSED"
