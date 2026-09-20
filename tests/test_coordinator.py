"""Coordinator daemon: protocol, thin-client surface, job survival (H).

The daemon owns registries, jobs and watchers; clients are stateless
front-ends. Background indexing must survive a client disconnecting, the
handshake must refuse version mismatches, and domain errors must travel
across the wire as PriorartError codes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from priorart.coordinator import (
    PROTOCOL_VERSION,
    DaemonClient,
    RemoteRegistry,
)
from priorart.coordinator import (
    serve as serve_daemon,
)
from priorart.core.errors import DAEMON_MISMATCH, PriorartError
from priorart.storage.store import APP_VERSION
from tests.helpers import git, init_repo, make_config

SAMPLE = "def daemon_target():\n    pass\n"


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "app.py").write_text(SAMPLE)
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


class _Daemon:
    """Daemon process wrapper for one test."""

    def __init__(self, tmp_path: Path, *, socket: Path | None = None, **config_overrides) -> None:
        # AF_UNIX paths are length-limited (104 bytes on macOS); pytest's
        # tmp dirs exceed that, so sockets live under a short /tmp name
        import tempfile

        self._socket_dir: Path | None = None
        if socket is not None:
            self.socket = socket
        else:
            self._socket_dir = Path(tempfile.mkdtemp(prefix="pa-daemon-", dir="/tmp"))
            self.socket = self._socket_dir / "d.sock"
        self.config = make_config(tmp_path, daemon_socket=str(self.socket), **config_overrides)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.refused = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            serve_daemon(self.config, self.socket, stop_event=self.stop, ready_event=self.ready)
        except SystemExit:
            self.refused = True

    def __enter__(self) -> RemoteRegistry:
        self.thread.start()
        assert self.ready.wait(timeout=10)
        self.registry = RemoteRegistry(lambda: DaemonClient(self.socket))
        return self.registry

    def __exit__(self, *exc_info) -> None:
        # closing the client first lets the daemon's connection threads
        # finish, so the serve thread can exit inside the join timeout
        if getattr(self, "registry", None) is not None:
            self.registry.close()
        self.stop.set()
        self.thread.join(timeout=10)
        if self._socket_dir is not None:
            import shutil

            shutil.rmtree(self._socket_dir, ignore_errors=True)


def _wait_job(registry, job_id: str, timeout: float = 30.0):
    from priorart.core.jobs import FINAL_STATES

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _handle, job = registry.get_job(job_id)
        if job.state in FINAL_STATES:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


# --- handshake ---------------------------------------------------------------


def test_handshake_roundtrip(tmp_path):
    with _Daemon(tmp_path) as registry:
        assert registry.workspaces() == {"context": [], "configured": [], "known_indexed": []}


def test_handshake_refuses_protocol_mismatch(tmp_path):
    daemon = _Daemon(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    client = DaemonClient(daemon.socket)
    try:
        with pytest.raises(PriorartError) as err:
            client.call("handshake", protocol=999, app_version=APP_VERSION)
        assert err.value.code == DAEMON_MISMATCH
    finally:
        client.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_handshake_refuses_app_version_mismatch(tmp_path):
    daemon = _Daemon(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    client = DaemonClient(daemon.socket)
    try:
        with pytest.raises(PriorartError) as err:
            client.call("handshake", protocol=PROTOCOL_VERSION, app_version="0.0.0-not-real")
        assert err.value.code == DAEMON_MISMATCH
    finally:
        client.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


# --- thin-client surface -----------------------------------------------------


def test_remote_search_status_map_roundtrip(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    with _Daemon(tmp_path) as registry:
        handle = registry.resolve(repo)
        assert str(handle.root) == str(repo.resolve())

        with pytest.raises(PriorartError) as err:
            handle.search("daemon_target", k=3)
        assert err.value.code == "INDEX_NOT_READY"

        job = registry.submit_refresh(handle)
        job = _wait_job(registry, job.job_id)
        assert job.state == "completed"
        assert job.lexical_ready

        report = handle.search("daemon_target", k=3)
        assert report.stages_used == ["exact"]
        assert report.candidates[0].qualname == "daemon_target"
        assert report.candidates[0].source_role == "production"

        summary = handle.status()
        assert summary.state == "ready"
        assert summary.freshness in {"fresh", "stale"}
        assert "repo:" in handle.status_text()

        rows, next_cursor = handle.map_symbols("app.py")
        assert [row["qualname"] for row in rows] == ["daemon_target"]
        assert next_cursor is None


def test_domain_errors_travel_across_the_wire(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    with _Daemon(tmp_path) as registry:
        with pytest.raises(PriorartError) as err:
            registry.get_job("no-such-job", repo)
        assert err.value.code == "JOB_NOT_FOUND"

        handle = registry.resolve(repo)
        job = registry.submit_refresh(handle)
        _wait_job(registry, job.job_id)
        with pytest.raises(PriorartError) as err:
            registry.get_job(job.job_id, repo=tmp_path / "other")
        assert err.value.code == "JOB_REPOSITORY_MISMATCH"


# --- job survival (the reason the daemon exists) -----------------------------


def test_background_job_survives_client_disconnect(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    daemon = _Daemon(tmp_path, watch_interval=0.0)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)

    # a client submits a refresh and disconnects immediately
    client = DaemonClient(daemon.socket)
    registry = RemoteRegistry(lambda: DaemonClient(daemon.socket))
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    client.close()

    # the job keeps running in the daemon and a fresh client observes it
    fresh = RemoteRegistry(lambda: DaemonClient(daemon.socket))
    try:
        finished = _wait_job(fresh, job.job_id)
        assert finished.state == "completed"
        report = fresh.resolve(repo).search("daemon_target", k=3)
        assert report.candidates[0].qualname == "daemon_target"
    finally:
        fresh.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_daemon_refuses_second_instance_on_same_socket(tmp_path):
    first = _Daemon(tmp_path)
    first.thread.start()
    assert first.ready.wait(timeout=10)
    try:
        second = _Daemon(tmp_path, socket=first.socket)
        second.thread.start()
        # the refusal happens inside the second thread: it exits without
        # ever becoming ready
        deadline = time.monotonic() + 5
        while second.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not second.thread.is_alive()
        assert not second.ready.is_set()
        assert second.refused

        # the first daemon still owns the socket and serves requests
        client = DaemonClient(first.socket)
        try:
            assert client.call("handshake", protocol=PROTOCOL_VERSION, app_version=APP_VERSION) == {
                "app_version": APP_VERSION
            }
        finally:
            client.close()
    finally:
        first.stop.set()
        first.thread.join(timeout=10)
