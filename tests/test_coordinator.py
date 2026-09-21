"""Coordinator daemon: protocol, thin-client surface, job survival (H).

The daemon owns registries, jobs and watchers; clients are stateless
front-ends. Background indexing must survive a client disconnecting, the
handshake must refuse version mismatches, and domain errors must travel
across the wire as PriorartError codes.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from priorart.coordinator import (
    _OPS,
    _REPLAY_SAFE_OPS,
    PROTOCOL_VERSION,
    DaemonClient,
    RemoteRegistry,
    _BrokenDaemonTransport,
    profile_fingerprint,
)
from priorart.core import Config, code_identity
from priorart.core.errors import DAEMON_MISMATCH, HANDLE_CLOSED, PriorartError
from priorart.retrieval import format_status
from priorart.storage.store import APP_VERSION
from tests.helpers import DaemonFixture, git, init_repo, make_config, wait_job

SAMPLE = "def daemon_target():\n    pass\n"


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "app.py").write_text(SAMPLE)
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


class _FakeClient:
    """Scripted daemon client: records calls and closes."""

    def __init__(self, behavior) -> None:
        self.calls: list[str] = []
        self.closed = 0
        self._behavior = behavior

    def call(self, op: str, **_args) -> dict:
        self.calls.append(op)
        return self._behavior(op)

    def close(self) -> None:
        self.closed += 1


# --- handshake ---------------------------------------------------------------


def test_handshake_roundtrip(tmp_path):
    with DaemonFixture(tmp_path) as registry:
        assert registry.workspaces() == {"configured": [], "known_indexed": []}


def test_handshake_refuses_protocol_mismatch(tmp_path):
    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    client = DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
    try:
        with pytest.raises(PriorartError) as err:
            client.call("handshake", protocol=999, app_version=APP_VERSION)
        assert err.value.code == DAEMON_MISMATCH
        # the mismatch details report the daemon's own values, not the
        # client's request (999 here) or its app version
        assert err.value.payload()["daemon_protocol"] == PROTOCOL_VERSION
        assert err.value.payload()["daemon_app_version"] == APP_VERSION
    finally:
        client.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_handshake_refuses_code_identity_mismatch(tmp_path):
    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    client = DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
    try:
        with pytest.raises(PriorartError) as err:
            client.call("handshake", protocol=PROTOCOL_VERSION, code_id="0" * 64)
        assert err.value.code == DAEMON_MISMATCH
        assert "different priorart code" in err.value.message
    finally:
        client.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_handshake_refuses_effective_profile_mismatch(tmp_path):
    from priorart.core.errors import DAEMON_PROFILE_MISMATCH

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    try:
        with pytest.raises(PriorartError) as err:
            DaemonClient(daemon.socket, "not-the-daemon-profile")
        assert err.value.code == DAEMON_PROFILE_MISMATCH
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_mcp_stays_alive_and_returns_structured_profile_mismatch(tmp_path):
    from priorart import server as server_mod

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    different = make_config(
        tmp_path,
        daemon_socket=str(daemon.socket),
        index_dir=tmp_path / "different-indexes",
    )
    try:
        mcp = server_mod.build_server(None, config=different)
        result = asyncio.run(mcp.call_tool("list_workspaces", {}))
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == "DAEMON_PROFILE_MISMATCH"
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_profile_fingerprint_ignores_socket_but_covers_index_root(tmp_path):
    first_socket = str(tmp_path / "first.sock")
    first = make_config(tmp_path, daemon_socket=first_socket)
    other_socket = make_config(tmp_path, daemon_socket=str(tmp_path / "second.sock"))
    other_index = make_config(
        tmp_path, daemon_socket=first_socket, index_dir=tmp_path / "other-indexes"
    )
    assert profile_fingerprint(first) == profile_fingerprint(other_socket)
    assert profile_fingerprint(first) != profile_fingerprint(other_index)


def test_autostart_forwards_explicit_config_path(tmp_path, monkeypatch):
    from priorart import coordinator

    config_path = tmp_path / "priorart.env"
    config_path.write_text("PRIORART_WATCH_INTERVAL=0\n")
    config = make_config(tmp_path, daemon_socket=str(tmp_path / "missing.sock"))
    attempts = 0
    expected = object()
    launched = []

    def fake_client(_path, _profile):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError
        return expected

    monkeypatch.setattr(coordinator, "DaemonClient", fake_client)
    monkeypatch.setattr(
        coordinator.subprocess, "Popen", lambda argv, **kwargs: launched.append(argv)
    )
    assert coordinator.connect(config, config_path=config_path) is expected
    assert launched == [
        [
            coordinator.sys.executable,
            "-m",
            "priorart",
            "daemon",
            "start",
            "--socket",
            str(tmp_path / "missing.sock"),
            "--config",
            str(config_path.resolve()),
        ]
    ]


def test_two_clients_concurrently_autostart_one_daemon(tmp_path, monkeypatch):
    from priorart import coordinator

    socket_dir = Path(tempfile.mkdtemp(prefix="pa-autostart-", dir="/tmp"))
    socket_path = socket_dir / "daemon.sock"
    config_path = tmp_path / "autostart.env"
    config_path.write_text(
        "\n".join(
            (
                f"PRIORART_INDEX_DIR={tmp_path / 'indexes'}",
                f"PRIORART_DAEMON_SOCKET={socket_path}",
                "PRIORART_WATCH_INTERVAL=0",
                "PRIORART_LLM_MODEL=",
                "PRIORART_EMBED_MODEL=",
                "PRIORART_EMBED_DIM=4",
                "PRIORART_RERANK_MODEL=",
            )
        )
        + "\n"
    )
    config = Config(_env_file=config_path)
    real_popen = coordinator.subprocess.Popen
    launched = []
    launched_lock = threading.Lock()

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        with launched_lock:
            launched.append(process)
        return process

    monkeypatch.setattr(coordinator.subprocess, "Popen", capture_popen)
    barrier = threading.Barrier(3)
    clients = []
    failures = []

    def connect_client():
        try:
            barrier.wait(timeout=10)
            clients.append(coordinator.connect(config, config_path=config_path))
        except BaseException as err:  # noqa: BLE001 - thread evidence returned to test
            failures.append(err)

    threads = [threading.Thread(target=connect_client) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=15)
        assert not failures
        assert len(clients) == 2
        assert all(
            client.call("workspaces") == {"configured": [], "known_indexed": []}
            for client in clients
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and sum(p.poll() is None for p in launched) != 1:
            time.sleep(0.02)
        assert sum(process.poll() is None for process in launched) == 1
    finally:
        for client in clients:
            client.close()
        for process in launched:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        shutil.rmtree(socket_dir, ignore_errors=True)


# --- thin-client surface -----------------------------------------------------


def test_remote_search_status_map_roundtrip(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    with DaemonFixture(tmp_path) as registry:
        handle = registry.resolve(repo)
        assert str(handle.root) == str(repo.resolve())

        with pytest.raises(PriorartError) as err:
            handle.search("daemon_target", k=3)
        assert err.value.code == "INDEX_NOT_READY"

        job = registry.submit_refresh(handle)
        job = wait_job(registry, job.job_id)
        assert job.state == "completed"
        assert job.lexical_ready

        report = handle.search("daemon_target", k=3)
        assert report.stages_used == ["exact"]
        assert report.candidates[0].qualname == "daemon_target"
        assert report.candidates[0].source_role == "production"

        summary = handle.status()
        assert summary.state == "ready"
        assert summary.freshness in {"fresh", "stale"}
        # the summary carries everything the adapter-side text rendering
        # needs: one snapshot, no second daemon read
        assert format_status(summary).startswith("repo: ")

        rows, next_cursor = handle.map_symbols("app.py")
        assert [row["qualname"] for row in rows] == ["daemon_target"]
        assert next_cursor is None


def test_remote_workspaces_include_the_client_startup_default(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    registry = RemoteRegistry(
        lambda: DaemonClient(daemon.socket, profile_fingerprint(daemon.config)),
        default_repo=str(repo),
    )
    try:
        assert str(repo) in registry.workspaces()["configured"]
    finally:
        registry.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_domain_errors_travel_across_the_wire(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    other = _repo_with(tmp_path / "other")
    with DaemonFixture(tmp_path) as registry:
        with pytest.raises(PriorartError) as err:
            registry.get_job("no-such-job", repo)
        assert err.value.code == "JOB_NOT_FOUND"

        handle = registry.resolve(repo)
        job = registry.submit_refresh(handle)
        wait_job(registry, job.job_id)
        with pytest.raises(PriorartError) as err:
            registry.get_job(job.job_id, repo=other)
        assert err.value.code == "JOB_REPOSITORY_MISMATCH"


# --- job survival (the reason the daemon exists) -----------------------------


def test_background_job_survives_client_disconnect(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    daemon = DaemonFixture(tmp_path, watch_interval=0.0)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)

    # a client submits a refresh and disconnects immediately
    registry = RemoteRegistry(
        lambda: DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
    )
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    registry.close()

    # the job keeps running in the daemon and a fresh client observes it
    fresh = RemoteRegistry(lambda: DaemonClient(daemon.socket, profile_fingerprint(daemon.config)))
    try:
        finished = wait_job(fresh, job.job_id)
        assert finished.state == "completed"
        report = fresh.resolve(repo).search("daemon_target", k=3)
        assert report.candidates[0].qualname == "daemon_target"
    finally:
        fresh.close()
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_daemon_refuses_second_instance_on_same_socket(tmp_path):
    first = DaemonFixture(tmp_path)
    first.thread.start()
    assert first.ready.wait(timeout=10)
    try:
        second = DaemonFixture(tmp_path, socket=first.socket)
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
        client = DaemonClient(first.socket, profile_fingerprint(first.config))
        try:
            assert client.call(
                "handshake",
                protocol=PROTOCOL_VERSION,
                code_id=code_identity(),
                profile=profile_fingerprint(first.config),
            ) == {"app_version": APP_VERSION, "code_id": code_identity()}
        finally:
            client.close()
    finally:
        first.stop.set()
        first.thread.join(timeout=10)


def test_graceful_stop_closes_idle_clients_before_releasing_singleton(tmp_path):
    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    client = DaemonClient(daemon.socket, profile_fingerprint(daemon.config))

    daemon.stop.set()
    daemon.thread.join(timeout=10)
    assert not daemon.thread.is_alive()
    with pytest.raises(PriorartError) as closed:
        client.call("workspaces")
    assert closed.value.code == DAEMON_MISMATCH

    replacement = DaemonFixture(tmp_path, socket=daemon.socket)
    replacement.thread.start()
    try:
        assert replacement.ready.wait(timeout=10)
        assert not replacement.refused
    finally:
        client.close()
        replacement.stop.set()
        replacement.thread.join(timeout=10)


# --- transport ownership: replay, drain, parallelism (H) ----------------------


def test_replay_safe_op_retries_once_on_broken_transport():
    clients: list[_FakeClient] = []

    def factory():
        if not clients:
            client = _FakeClient(
                lambda _op: (_ for _ in ()).throw(_BrokenDaemonTransport("link broke"))
            )
        else:
            client = _FakeClient(lambda _op: {"ok": True})
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    assert registry.call_op("workspaces") == {"ok": True}
    assert len(clients) == 2
    assert [client.calls for client in clients] == [["workspaces"], ["workspaces"]]
    assert all(client.closed == 1 for client in clients)


def test_broken_handshake_replays_any_op_before_anything_was_sent():
    clients: list[_FakeClient] = []
    attempts = []

    def factory():
        attempts.append("factory")
        if len(attempts) == 1:
            raise _BrokenDaemonTransport("daemon died during handshake")
        client = _FakeClient(lambda _op: {"ok": True})
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    # refresh is never replayed after a send, but a dead handshake sent
    # nothing: one replay is provably safe
    assert registry.call_op("refresh", repo="/repo") == {"ok": True}
    assert len(attempts) == 2
    assert clients[0].calls == ["refresh"]


def test_executed_refresh_with_lost_response_is_never_replayed():
    wire_attempts: list[str] = []

    def behavior(op):
        wire_attempts.append(op)
        raise _BrokenDaemonTransport("response lost after the daemon executed the op")

    clients = []

    def factory():
        client = _FakeClient(behavior)
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("refresh", repo="/repo")
    assert err.value.code == DAEMON_MISMATCH
    payload = err.value.payload()
    assert "refresh" in payload["next_action"]
    assert "join" in payload["next_action"]
    # one wire attempt: no hidden second job
    assert wire_attempts == ["refresh"]
    assert len(clients) == 1
    assert clients[0].closed == 1


def test_unknown_op_with_broken_transport_is_not_replayed():
    clients: list[_FakeClient] = []

    def factory():
        client = _FakeClient(
            lambda _op: (_ for _ in ()).throw(_BrokenDaemonTransport("link broke"))
        )
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("telemetry")
    assert err.value.code == DAEMON_MISMATCH
    assert "unknown" in err.value.message
    assert "inspect the affected state" in err.value.payload()["next_action"]
    assert len(clients) == 1
    assert clients[0].closed == 1


def test_daemon_refusals_are_never_retried():
    clients: list[_FakeClient] = []

    def factory():
        client = _FakeClient(
            lambda _op: (_ for _ in ()).throw(
                PriorartError(DAEMON_MISMATCH, "daemon speaks a different protocol")
            )
        )
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("workspaces")
    assert err.value.code == DAEMON_MISMATCH
    assert len(clients) == 1
    assert clients[0].closed == 1


def test_concurrent_call_ops_use_distinct_connections_and_overlap():
    barrier = threading.Barrier(2, timeout=10)
    clients: list[_FakeClient] = []
    lock = threading.Lock()

    def factory():
        client = _FakeClient(lambda _op: {"ok": True})
        with lock:
            clients.append(client)
        # a shared serialized connection would keep the second factory
        # waiting until the first op finishes and break this barrier
        barrier.wait()
        return client

    registry = RemoteRegistry(factory)
    results = []

    def run():
        results.append(registry.call_op("workspaces"))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert results == [{"ok": True}, {"ok": True}]
    assert len(clients) == 2
    assert all(client.closed == 1 for client in clients)


def test_close_drains_live_calls_and_rejects_new_ones():
    entered = threading.Event()
    release = threading.Event()
    clients: list[_FakeClient] = []

    def behavior(_op):
        entered.set()
        assert release.wait(timeout=10)
        return {"ok": True}

    def factory():
        client = _FakeClient(behavior)
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    results = []

    def run():
        results.append(registry.call_op("workspaces"))

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(timeout=10)

    closing = threading.Thread(target=registry.close)
    closing.start()
    time.sleep(0.05)
    with pytest.raises(PriorartError) as rejected:
        registry.call_op("workspaces")
    assert rejected.value.code == HANDLE_CLOSED
    # the live socket is not closed under the running call
    assert clients[0].closed == 0

    release.set()
    worker.join(timeout=10)
    closing.join(timeout=10)
    assert results == [{"ok": True}]
    assert clients[0].closed == 1


def test_close_is_idempotent():
    registry = RemoteRegistry(lambda: _FakeClient(lambda _op: {"ok": True}))
    registry.close()
    registry.close()
    with pytest.raises(PriorartError) as rejected:
        registry.call_op("workspaces")
    assert rejected.value.code == HANDLE_CLOSED


def test_replay_allowlist_tracks_the_actual_op_surface():
    assert _OPS.keys() >= _REPLAY_SAFE_OPS
    assert "refresh" not in _REPLAY_SAFE_OPS


def test_unreachable_daemon_is_a_structured_error():
    def factory():
        raise ConnectionRefusedError("socket is gone")

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("workspaces")
    assert err.value.code == DAEMON_MISMATCH
    assert "could not reach the priorart daemon" in err.value.message


def test_exhausted_replay_surfaces_the_public_error_type():
    def broken(_op):
        raise _BrokenDaemonTransport("link broke")

    clients: list[_FakeClient] = []

    def factory():
        client = _FakeClient(broken)
        clients.append(client)
        return client

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("workspaces")
    assert err.value.code == DAEMON_MISMATCH
    assert not isinstance(err.value, _BrokenDaemonTransport)
    assert len(clients) == 2
    assert all(client.closed == 1 for client in clients)


def test_exhausted_handshake_retry_surfaces_the_public_error_type():
    attempts = []

    def factory():
        attempts.append("factory")
        raise _BrokenDaemonTransport("daemon died during handshake")

    registry = RemoteRegistry(factory)
    with pytest.raises(PriorartError) as err:
        registry.call_op("refresh", repo="/repo")
    assert err.value.code == DAEMON_MISMATCH
    assert not isinstance(err.value, _BrokenDaemonTransport)
    assert attempts == ["factory", "factory"]


# --- stale daemon code (protocol 4 pins the loaded code identity) --------------


def _stale_daemon(tmp_path, monkeypatch):
    """Start a daemon whose serve-time code identity is not ours."""
    import priorart.coordinator as coordinator_mod

    monkeypatch.setattr(coordinator_mod, "code_identity", lambda: "stale-daemon-code")
    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    monkeypatch.undo()
    return daemon


def test_stale_daemon_refuses_every_op_with_structured_error(tmp_path, monkeypatch):
    daemon = _stale_daemon(tmp_path, monkeypatch)
    try:
        registry = RemoteRegistry(
            lambda: DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
        )
        for _attempt in (1, 2):  # per-op connections: every call is refused
            with pytest.raises(PriorartError) as err:
                registry.workspaces()
            assert err.value.code == DAEMON_MISMATCH
            assert "different priorart code" in err.value.message
            assert "priorart daemon restart" in err.value.payload()["next_action"]
            # the mismatch details report the daemon's own identity, so an
            # agent can compare it against its running code
            assert err.value.payload()["daemon_code_id"] == "stale-daemon-code"
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_mcp_boundary_returns_structured_mismatch_for_stale_daemon(tmp_path, monkeypatch):
    from priorart import server as server_mod

    daemon = _stale_daemon(tmp_path, monkeypatch)
    same_socket = make_config(
        tmp_path, daemon_socket=str(daemon.socket), index_dir=tmp_path / "indexes"
    )
    try:
        mcp = server_mod.build_server(None, config=same_socket)
        result = asyncio.run(mcp.call_tool("list_workspaces", {}))
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == DAEMON_MISMATCH
        assert "priorart daemon restart" in result.structured_content["error"]["next_action"]
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_daemon_survives_refusals_and_keeps_serving_matching_clients(tmp_path, monkeypatch):
    import priorart.coordinator as coordinator_mod

    daemon = _stale_daemon(tmp_path, monkeypatch)
    try:
        # a current-identity client is refused by the stale daemon...
        with pytest.raises(PriorartError) as err:
            DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
        assert err.value.code == DAEMON_MISMATCH
        # ...a profile-mismatch client is refused...
        with pytest.raises(PriorartError):
            DaemonClient(daemon.socket, "not-the-daemon-profile")
        # ...and the daemon still serves a client that matches it
        monkeypatch.setattr(coordinator_mod, "code_identity", lambda: "stale-daemon-code")
        client = DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
        try:
            assert client.call("workspaces") == {"configured": [], "known_indexed": []}
        finally:
            client.close()
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_registry_recovers_after_daemon_replacement(tmp_path, monkeypatch):
    daemon = _stale_daemon(tmp_path, monkeypatch)
    registry = RemoteRegistry(
        lambda: DaemonClient(daemon.socket, profile_fingerprint(daemon.config))
    )
    try:
        with pytest.raises(PriorartError) as err:
            registry.workspaces()
        assert err.value.code == DAEMON_MISMATCH
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)
    replacement = DaemonFixture(tmp_path, socket=daemon.socket)
    replacement.thread.start()
    assert replacement.ready.wait(timeout=10)
    try:
        # the same registry object recovers: per-op connections re-handshake
        assert registry.workspaces() == {"configured": [], "known_indexed": []}
    finally:
        registry.close()
        replacement.stop.set()
        replacement.thread.join(timeout=10)


def test_singleton_lock_records_the_daemon_pid(tmp_path):
    import os

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    try:
        assert daemon.socket.with_suffix(".lock").read_text().strip() == str(os.getpid())
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


# --- stop/restart hygiene ----------------------------------------------------


def test_handshake_refuses_a_daemon_reporting_foreign_code_identity(tmp_path):
    import json
    import socket as socket_mod
    import tempfile

    sock_dir = Path(tempfile.mkdtemp(prefix="pa-fake-", dir="/tmp"))
    server = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
    server.bind(str(sock_dir / "fake.sock"))
    server.listen(1)

    def reply_with_foreign_identity() -> None:
        conn, _ = server.accept()
        with conn:
            reader = conn.makefile("r")
            writer = conn.makefile("w")
            request = json.loads(reader.readline())
            writer.write(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": True,
                        "data": {"app_version": "9.9.9", "code_id": "0" * 64},
                    }
                )
                + "\n"
            )
            writer.flush()
            reader.close()
            writer.close()
        server.close()

    threading.Thread(target=reply_with_foreign_identity, daemon=True).start()
    with pytest.raises(PriorartError) as err:
        DaemonClient(sock_dir / "fake.sock", "any-profile")
    assert err.value.code == DAEMON_MISMATCH
    assert "different code identity" in err.value.message


def test_stop_daemon_without_a_lock_reports_nothing_to_stop(tmp_path):
    from priorart.coordinator import stop_daemon

    assert stop_daemon(tmp_path / "d.sock") == "no live daemon: nothing to stop"


def test_stop_daemon_ignores_a_lock_nobody_holds(tmp_path):

    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("4194303")
    assert stop_daemon(tmp_path / "d.sock") == "no live daemon: nothing to stop"


def test_stop_daemon_reports_a_holder_without_a_parseable_pid(tmp_path):
    import fcntl
    import os

    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("not-a-pid")
    fd = os.open(tmp_path / "d.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert stop_daemon(tmp_path / "d.sock") == (
            "the daemon holds its claim but recorded no pid; stop it manually"
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_stop_daemon_signals_the_holder_and_waits_for_the_drain(tmp_path, monkeypatch):
    import os
    import signal

    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import stop_daemon

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)

    def fake_kill(pid, sig):
        assert sig == signal.SIGTERM
        daemon.stop.set()

    monkeypatch.setattr(coordinator_mod.os, "kill", fake_kill)
    try:
        assert stop_daemon(daemon.socket) == f"stopped priorart daemon pid {os.getpid()}"
        daemon.thread.join(timeout=10)
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_stop_daemon_reports_draining_when_the_holder_never_releases(tmp_path, monkeypatch):
    import fcntl
    import os

    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("4194303")
    fd = os.open(tmp_path / "d.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        monkeypatch.setattr(coordinator_mod.os, "kill", lambda pid, sig: None)
        message = stop_daemon(tmp_path / "d.sock", timeout=0.3)
        assert "is still draining" in message
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_restart_daemon_spawns_a_replacement_and_confirms_readiness(tmp_path, monkeypatch):
    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import restart_daemon

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    replacement = DaemonFixture(tmp_path, socket=daemon.socket)
    spawned = []

    def fake_spawn(path, *, config_path=None):
        spawned.append((path, config_path))
        replacement.thread.start()
        assert replacement.ready.wait(timeout=10)

    def fake_kill(pid, sig):
        daemon.stop.set()

    monkeypatch.setattr(coordinator_mod, "spawn_daemon", fake_spawn)
    monkeypatch.setattr(coordinator_mod.os, "kill", fake_kill)
    try:
        message = restart_daemon(daemon.config, daemon.socket)
        assert "; restarted priorart daemon on " in message
        assert spawned == [(daemon.socket, None)]
        daemon.thread.join(timeout=10)
    finally:
        daemon.stop.set()
        replacement.stop.set()
        daemon.thread.join(timeout=10)
        replacement.thread.join(timeout=10)


def test_restart_daemon_raises_when_no_replacement_becomes_ready(tmp_path, monkeypatch):
    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import restart_daemon

    monkeypatch.setattr(coordinator_mod, "_HANDSHAKE_TIMEOUT", 0.3)
    import tempfile

    sock_dir = Path(tempfile.mkdtemp(prefix="pa-restart-", dir="/tmp"))
    monkeypatch.setattr(coordinator_mod, "spawn_daemon", lambda path, *, config_path=None: None)
    with pytest.raises(FileNotFoundError):
        restart_daemon(make_config(tmp_path), sock_dir / "d.sock")


@pytest.mark.parametrize(
    "stopped",
    [
        "the daemon holds its claim but recorded no pid; stop it manually",
        "priorart daemon pid 4242 is still draining after 60s; it releases the socket when done",
    ],
)
def test_restart_daemon_refuses_to_spawn_beside_a_daemon_it_could_not_stop(
    tmp_path, monkeypatch, stopped
):
    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import restart_daemon
    from priorart.core.errors import DAEMON_STOP_FAILED

    spawned = []

    def fake_spawn(path, *, config_path=None):
        spawned.append(path)

    monkeypatch.setattr(coordinator_mod, "stop_daemon", lambda path, timeout=60.0: stopped)
    monkeypatch.setattr(coordinator_mod, "spawn_daemon", fake_spawn)
    with pytest.raises(PriorartError) as err:
        restart_daemon(make_config(tmp_path), tmp_path / "d.sock")
    assert err.value.code == DAEMON_STOP_FAILED
    assert err.value.message == stopped
    assert "priorart daemon restart again" in err.value.payload()["next_action"]
    assert not spawned  # never spawn beside a daemon the stop could not replace


def test_restart_daemon_reports_a_refusing_replacement_cleanly(tmp_path, monkeypatch):
    import json
    import socket as socket_mod
    import tempfile

    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import restart_daemon

    sock_dir = Path(tempfile.mkdtemp(prefix="pa-refuse-", dir="/tmp"))
    sock = sock_dir / "d.sock"
    server = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
    server.bind(str(sock))
    server.listen(1)

    def refuse_handshake() -> None:
        conn, _ = server.accept()
        with conn:
            reader = conn.makefile("r")
            writer = conn.makefile("w")
            request = json.loads(reader.readline())
            writer.write(
                json.dumps(
                    {
                        "id": request["id"],
                        "ok": False,
                        "error": {"code": "DAEMON_MISMATCH", "message": "stale daemon"},
                    }
                )
                + "\n"
            )
            writer.flush()
            reader.close()
            writer.close()
        server.close()

    threading.Thread(target=refuse_handshake, daemon=True).start()
    monkeypatch.setattr(coordinator_mod, "spawn_daemon", lambda path, *, config_path=None: None)
    with pytest.raises(PriorartError) as err:
        restart_daemon(make_config(tmp_path), sock)
    assert "refused the restarted client: stale daemon" in err.value.message
    assert err.value.code == DAEMON_MISMATCH


def test_stop_daemon_ignores_a_holder_that_exited_before_the_signal(tmp_path, monkeypatch):
    import fcntl
    import os

    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("4242")
    fd = os.open(tmp_path / "d.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        def exited_before_signal(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(coordinator_mod.os, "kill", exited_before_signal)
        assert stop_daemon(tmp_path / "d.sock") == "no live daemon: nothing to stop"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_stop_daemon_reports_a_holder_of_another_user(tmp_path, monkeypatch):
    import fcntl
    import os

    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("4242")
    fd = os.open(tmp_path / "d.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        def foreign_holder(pid, sig):
            raise PermissionError

        monkeypatch.setattr(coordinator_mod.os, "kill", foreign_holder)
        assert stop_daemon(tmp_path / "d.sock") == (
            "priorart daemon pid 4242 belongs to another user; stop it manually"
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_stop_daemon_reports_a_lock_of_another_user(tmp_path, monkeypatch):
    import priorart.coordinator as coordinator_mod
    from priorart.coordinator import stop_daemon

    (tmp_path / "d.lock").write_text("4242")

    def foreign_open(path, flags):
        raise PermissionError

    monkeypatch.setattr(coordinator_mod.os, "open", foreign_open)
    assert stop_daemon(tmp_path / "d.sock") == (
        "the daemon lock file belongs to another user; stop that daemon manually"
    )
