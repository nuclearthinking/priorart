"""Restart hygiene: the operator can always make the daemon current.

Real subprocess daemons drive the CLI surface — stop, restart, doctor —
because a signal aimed at an in-process fixture daemon would hit the test
runner itself. The golden scenario copies the package tree, starts a
daemon from the older copy and proves a current client is refused with a
structured error instead of a framework payload mismatch.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from priorart.coordinator import DaemonClient, profile_fingerprint
from priorart.core import Config
from priorart.core.errors import DAEMON_MISMATCH, PriorartError
from tests.helpers import repo_with_files


def _socket_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="pa-restart-", dir="/tmp"))


def _write_config(config_path: Path, socket_path: Path, index_dir: Path) -> None:
    config_path.write_text(
        "\n".join(
            (
                f"PRIORART_INDEX_DIR={index_dir}",
                f"PRIORART_DAEMON_SOCKET={socket_path}",
                "PRIORART_WATCH_INTERVAL=0",
                "PRIORART_LLM_MODEL=",
                "PRIORART_EMBED_MODEL=",
                "PRIORART_RERANK_MODEL=",
            )
        )
        + "\n"
    )


def _base_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PRIORART_")}
    return {**env, **(extra or {})}


def _run_cli(*args: str, env: dict[str, str] | None = None, timeout: float = 90.0):
    return subprocess.run(  # noqa: S603 - fixed interpreter/module argv
        [sys.executable, "-m", "priorart", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env or _base_env(),
        timeout=timeout,
    )


def _wait_for_socket(socket_path: Path, proc=None, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if socket_path.exists():
            try:
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                probe.connect(str(socket_path))
            except OSError:
                pass
            else:
                probe.close()
                return
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"daemon exited early with {proc.returncode}: {proc.stderr}")
        time.sleep(0.05)
    raise AssertionError(f"daemon socket never became ready: {socket_path}")


def _spawn_daemon(config_path: Path, *, env: dict[str, str] | None = None):
    proc = subprocess.Popen(  # noqa: S603 - fixed interpreter/module argv
        [sys.executable, "-m", "priorart", "daemon", "start", "--config", str(config_path)],
        cwd="/",
        env=env or _base_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    socket_path = _socket_of(config_path)
    _wait_for_socket(socket_path, proc)
    return proc


def _socket_of(config_path: Path) -> Path:
    return Path(Config(_env_file=config_path).daemon_socket).expanduser()


def _client(config_path: Path) -> DaemonClient:
    return DaemonClient(_socket_of(config_path), profile_fingerprint(Config(_env_file=config_path)))


def _stop(config_path: Path) -> str:
    result = _run_cli("daemon", "stop", "--config", str(config_path))
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def workspace(tmp_path):
    socket_dir = _socket_dir()
    socket_path = socket_dir / "priorart.sock"
    config_path = tmp_path / "priorart.env"
    _write_config(config_path, socket_path, tmp_path / "indexes")
    yield config_path
    _stop(config_path)
    shutil.rmtree(socket_dir, ignore_errors=True)


def test_daemon_stop_is_graceful_and_idempotent(workspace):
    proc = _spawn_daemon(workspace)
    socket_path = _socket_of(workspace)
    try:
        out = _stop(workspace)
        assert "stopped priorart daemon pid" in out
        proc.wait(timeout=20)
        assert proc.returncode == 0  # SIGTERM drained and exited cleanly
        assert not socket_path.exists()
        again = _stop(workspace)
        assert "nothing to stop" in again
    finally:
        if proc.poll() is None:
            proc.kill()


def test_daemon_stop_ignores_a_live_pid_when_no_daemon_holds_the_claim(workspace):
    socket_path = _socket_of(workspace)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # a stale lock records a live pid (ours); with no flock holder the stop
    # must not signal it
    socket_path.with_suffix(".lock").write_text(str(os.getpid()))
    out = _stop(workspace)
    assert "nothing to stop" in out


def test_daemon_restart_replaces_the_process(workspace):
    proc = _spawn_daemon(workspace)
    try:
        old_pid = int(_socket_of(workspace).with_suffix(".lock").read_text())
        result = _run_cli("daemon", "restart", "--config", str(workspace))
        assert result.returncode == 0, result.stderr
        assert "restarted priorart daemon" in result.stdout
        proc.wait(timeout=20)  # the original process is gone
        new_pid = int(_socket_of(workspace).with_suffix(".lock").read_text())
        assert new_pid != old_pid
        client = _client(workspace)
        try:
            assert client.call("workspaces") == {"configured": [], "known_indexed": []}
        finally:
            client.close()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_sigkilled_daemon_leaves_a_socket_the_next_start_replaces(workspace):
    proc = _spawn_daemon(workspace)
    socket_path = _socket_of(workspace)
    try:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=20)
        assert socket_path.exists()  # SIGKILL ran no finally block
    finally:
        if proc.poll() is None:
            proc.kill()
    replacement = _spawn_daemon(workspace)
    try:
        client = _client(workspace)
        try:
            assert client.call("workspaces") == {"configured": [], "known_indexed": []}
        finally:
            client.close()
    finally:
        if replacement.poll() is None:
            replacement.kill()


def _stale_package_tree(tmp_path: Path) -> Path:
    import priorart

    package_root = Path(priorart.__file__).resolve().parent
    stale = tmp_path / "stale" / "priorart"
    shutil.copytree(package_root, stale)
    # the edit lives outside core/: the daemon protocol itself changed
    (stale / "coordinator.py").write_text(
        (stale / "coordinator.py").read_text() + "\n# a stale local edit\n"
    )
    return stale.parent


def test_stale_code_daemon_refused_then_replaced_by_autostart(tmp_path, workspace):
    stale_parent = _stale_package_tree(tmp_path)
    proc = _spawn_daemon(workspace, env=_base_env({"PYTHONPATH": str(stale_parent)}))
    config = Config(_env_file=workspace)
    try:
        with pytest.raises(PriorartError) as err:
            _client(workspace)
        assert err.value.code == DAEMON_MISMATCH
        assert "different priorart code" in err.value.message
        assert "priorart daemon restart" in err.value.payload()["next_action"]

        # the MCP boundary surfaces the same refusal as a structured error
        from priorart import server as server_mod

        mcp = server_mod.build_server(None, config=config)
        result = asyncio.run(mcp.call_tool("list_workspaces", {}))
        assert result.is_error is True
        assert result.structured_content["error"]["code"] == DAEMON_MISMATCH
        # the operator stops the stale daemon while it is still running
        assert "stopped priorart daemon pid" in _stop(workspace)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=20)
    # the next client autostarts a current daemon; the same code path serves
    # the documented recovery
    from priorart.coordinator import RemoteRegistry, connect

    registry = RemoteRegistry(lambda: connect(config, config_path=workspace))
    try:
        repo = repo_with_files(
            tmp_path / "repo", {"app.py": "def restart_hygiene_target():\n    pass\n"}
        )
        handle = registry.resolve(repo)
        from tests.helpers import wait_job

        wait_job(registry, registry.submit_refresh(handle).job_id)
        report = handle.search("restart hygiene target")
        assert report.candidates[0].qualname == "restart_hygiene_target"
    finally:
        registry.close()


def test_daemon_restart_survives_a_refresh_job_via_journal(tmp_path, workspace):
    from priorart.coordinator import RemoteRegistry
    from tests.helpers import wait_job

    _spawn_daemon(workspace)
    config = Config(_env_file=workspace)
    registry = RemoteRegistry(
        lambda: DaemonClient(_socket_of(workspace), profile_fingerprint(config))
    )
    try:
        repo = repo_with_files(tmp_path / "repo", {"app.py": "def journal_target():\n    pass\n"})
        handle = registry.resolve(repo)
        job = registry.submit_refresh(handle)
        # the daemon stops while the refresh is in flight or right after it
        assert "stopped priorart daemon pid" in _stop(workspace)
        result = _run_cli("daemon", "restart", "--config", str(workspace))
        assert result.returncode == 0, result.stderr
        _handle, survived = registry.get_job(job.job_id, repo=repo)
        # a fast refresh may complete before the stop lands; the journal
        # must never lose it either way, and an interruption must reconcile
        # as an honestly retryable failure, never as a silent loss
        assert survived.state in {"interrupted", "completed"}
        if survived.state == "interrupted":
            assert survived.failure["code"] == "JOB_INTERRUPTED"
            assert survived.failure["retryable"] is True
        finished = wait_job(registry, registry.submit_refresh(handle).job_id)
        assert finished.state in {"completed", "degraded"}
    finally:
        registry.close()


def test_doctor_reports_daemon_state(workspace):
    # nothing listens yet
    out = _run_cli("doctor", "--config", str(workspace))
    assert out.returncode == 0, out.stderr
    assert "daemon: not reachable" in out.stdout

    proc = _spawn_daemon(workspace)
    try:
        out = _run_cli("doctor", "--config", str(workspace))
        assert "daemon: reachable, current" in out.stdout
    finally:
        if proc.poll() is None:
            proc.kill()


def test_doctor_reports_a_stale_code_daemon(tmp_path, workspace):
    stale_parent = _stale_package_tree(tmp_path)
    proc = _spawn_daemon(workspace, env=_base_env({"PYTHONPATH": str(stale_parent)}))
    try:
        out = _run_cli("doctor", "--config", str(workspace))
        assert out.returncode == 0, out.stderr
        assert "daemon: reachable, but refused this client" in out.stdout
        assert "different priorart code" in out.stdout
    finally:
        if proc.poll() is None:
            proc.kill()


def test_concurrent_restarts_leave_one_daemon(workspace):
    _spawn_daemon(workspace)
    first = subprocess.Popen(  # noqa: S603 - fixed interpreter/module argv
        [sys.executable, "-m", "priorart", "daemon", "restart", "--config", str(workspace)],
        cwd="/",
        env=_base_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    second = subprocess.Popen(  # noqa: S603 - fixed interpreter/module argv
        [sys.executable, "-m", "priorart", "daemon", "restart", "--config", str(workspace)],
        cwd="/",
        env=_base_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert first.wait(timeout=90) == 0
    assert second.wait(timeout=90) == 0
    client = _client(workspace)
    try:
        assert client.call("workspaces") == {"configured": [], "known_indexed": []}
    finally:
        client.close()
    # a third foreground start must refuse: exactly one daemon holds the claim
    try:
        extra = _run_cli("daemon", "start", "--config", str(workspace), timeout=10)
    except subprocess.TimeoutExpired:
        pytest.fail("a third daemon started serving beside the singleton holder")
    assert extra.returncode != 0
    assert "another priorart daemon already listens" in extra.stderr
