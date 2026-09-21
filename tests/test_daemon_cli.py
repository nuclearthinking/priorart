"""CLI daemon commands and the doctor daemon probe, exercised in-process."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import typer

from tests.helpers import DaemonFixture


def _env_file(tmp_path: Path, socket: Path, index_dir: Path | None = None) -> Path:
    env_path = tmp_path / "priorart.env"
    env_path.write_text(
        "\n".join(
            [
                f"PRIORART_INDEX_DIR={index_dir or tmp_path / 'indexes'}",
                f"PRIORART_DAEMON_SOCKET={socket}",
                "PRIORART_WATCH_INTERVAL=0",
                "PRIORART_EMBED_DIM=4",
                "",
            ]
        )
    )
    return env_path


def test_doctor_reports_a_reachable_current_daemon(tmp_path, capsys):
    from priorart.cli import doctor

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    try:
        doctor(config=_env_file(tmp_path, daemon.socket))
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)
    out = capsys.readouterr().out
    assert "daemon: reachable, current" in out


def test_doctor_reports_a_daemon_that_refused_this_client(tmp_path, capsys):
    from priorart.cli import doctor

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)
    try:
        # a different index dir is a different service profile: the daemon
        # refuses the client instead of serving mismatched indexes
        doctor(config=_env_file(tmp_path, daemon.socket, index_dir=tmp_path / "other"))
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)
    out = capsys.readouterr().out
    assert "daemon: reachable, but refused this client" in out


def test_doctor_reports_an_unreachable_daemon(tmp_path, capsys):
    from priorart.cli import doctor

    socket_dir = Path(tempfile.mkdtemp(prefix="pa-empty-", dir="/tmp"))
    doctor(config=_env_file(tmp_path, socket_dir / "nobody.sock"))
    out = capsys.readouterr().out
    assert "daemon: not reachable (priorart daemon restart starts one)" in out


def test_daemon_start_runs_the_serve_loop_with_a_sigterm_stop_event(tmp_path, monkeypatch):
    import signal

    from priorart import cli as cli_mod

    installed = []
    monkeypatch.setattr(signal, "signal", lambda num, handler: installed.append((num, handler)))
    served = {}

    def fake_serve(settings, path, stop_event=None):
        served["path"] = path
        served["stop_event"] = stop_event

    monkeypatch.setattr("priorart.coordinator.serve", fake_serve)
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-start-", dir="/tmp"))
    cli_mod.daemon_start(socket_path=socket_dir / "d.sock")
    assert served["path"] == socket_dir / "d.sock"
    assert served["stop_event"] is not None
    assert installed
    assert installed[0][0] == signal.SIGTERM


def test_daemon_stop_command_stops_the_running_daemon(tmp_path, monkeypatch, capsys):
    import priorart.coordinator as coordinator_mod
    from priorart import cli as cli_mod

    daemon = DaemonFixture(tmp_path)
    daemon.thread.start()
    assert daemon.ready.wait(timeout=10)

    def fake_kill(pid, sig):
        daemon.stop.set()

    monkeypatch.setattr(coordinator_mod.os, "kill", fake_kill)
    try:
        cli_mod.stop(socket_path=daemon.socket)
        assert "stopped priorart daemon pid" in capsys.readouterr().out
        daemon.thread.join(timeout=10)
    finally:
        daemon.stop.set()
        daemon.thread.join(timeout=10)


def test_daemon_restart_command_delegates_with_the_config_path(tmp_path, monkeypatch, capsys):
    import priorart.coordinator as coordinator_mod
    from priorart import cli as cli_mod

    restarted = []
    monkeypatch.setattr(
        coordinator_mod,
        "restart_daemon",
        lambda settings, path, config_path=None: (
            restarted.append((settings, path, config_path))
            or "stopped priorart daemon pid 1; restarted priorart daemon on /x"
        ),
    )
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-restart-", dir="/tmp"))
    env_path = _env_file(tmp_path, socket_dir / "d.sock")
    cli_mod.restart(socket_path=socket_dir / "d.sock", config=env_path)
    out = capsys.readouterr().out
    assert "restarted priorart daemon on" in out
    assert len(restarted) == 1
    settings, path, config_path = restarted[0]
    assert path == socket_dir / "d.sock"
    assert config_path == env_path
    assert settings.daemon_socket == str(socket_dir / "d.sock")


def test_daemon_restart_command_fails_cleanly_when_stop_fails(tmp_path, monkeypatch, capsys):
    import priorart.coordinator as coordinator_mod
    from priorart import cli as cli_mod
    from priorart.core import PriorartError
    from priorart.core.errors import DAEMON_STOP_FAILED

    def refused(settings, path, config_path=None):
        raise PriorartError(
            DAEMON_STOP_FAILED,
            "the daemon holds its claim but recorded no pid; stop it manually",
        )

    monkeypatch.setattr(coordinator_mod, "restart_daemon", refused)
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-fail-", dir="/tmp"))
    with pytest.raises(typer.Exit) as exited:
        cli_mod.restart(socket_path=socket_dir / "d.sock")
    assert exited.value.exit_code == 1
    assert "recorded no pid; stop it manually" in capsys.readouterr().err


def test_daemon_restart_command_fails_cleanly_when_no_replacement_becomes_ready(
    tmp_path, monkeypatch, capsys
):
    import priorart.coordinator as coordinator_mod
    from priorart import cli as cli_mod

    def never_ready(settings, path, config_path=None):
        raise ConnectionRefusedError("socket is down")

    monkeypatch.setattr(coordinator_mod, "restart_daemon", never_ready)
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-fail-", dir="/tmp"))
    with pytest.raises(typer.Exit) as exited:
        cli_mod.restart(socket_path=socket_dir / "d.sock")
    assert exited.value.exit_code == 1
    assert "could not restart the priorart daemon" in capsys.readouterr().err


def test_daemon_stop_command_fails_cleanly_on_an_os_error(tmp_path, monkeypatch, capsys):
    import priorart.coordinator as coordinator_mod
    from priorart import cli as cli_mod

    def broken_stop(path):
        raise OSError("lock io failed")

    monkeypatch.setattr(coordinator_mod, "stop_daemon", broken_stop)
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-fail-", dir="/tmp"))
    with pytest.raises(typer.Exit) as exited:
        cli_mod.stop(socket_path=socket_dir / "d.sock")
    assert exited.value.exit_code == 1
    assert "could not stop the priorart daemon" in capsys.readouterr().err
