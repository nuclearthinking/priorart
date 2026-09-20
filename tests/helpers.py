"""Shared test helpers: git plumbing and runtime configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

from priorart.core.config import Config


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


def init_repo(path: Path) -> Path:
    """Create a git repository directory ready for indexing."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    return path


def make_config(tmp_path: Path, **overrides) -> Config:
    fields = {
        "base_url": None,
        "api_key": None,
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
        "index_dir": tmp_path / "indexes",
        "watch_interval": 0.0,
    }
    fields.update(overrides)
    return Config(**fields)


def wait_job(registry, job_id, timeout: float = 30.0):
    """Poll a job until it reaches a final state."""
    import time

    from priorart.core.jobs import FINAL_STATES

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _handle, job = registry.get_job(job_id)
        if job.state in FINAL_STATES:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def repo_with_files(path: Path, files: dict[str, str]) -> Path:
    """Init a git repo with the given {rel_path: content} committed."""
    repo = init_repo(path)
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", "init")
    return repo


class DaemonFixture:
    """In-process daemon thread wrapper (AF_UNIX path length limits apply)."""

    def __init__(self, tmp_path: Path, *, socket: Path | None = None, **config_overrides) -> None:
        import tempfile
        import threading

        from priorart import coordinator

        self._socket_dir = None
        if socket is not None:
            self.socket = socket
        else:
            self._socket_dir = Path(tempfile.mkdtemp(prefix="pa-daemon-", dir="/tmp"))
            self.socket = self._socket_dir / "d.sock"
        self.config = make_config(tmp_path, daemon_socket=str(self.socket), **config_overrides)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.refused = False
        self.thread = threading.Thread(
            target=self._run,
            args=(coordinator,),
            daemon=True,
        )

    def _run(self, coordinator) -> None:
        try:
            coordinator.serve(
                self.config, self.socket, stop_event=self.stop, ready_event=self.ready
            )
        except SystemExit:
            self.refused = True

    def __enter__(self):
        from priorart.coordinator import DaemonClient, RemoteRegistry, profile_fingerprint

        self.thread.start()
        assert self.ready.wait(timeout=10)
        self.registry = RemoteRegistry(
            lambda: DaemonClient(self.socket, profile_fingerprint(self.config))
        )
        return self.registry

    def __exit__(self, *exc_info) -> None:
        import shutil

        # closing the client first lets the daemon's connection threads
        # finish, so the serve thread can exit inside the join timeout
        if getattr(self, "registry", None) is not None:
            self.registry.close()
        self.stop.set()
        self.thread.join(timeout=10)
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors=True)
