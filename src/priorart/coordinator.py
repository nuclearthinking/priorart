"""Shared local coordinator: one daemon process owning registries, jobs,
watchers and model clients; thin clients talk to it over a Unix socket.

The daemon exists so background indexing survives a client disconnecting:
jobs, watchers and the embedding cache live in the coordinator process, and
every client (MCP stdio server, future CLI adapters) is a stateless
front-end. Correctness never depends on the daemon: stores, locks and
atomic commits stay valid with or without it, and a dead daemon is simply
restarted — its jobs were reconciled honestly as interrupted.

Protocol: one JSON object per line, responses mirror the request id.
Every op returns ``{"id": n, "ok": true, "data": ...}`` or
``{"id": n, "ok": false, "error": {code, message, ...details}}``; domain
errors travel as ``PriorartError`` codes and are re-raised client-side.
The handshake pins the protocol, the identity of the loaded code and the
effective service profile: a mismatch is a hard refusal, never a
best-effort mix. The reported app version is diagnostic echo for
``priorart doctor`` and is never a gate: a version string can stay the
same while the running code differs (an editable checkout changes on
every edit), which is exactly what the code identity pins.

Client side: one connection per admitted operation — concurrent callers
never serialize behind a shared socket. A broken transport link replays
only provably safe operations (``_REPLAY_SAFE_OPS``); anything else, and
a refresh in particular, surfaces as a structured uncertain-outcome error
instead of silently sending a second request.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from priorart.core import (
    APP_VERSION,
    DAEMON_MISMATCH,
    DAEMON_PROFILE_MISMATCH,
    DAEMON_STOP_FAILED,
    HANDLE_CLOSED,
    Job,
    PriorartError,
    code_identity,
)
from priorart.registry import RuntimeRegistry
from priorart.retrieval import IndexSummary, SearchReport

PROTOCOL_VERSION = 4
DEFAULT_SOCKET = "~/.priorart/daemon.sock"
_HANDSHAKE_TIMEOUT = 10.0

# Operations whose repeat after a broken transport link is provably safe:
# reads and the idempotent cancel. ``refresh`` is absent on purpose: a resend
# could start a second job nobody asked for. Future operations are not
# retryable until explicitly added here.
_REPLAY_SAFE_OPS = frozenset(
    {
        "resolve",
        "search",
        "status",
        "map_symbols",
        "metadata",
        "get_job",
        "workspaces",
        "cancel_job",
    }
)


# --- serialization -----------------------------------------------------------
# Candidate/SearchReport/IndexSummary carry their own payload()/from_payload()
# (the Job pattern); the daemon and its thin clients share those wire forms.


# --- daemon side -------------------------------------------------------------


def _claim_singleton(socket_path: Path) -> int:
    """Claim the coordinator slot, or refuse to run beside a live daemon.

    An flock'd sidecar file is the single-instance truth: a socket probe
    alone races (a daemon between ``bind`` and ``listen`` looks dead), and
    unlinking its socket would orphan a live process holding writer locks.
    The claim records the daemon pid for ``daemon stop``. Returns the lock
    fd: the claim holds exactly as long as the caller (the serve loop)
    keeps it open.
    """
    lock_path = socket_path.with_suffix(".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        raise SystemExit(f"another priorart daemon already listens on {socket_path}") from None
    os.ftruncate(lock_fd, 0)
    os.write(lock_fd, str(os.getpid()).encode())
    # the socket file itself is stale only if no live process holds the lock
    if socket_path.exists():
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.connect(str(socket_path))
            probe.close()
            raise SystemExit(f"another priorart daemon already listens on {socket_path}")
        except OSError:
            socket_path.unlink()
    return lock_fd


def serve(config, socket_path: Path, *, stop_event=None, ready_event=None) -> None:
    """Run the coordinator until ``stop_event`` is set."""
    path = Path(socket_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _claim_singleton(path)
    registry = RuntimeRegistry(config)
    fingerprint = profile_fingerprint(config)
    identity = code_identity()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(8)
    listener.settimeout(0.5)
    stop = stop_event if stop_event is not None else threading.Event()
    connections: list[tuple[socket.socket, threading.Thread]] = []
    if ready_event is not None:
        ready_event.set()
    try:
        while not stop.is_set():
            try:
                conn, _addr = listener.accept()
            except (TimeoutError, OSError):
                connections = [item for item in connections if item[1].is_alive()]
                continue
            thread = threading.Thread(
                target=_serve_connection, args=(conn, registry, fingerprint, identity), daemon=True
            )
            thread.start()
            connections.append((conn, thread))
    finally:
        listener.close()
        # Stop idle readers from keeping the daemon alive, but leave the
        # write side available so an in-flight request can finish its reply.
        # Every handler must exit before shared resources and the singleton
        # claim are released.
        for conn, _thread in connections:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RD)
        for _conn, thread in connections:
            thread.join()
        # unlink only after the drain: a new client may autostart a daemon
        # during the drain window, and removing the socket from under it
        # would orphan it (its flock still held, its socket gone)
        path.unlink(missing_ok=True)
        registry.close()
        # release the singleton claim so a restarted daemon can take it
        os.close(lock_fd)


def _serve_connection(
    conn: socket.socket, registry: RuntimeRegistry, fingerprint: str, code_id: str
) -> None:
    with conn:
        reader = conn.makefile("r")
        writer = conn.makefile("w")
        for raw_line in reader:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = _dispatch(registry, request, fingerprint, code_id)
            except PriorartError as err:
                response = {
                    "id": _request_id(line),
                    "ok": False,
                    "error": {"code": err.code, "message": err.message, **err.details},
                }
            except Exception as err:  # noqa: BLE001 - one bad request must not kill the daemon
                response = {
                    "id": _request_id(line),
                    "ok": False,
                    "error": {"code": "DAEMON_INTERNAL", "message": f"{type(err).__name__}: {err}"},
                }
            writer.write(json.dumps(response) + "\n")
            writer.flush()


def _request_id(line: str):
    try:
        parsed = json.loads(line)
    except ValueError:
        return None
    return parsed.get("id") if isinstance(parsed, dict) else None


def _dispatch(registry: RuntimeRegistry, request: dict, fingerprint: str, code_id: str) -> dict:
    op = request.get("op")
    request_id = request.get("id")
    if op == "handshake":
        if request.get("protocol") != PROTOCOL_VERSION:
            raise PriorartError(
                DAEMON_MISMATCH,
                "the priorart daemon speaks a different protocol; "
                "restart it (priorart daemon restart)",
                daemon_protocol=PROTOCOL_VERSION,
                daemon_app_version=APP_VERSION,
            )
        if request.get("code_id") != code_id:
            raise PriorartError(
                DAEMON_MISMATCH,
                "the priorart daemon runs different priorart code; "
                "restart it (priorart daemon restart)",
                daemon_code_id=code_id,
                daemon_app_version=APP_VERSION,
            )
        if request.get("profile") != fingerprint:
            raise PriorartError(
                DAEMON_PROFILE_MISMATCH,
                "the priorart daemon uses a different effective service configuration",
            )
        return {
            "id": request_id,
            "ok": True,
            "data": {"app_version": APP_VERSION, "code_id": code_id},
        }
    payload = _OPS[op](registry, request) if op in _OPS else None
    if payload is None:
        raise PriorartError("DAEMON_UNKNOWN_OP", f"unknown daemon op {op!r}")
    return {"id": request_id, "ok": True, "data": payload}


def _op_resolve(registry, request):
    handle = registry.resolve(request.get("repo"))
    return {"root": str(handle.root)}


def _op_search(registry, request):
    handle = registry.resolve(request["repo"])
    report = handle.search(
        request["query"],
        k=request.get("k", 10),
        mode=request.get("mode", "balanced"),
        intent=request.get("intent", "implementation"),
    )
    return report.payload()


def _op_status(registry, request):
    handle = registry.resolve(request["repo"])
    return handle.status().payload()


def _op_map_symbols(registry, request):
    handle = registry.resolve(request["repo"])
    rows, next_cursor = handle.map_symbols(
        request.get("path_glob", "*"),
        limit=request.get("limit", 100),
        offset=request.get("cursor", 0) or 0,
    )
    return {"rows": rows, "next_cursor": next_cursor}


def _op_refresh(registry, request):
    handle = registry.resolve(request["repo"])
    job, submission = registry.submit_refresh(
        handle,
        rebuild=request.get("rebuild", False),
        paths=request.get("paths"),
        include_submission=True,
    )
    return {"job": job.snapshot(), "submission": submission}


def _op_metadata(registry, request):
    return registry.resolve(request["repo"]).index_metadata()


def _op_get_job(registry, request):
    handle, job = registry.get_job(request["job_id"], request.get("repo"))
    return {"repo": str(handle.root), "job": job.snapshot()}


def _op_cancel_job(registry, request):
    job = registry.cancel_job(request["job_id"], request.get("repo"))
    return job.snapshot()


def _op_workspaces(registry, _request):
    return registry.workspaces()


_OPS = {
    "resolve": _op_resolve,
    "search": _op_search,
    "status": _op_status,
    "map_symbols": _op_map_symbols,
    "metadata": _op_metadata,
    "refresh": _op_refresh,
    "get_job": _op_get_job,
    "cancel_job": _op_cancel_job,
    "workspaces": _op_workspaces,
}


# --- client side -------------------------------------------------------------


class _BrokenDaemonTransport(PriorartError):  # noqa: N818 - a transport signal, not a domain error name
    """The connection broke locally; the daemon-side outcome is unknown.

    Only local write/read/EOF/framing and response-id failures raise this
    signal. Daemon-side refusals (protocol, loaded code identity, effective
    service profile) stay plain ``PriorartError`` and are never replayed
    automatically.
    """

    def __init__(self, message: str) -> None:
        super().__init__(DAEMON_MISMATCH, message)


class DaemonClient:
    """Line-protocol client of one coordinator connection.

    One client holds one in-flight request at a time (its mutex serializes
    write/flush/read); concurrent callers use one client each.
    """

    def __init__(self, socket_path: Path, profile: str) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.settimeout(120.0)
        self._conn.connect(str(self.socket_path))
        self._reader = self._conn.makefile("r")
        self._writer = self._conn.makefile("w")
        self._next_id = 0
        self._profile = profile
        self._mutex = threading.Lock()
        try:
            self._handshake()
        except BaseException:
            self._close_handles()
            raise

    def _handshake(self) -> None:
        identity = code_identity()
        response = self.call(
            "handshake",
            protocol=PROTOCOL_VERSION,
            app_version=APP_VERSION,
            code_id=identity,
            profile=self._profile,
        )
        if response.get("code_id") != identity:
            raise PriorartError(DAEMON_MISMATCH, "daemon reported a different code identity")

    def call(self, op: str, **args) -> dict:
        with self._mutex:
            self._next_id += 1
            request = {"op": op, "id": self._next_id, **args}
            try:
                self._writer.write(json.dumps(request) + "\n")
                self._writer.flush()
                line = self._reader.readline()
                # parsed here: a daemon death mid-response-line leaves a
                # partial line and JSONDecodeError (a ValueError) in the
                # same guard instead of a raw traceback
                response = json.loads(line) if line else None
            except (OSError, ValueError) as err:
                # a daemon death mid-request (OSError) or a locally closed
                # connection (ValueError on the file objects) is a broken
                # transport link, not a domain refusal
                raise _BrokenDaemonTransport(
                    f"the priorart daemon connection broke: {err}"
                ) from err
        if response is None:
            raise _BrokenDaemonTransport("the priorart daemon closed the connection")
        if response.get("id") != request["id"]:
            raise _BrokenDaemonTransport("daemon response id does not match the request")
        if response.get("ok"):
            return response.get("data", {})
        error = response.get("error", {})
        details = {key: value for key, value in error.items() if key not in ("code", "message")}
        raise PriorartError(
            error.get("code", "DAEMON_INTERNAL"), error.get("message", "daemon error"), **details
        )

    def close(self) -> None:
        with self._mutex:
            self._close_handles()

    def _close_handles(self) -> None:
        # closing the makefile buffers of a dead socket flushes and raises;
        # teardown must not crash on the way out
        with contextlib.suppress(OSError):
            self._reader.close()
        with contextlib.suppress(OSError):
            self._writer.close()
        with contextlib.suppress(OSError):
            self._conn.close()


def spawn_daemon(socket_path: Path, *, config_path: Path | None = None) -> None:
    """Start a detached coordinator daemon listening on ``socket_path``."""
    argv = [sys.executable, "-m", "priorart", "daemon", "start", "--socket", str(socket_path)]
    if config_path is not None:
        argv.extend(("--config", str(Path(config_path).expanduser().resolve())))
    subprocess.Popen(  # noqa: S603 - fixed priorart argv
        argv,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _signal_holder(lock_fd: int) -> tuple[int | None, str | None]:
    """Signal the recorded holder pid; returns ``(pid, message)``.

    ``pid`` is None when the claim's holder could not be signalled at all;
    ``message`` is None when the SIGTERM was delivered and the caller should
    wait for the drain.
    """
    try:
        pid = int(os.pread(lock_fd, 32, 0).strip())
    except ValueError:
        return None, "the daemon holds its claim but recorded no pid; stop it manually"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # the holder exited between the flock probe and the signal
        return None, "no live daemon: nothing to stop"
    except PermissionError:
        return None, f"priorart daemon pid {pid} belongs to another user; stop it manually"
    return pid, None


def stop_daemon(socket_path: Path, *, timeout: float = 60.0) -> str:
    """Stop the daemon on ``socket_path``; idempotent, never signals a stranger.

    The flock'd sidecar is the single-instance truth: acquiring it proves
    no live daemon holds the slot, whatever a stale pid file may record.
    Only a live holder's recorded pid is signalled, with SIGTERM, so the
    daemon drains in-flight work before releasing the socket.
    """
    lock_path = Path(socket_path).expanduser().with_suffix(".lock")
    try:
        lock_fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return "no live daemon: nothing to stop"
    except PermissionError:
        return "the daemon lock file belongs to another user; stop that daemon manually"
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass  # a live daemon holds the claim
        else:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return "no live daemon: nothing to stop"
        pid, refused = _signal_holder(lock_fd)
        if refused is not None:
            return refused
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                time.sleep(0.1)
                continue
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            return f"stopped priorart daemon pid {pid}"
        return (
            f"priorart daemon pid {pid} is still draining after {timeout:.0f}s; "
            "it releases the socket when done"
        )
    finally:
        os.close(lock_fd)


def restart_daemon(config, socket_path: Path, *, config_path: Path | None = None) -> str:
    """Stop the daemon on ``socket_path`` and start a fresh detached one.

    Refuses to spawn beside a daemon it could not stop: the replacement
    would die on the singleton claim and the readiness probe would talk to
    the surviving old daemon.
    """
    path = Path(socket_path).expanduser()
    stopped = stop_daemon(path)
    if stopped != "no live daemon: nothing to stop" and not stopped.startswith(
        "stopped priorart daemon pid"
    ):
        raise PriorartError(DAEMON_STOP_FAILED, stopped)
    spawn_daemon(path, config_path=config_path)
    deadline = time.monotonic() + _HANDSHAKE_TIMEOUT
    while True:
        try:
            client = DaemonClient(path, profile_fingerprint(config))
        except (FileNotFoundError, ConnectionRefusedError, _BrokenDaemonTransport):
            # a broken transport here is a fresh daemon dying mid-handshake
            # (e.g. a concurrent stop): retry like a not-yet-started one
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
        except PriorartError as err:
            # something answers on the socket but refuses this client —
            # most likely a daemon the stop step could not replace
            raise PriorartError(
                err.code,
                f"the daemon on {path} refused the restarted client: {err.message}",
                **err.details,
            ) from err
        else:
            client.close()
            return f"{stopped}; restarted priorart daemon on {path}"


def connect(config, *, autostart: bool = True, config_path: Path | None = None) -> DaemonClient:
    """Connect to the configured daemon, starting one when allowed."""
    path = Path(config.daemon_socket or DEFAULT_SOCKET).expanduser()
    try:
        return DaemonClient(path, profile_fingerprint(config))
    except (FileNotFoundError, ConnectionRefusedError):
        if not autostart:
            raise
    spawn_daemon(path, config_path=config_path)
    deadline = time.monotonic() + _HANDSHAKE_TIMEOUT
    while True:
        try:
            return DaemonClient(path, profile_fingerprint(config))
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


class RemoteHandle:
    """Duck-typed repo handle delegating to the daemon.

    Calls go through the owning registry so a daemon restart between calls
    reconnects once instead of failing every later operation.
    """

    def __init__(self, registry: RemoteRegistry, root: str) -> None:
        self.root = Path(root)
        self._registry = registry

    def search(
        self, query: str, k: int = 10, mode: str = "balanced", intent: str = "implementation"
    ):
        return SearchReport.from_payload(
            self._registry.call_op(
                "search", repo=str(self.root), query=query, k=k, mode=mode, intent=intent
            )
        )

    def status(self) -> IndexSummary:
        return IndexSummary.from_payload(self._registry.call_op("status", repo=str(self.root)))

    def map_symbols(self, path_glob: str, limit: int = 100, offset: int = 0):
        payload = self._registry.call_op(
            "map_symbols", repo=str(self.root), path_glob=path_glob, limit=limit, cursor=offset
        )
        return payload["rows"], payload["next_cursor"]

    def index_metadata(self) -> dict:
        return self._registry.call_op("metadata", repo=str(self.root))


class RemoteRegistry:
    """Duck-typed registry surface backed by the daemon process.

    Every admitted operation owns its connection: one client, one handshake,
    one call, one close — operations never serialize behind a shared socket.
    A broken link replays only operations whose repeat is provably safe;
    ``close()`` drains live calls instead of cutting sockets under them.
    """

    def __init__(self, client_factory, *, default_repo: str | None = None) -> None:
        self._client_factory = client_factory
        self._default_repo = default_repo
        self._active = 0
        self._drained = threading.Condition()
        self._closed = False

    def call_op(self, op: str, **args) -> dict:
        """One op over the daemon wire, replaying only when provably safe."""
        with self._drained:
            if self._closed:
                raise PriorartError(HANDLE_CLOSED, "this priorart registry is closed")
            self._active += 1
        client = None
        try:
            try:
                client = self._connect()
            except _BrokenDaemonTransport:
                # the handshake died before anything was sent: one replay
                # is safe for every operation
                client = self._connect()
            try:
                return client.call(op, **args)
            except _BrokenDaemonTransport:
                if op not in _REPLAY_SAFE_OPS:
                    raise _uncertain_outcome(op) from None
                client.close()
                client = self._connect()
                return client.call(op, **args)
        except _BrokenDaemonTransport as err:
            # retries are exhausted: surface the plain public contract, never
            # the private transport signal type
            raise PriorartError(err.code, err.message) from err
        finally:
            if client is not None:
                client.close()
            with self._drained:
                self._active -= 1
                if self._closed and self._active == 0:
                    self._drained.notify_all()

    def _connect(self):
        """Open one client; an unreachable daemon is a structured failure."""
        try:
            return self._client_factory()
        except OSError as err:
            # connect() already spent its autostart deadline retrying; a
            # raw OSError must not cross the registry surface as a traceback
            raise PriorartError(
                DAEMON_MISMATCH, f"could not reach the priorart daemon: {err}"
            ) from err

    def resolve(self, repo: Path | str | None = None) -> RemoteHandle:
        # a startup default (priorart serve <repo>) stays effective in
        # daemon mode instead of silently falling back to daemon-side context
        effective = repo if repo is not None else self._default_repo
        payload = self.call_op("resolve", repo=str(effective) if effective is not None else None)
        return RemoteHandle(self, payload["root"])

    def submit_refresh(
        self,
        handle,
        *,
        rebuild: bool = False,
        paths=None,
        include_submission: bool = False,
    ) -> Job | tuple[Job, str]:
        payload = self.call_op("refresh", repo=str(handle.root), rebuild=rebuild, paths=paths)
        job = Job.from_snapshot(payload["job"])
        return (job, payload["submission"]) if include_submission else job

    def get_job(self, job_id: str, repo: Path | str | None = None) -> tuple[RemoteHandle, Job]:
        payload = self.call_op(
            "get_job", job_id=job_id, repo=str(repo) if repo is not None else None
        )
        return RemoteHandle(self, payload["repo"]), Job.from_snapshot(payload["job"])

    def cancel_job(self, job_id: str, repo: Path | str | None = None) -> Job:
        return Job.from_snapshot(
            self.call_op("cancel_job", job_id=job_id, repo=str(repo) if repo is not None else None)
        )

    def workspaces(self) -> dict:
        workspaces = self.call_op("workspaces")
        if self._default_repo is not None:
            configured = workspaces.setdefault("configured", [])
            if self._default_repo not in configured:
                configured.append(self._default_repo)
        return workspaces

    def close(self) -> None:
        """Drain: reject new operations, wait for live ones, stay idempotent.

        Live calls close their own connections when they finish; this method
        never closes a socket under a running operation.
        """
        with self._drained:
            self._closed = True
            while self._active:
                self._drained.wait(timeout=0.1)


def _uncertain_outcome(op: str) -> PriorartError:
    """Structured error for a non-replayable op with an unknown outcome."""
    if op == "refresh":
        return PriorartError(
            DAEMON_MISMATCH,
            "the refresh outcome is unknown: the daemon connection broke in flight",
            next_action=(
                "The first refresh may have started. Call refresh_index again to join "
                "the active job or start a new one if it already finished; call "
                "get_index_job if you kept the job id."
            ),
        )
    return PriorartError(
        DAEMON_MISMATCH,
        f"the outcome of daemon op {op!r} is unknown: the connection broke in flight",
        next_action=(
            "The first attempt may have acted; inspect the affected state "
            "before sending the operation again."
        ),
    )


def remote_registry(
    config, *, default_repo: Path | str | None = None, config_path: Path | None = None
) -> RemoteRegistry:
    """Registry talking to the configured or default daemon socket."""
    return RemoteRegistry(
        lambda: connect(config, config_path=config_path),
        default_repo=str(default_repo) if default_repo is not None else None,
    )


def profile_fingerprint(config) -> str:
    """Secret-safe identity of behavior-affecting effective configuration."""
    payload = config.model_dump(mode="json", exclude={"daemon_socket"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
