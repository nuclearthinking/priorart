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
The handshake pins the protocol and the application version: a mismatch is
a hard refusal, never a best-effort mix.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from priorart.core import APP_VERSION, DAEMON_MISMATCH, Job, PriorartError
from priorart.registry import RuntimeRegistry
from priorart.retrieval import IndexSummary, SearchReport

PROTOCOL_VERSION = 1
DEFAULT_SOCKET = "~/.priorart/daemon.sock"
_HANDSHAKE_TIMEOUT = 10.0


# --- serialization -----------------------------------------------------------
# Candidate/SearchReport/IndexSummary carry their own payload()/from_payload()
# (the Job pattern); the daemon and its thin clients share those wire forms.


# --- daemon side -------------------------------------------------------------


def _claim_singleton(socket_path: Path) -> int:
    """Claim the coordinator slot, or refuse to run beside a live daemon.

    An flock'd sidecar file is the single-instance truth: a socket probe
    alone races (a daemon between ``bind`` and ``listen`` looks dead), and
    unlinking its socket would orphan a live process holding writer locks.
    Returns the lock fd: the claim holds exactly as long as the caller (the
    serve loop) keeps it open.
    """
    lock_path = socket_path.with_suffix(".lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        raise SystemExit(f"another priorart daemon already listens on {socket_path}") from None
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
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(8)
    listener.settimeout(0.5)
    stop = stop_event if stop_event is not None else threading.Event()
    connections: list[threading.Thread] = []
    if ready_event is not None:
        ready_event.set()
    try:
        while not stop.is_set():
            try:
                conn, _addr = listener.accept()
            except (TimeoutError, OSError):
                connections = [thread for thread in connections if thread.is_alive()]
                continue
            thread = threading.Thread(target=_serve_connection, args=(conn, registry), daemon=True)
            thread.start()
            connections.append(thread)
    finally:
        listener.close()
        # in-flight requests must finish before the registry and its caches
        # are closed underneath them
        for thread in connections:
            thread.join(timeout=30)
        # unlink only after the drain: a new client may autostart a daemon
        # during the drain window, and removing the socket from under it
        # would orphan it (its flock still held, its socket gone)
        path.unlink(missing_ok=True)
        registry.close()
        # release the singleton claim so a restarted daemon can take it
        os.close(lock_fd)


def _serve_connection(conn: socket.socket, registry: RuntimeRegistry) -> None:
    with conn:
        reader = conn.makefile("r")
        writer = conn.makefile("w")
        for raw_line in reader:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = _dispatch(registry, request)
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


def _dispatch(registry: RuntimeRegistry, request: dict) -> dict:
    op = request.get("op")
    request_id = request.get("id")
    if op == "handshake":
        if request.get("protocol") != PROTOCOL_VERSION or request.get("app_version") != APP_VERSION:
            raise PriorartError(
                DAEMON_MISMATCH,
                "the priorart daemon speaks a different protocol or version; "
                "restart it (priorart daemon)",
                daemon_protocol=request.get("protocol"),
                daemon_app_version=request.get("app_version"),
            )
        return {"id": request_id, "ok": True, "data": {"app_version": APP_VERSION}}
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


def _op_status_text(registry, request):
    handle = registry.resolve(request["repo"])
    return {"text": handle.status_text()}


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
    job = registry.submit_refresh(
        handle, rebuild=request.get("rebuild", False), paths=request.get("paths")
    )
    return job.snapshot()


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
    "status_text": _op_status_text,
    "map_symbols": _op_map_symbols,
    "refresh": _op_refresh,
    "get_job": _op_get_job,
    "cancel_job": _op_cancel_job,
    "workspaces": _op_workspaces,
}


# --- client side -------------------------------------------------------------


class DaemonClient:
    """Line-protocol client of one coordinator connection."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._conn.settimeout(120.0)
        self._conn.connect(str(self.socket_path))
        self._reader = self._conn.makefile("r")
        self._writer = self._conn.makefile("w")
        self._next_id = 0
        self._mutex = threading.Lock()
        try:
            self._handshake()
        except BaseException:
            self._close_handles()
            raise

    def _handshake(self) -> None:
        response = self.call("handshake", protocol=PROTOCOL_VERSION, app_version=APP_VERSION)
        if response.get("app_version") != APP_VERSION:
            raise PriorartError(DAEMON_MISMATCH, "daemon reported a different application version")

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
                # connection (ValueError on the file objects) must surface
                # as a structured domain error, not a raw traceback
                raise PriorartError(
                    DAEMON_MISMATCH, f"the priorart daemon connection broke: {err}"
                ) from err
        if response is None:
            raise PriorartError(DAEMON_MISMATCH, "the priorart daemon closed the connection")
        if response.get("id") != request["id"]:
            raise PriorartError(DAEMON_MISMATCH, "daemon response id does not match the request")
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


def connect(config, *, autostart: bool = True) -> DaemonClient:
    """Connect to the configured daemon, starting one when allowed."""
    path = Path(config.daemon_socket or DEFAULT_SOCKET).expanduser()
    try:
        return DaemonClient(path)
    except (FileNotFoundError, ConnectionRefusedError):
        if not autostart:
            raise
    subprocess.Popen(  # noqa: S603 - fixed priorart argv
        [sys.executable, "-m", "priorart", "daemon", "--socket", str(path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + _HANDSHAKE_TIMEOUT
    while True:
        try:
            return DaemonClient(path)
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

    def status_text(self) -> str:
        return self._registry.call_op("status_text", repo=str(self.root))["text"]

    def map_symbols(self, path_glob: str, limit: int = 100, offset: int = 0):
        payload = self._registry.call_op(
            "map_symbols", repo=str(self.root), path_glob=path_glob, limit=limit, cursor=offset
        )
        return payload["rows"], payload["next_cursor"]

    def index_metadata(self) -> dict:
        return self.status().payload()


class RemoteRegistry:
    """Duck-typed registry surface backed by the daemon process.

    Holds a client factory, not one connection: a restarted daemon must not
    brick an MCP server that resolved handles before the restart. The first
    call on a dead connection transparently reconnects once.
    """

    def __init__(self, client_factory, *, default_repo: str | None = None) -> None:
        self._client_factory = client_factory
        self._client = client_factory()
        self._default_repo = default_repo
        self._reconnect_mutex = threading.Lock()

    def call_op(self, op: str, **args) -> dict:
        """One op over the daemon wire, reconnecting once on a broken link."""
        try:
            return self._client.call(op, **args)
        except PriorartError as err:
            if err.code != DAEMON_MISMATCH:
                raise
        with self._reconnect_mutex:
            old, self._client = self._client, self._client_factory()
            old.close()
            return self._client.call(op, **args)

    def resolve(self, repo: Path | str | None = None) -> RemoteHandle:
        # a startup default (priorart serve <repo>) stays effective in
        # daemon mode instead of silently falling back to daemon-side context
        effective = repo if repo is not None else self._default_repo
        payload = self.call_op("resolve", repo=str(effective) if effective is not None else None)
        return RemoteHandle(self, payload["root"])

    def submit_refresh(self, handle, *, rebuild: bool = False, paths=None) -> Job:
        return Job.from_snapshot(
            self.call_op("refresh", repo=str(handle.root), rebuild=rebuild, paths=paths)
        )

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
        return self.call_op("workspaces")

    def close(self) -> None:
        self._client.close()


def remote_registry(config, *, default_repo: Path | str | None = None) -> RemoteRegistry:
    """Registry talking to the configured daemon socket."""
    if not config.daemon_socket:
        raise PriorartError(
            DAEMON_MISMATCH, "no daemon socket configured; set PRIORART_DAEMON_SOCKET"
        )
    return RemoteRegistry(
        lambda: connect(config),
        default_repo=str(default_repo) if default_repo is not None else None,
    )
