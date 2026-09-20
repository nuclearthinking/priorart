"""Job manager: one worker thread per repository handle.

``refresh_index`` always returns quickly with a job id; identical requests
join the active job and a stronger request (rebuild during an incremental
run) is queued instead of silently weakening the desired mode. The worker
holds the inter-process writer lock for the duration of one job, so two
priorart processes never index the same worktree concurrently; a job whose
owner died is marked interrupted by the next lock owner.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

from priorart.core import (
    ACTIVE_STATES,
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_DEGRADED,
    JOB_FAILED,
    JOB_INTERRUPTED,
    JOB_QUEUED,
    JOB_RUNNING,
    MODE_REBUILD,
    Job,
    mode_covers,
    new_job_id,
)
from priorart.storage import FileLock

from .pipeline import RefreshCancelledError, index_repo

_HEARTBEAT_EVERY_EVENTS = 32


class JobManager:
    """Owns the job queue, the worker thread and the writer lock."""

    def __init__(  # noqa: PLR0913 - explicit collaborator set
        self,
        root: Path,
        *,
        journal_conn,
        writer_conn,
        embed_fn,
        lock_path: Path,
        embed_cache=None,
        embed_space_id: str = "",
    ) -> None:
        self.root = Path(root)
        self._journal_conn = journal_conn
        self._writer_conn = writer_conn
        self._embed_fn = embed_fn
        self._embed_cache = embed_cache
        self._embed_space_id = embed_space_id
        self._lock = FileLock(lock_path)
        # reentrant: submit() persists while already holding the mutex
        self._mutex = threading.RLock()
        self._wake = threading.Condition(self._mutex)
        self._queue: deque[Job] = deque()
        self._jobs: dict[str, Job] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True, name="priorart-index")
        self._thread.start()

    def submit(self, *, rebuild: bool = False, paths: list[str] | None = None) -> Job:
        """Enqueue a refresh and return its job; joins an equivalent active job."""
        mode = MODE_REBUILD if rebuild else "incremental"
        with self._mutex:
            if self._stop:
                # a shut-down manager cannot run the job: fail loudly
                # instead of stranding it at "queued" forever
                raise RuntimeError("this job manager is shut down")
            for job in self._jobs.values():
                if (
                    job.state in ACTIVE_STATES
                    and paths is None
                    # a full refresh must not join a scoped job: the caller
                    # would follow a job that only processed a subset
                    and job.paths is None
                    and not job.cancel_requested
                    and mode_covers(job.mode, mode)
                ):
                    return job
            job = Job(
                job_id=new_job_id(),
                repo=str(self.root),
                mode=mode,
                paths=paths,
                created_at=time.time(),
                heartbeat_at=time.time(),
            )
            job.counters = {"total": 0, "scanned": 0, "symbols": 0}
            self._jobs[job.job_id] = job
            self._queue.append(job)
            self._persist(job)
            self._wake.notify_all()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._mutex:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Request cancellation; the worker acts on it between work units."""
        with self._mutex:
            job = self._jobs.get(job_id)
            if job is not None and job.state in ACTIVE_STATES:
                job.cancel_requested = True
            return job

    def shutdown(self, *, wait: bool = True) -> None:
        with self._mutex:
            self._stop = True
            self._wake.notify_all()
        if wait:
            self._thread.join(timeout=30)
        # closing the connections under a live worker would abort its next
        # journal write mid-transaction; only a stopped worker owns them now
        if not self._thread.is_alive():
            for conn in (self._journal_conn, self._writer_conn):
                with contextlib.suppress(sqlite3.Error):
                    conn.close()

    def _worker(self) -> None:
        while True:
            with self._mutex:
                while not self._stop and not self._queue:
                    self._wake.wait(timeout=1.0)
                if self._stop:
                    return
                job = self._queue.popleft()
            try:
                self._run(job)
            except Exception as err:  # noqa: BLE001 - a dead worker would strand every later job
                # an unexpected failure outside _run's inner guards must
                # mark the job failed and keep the worker alive
                with self._mutex:
                    job.state = JOB_FAILED
                    job.error = f"{type(err).__name__}: {err}"
                self._persist(job)

    def _run(self, job: Job) -> None:
        if not self._lock.acquire():
            job.state = JOB_FAILED
            job.error = (
                "another priorart process holds the writer lock for this repository; "
                "retry when it finishes"
            )
            self._persist(job)
            return
        try:
            self._reconcile(job.job_id)
            job.state = JOB_RUNNING
            job.phase = "scanning"
            self._persist(job)
            try:
                stats = index_repo(
                    self._writer_conn,
                    self.root,
                    self._embed_fn,
                    rebuild=job.mode == MODE_REBUILD,
                    paths=job.paths,
                    progress=self._progress(job),
                    should_cancel=lambda: job.cancel_requested,
                    embed_cache=self._embed_cache,
                    embed_space_id=self._embed_space_id,
                )
            except RefreshCancelledError:
                job.state = JOB_CANCELLED
            except Exception as err:  # noqa: BLE001 - job failure must be diagnosable
                job.state = JOB_FAILED
                job.error = f"{type(err).__name__}: {err}"
            else:
                job.counters.update(
                    {
                        "files": stats["files"],
                        "symbols": stats["symbols"],
                        "removed": stats["removed"],
                        "embedded": stats["embedded"],
                        "cache_hits": stats["cache_hits"],
                        "embed_failures": stats["embed_failures"],
                        "missing_vectors": stats["missing_vectors"],
                    }
                )
                job.warnings = list(stats["warnings"])
                job.epoch = stats["epoch"]
                job.lexical_ready = True
                job.dense_ready = stats["missing_vectors"] == 0 and stats["embed_failures"] == 0
                job.state = JOB_DEGRADED if stats["embed_failures"] else JOB_COMPLETED
            self._persist(job)
        finally:
            self._lock.release()

    def _progress(self, job: Job):
        events = 0

        def progress(phase: str, counters: dict) -> None:
            nonlocal events
            events += 1
            job.phase = phase
            job.heartbeat_at = time.time()
            for key, value in counters.items():
                job.counters[key] = value
            if events % _HEARTBEAT_EVERY_EVENTS == 1:
                self._persist(job)

        return progress

    def _reconcile(self, current_job_id: str) -> None:
        """Mark jobs of dead owners as interrupted (we hold the writer lock).

        Scoped to this manager's repo (a store may in principle host more)
        and excluding the job this call is about to run, so the just-queued
        job does not flicker queued→interrupted→running in the journal.
        """
        with self._mutex:
            self._journal_conn.execute(
                "UPDATE jobs SET state = ? WHERE repo = ? AND job_id != ? AND state IN (?, ?)",
                (JOB_INTERRUPTED, str(self.root), current_job_id, JOB_QUEUED, JOB_RUNNING),
            )
            self._journal_conn.commit()

    def _persist(self, job: Job) -> None:
        # callers include non-worker threads (submit, watcher ticks); the
        # journal connection is shared, so every access is serialized
        with self._mutex:
            self._journal_conn.execute(
                "INSERT INTO jobs (job_id, repo, mode, state, phase, created_at, heartbeat_at, "
                "counters, error, epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET state = excluded.state, "
                "phase = excluded.phase, "
                "heartbeat_at = excluded.heartbeat_at, counters = excluded.counters, "
                "error = excluded.error, epoch = excluded.epoch",
                (
                    job.job_id,
                    job.repo,
                    job.mode,
                    job.state,
                    job.phase,
                    job.created_at,
                    job.heartbeat_at,
                    json.dumps(job.snapshot()["counters"]),
                    job.error,
                    job.epoch,
                ),
            )
            self._journal_conn.commit()


def journal_job(conn, job_id: str) -> Job | None:
    """Read one job from the journal; ``None`` when the id is unknown.

    A restarted service has no in-memory copy of jobs submitted before the
    restart; the journal row is the honest remainder.
    """
    from priorart.core import ACTIVE_STATES, JOB_INTERRUPTED

    row = conn.execute(
        "SELECT job_id, repo, mode, state, phase, created_at, heartbeat_at, "
        "counters, error, epoch FROM jobs WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    job = Job(
        job_id=row[0],
        repo=row[1],
        mode=row[2],
        state=row[3],
        phase=row[4],
        created_at=row[5],
        heartbeat_at=row[6],
        epoch=row[9],
    )
    job.counters = json.loads(row[7]) if row[7] else {}
    job.error = row[8]
    if job.state in ACTIVE_STATES:
        # the owning service died mid-run: the journal row never got a
        # final state, and the next lock owner would reconcile it exactly
        # like this
        job.state = JOB_INTERRUPTED
    return job
