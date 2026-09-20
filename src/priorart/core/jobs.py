"""Job state: the lifecycle vocabulary of one indexing run.

A job is immutable-bound to one repository and profile. States describe the
run as a whole; phases describe what the worker is doing right now.
``lexical_ready`` and ``dense_ready`` are independent flags: after the
lexical commit the job may still be running while search is already useful.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_DEGRADED = "degraded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
JOB_INTERRUPTED = "interrupted"

ACTIVE_STATES = frozenset({JOB_QUEUED, JOB_RUNNING})
FINAL_STATES = frozenset({JOB_COMPLETED, JOB_DEGRADED, JOB_FAILED, JOB_CANCELLED, JOB_INTERRUPTED})

MODE_INCREMENTAL = "incremental"
MODE_REBUILD = "rebuild"

_MODE_STRENGTH = {MODE_INCREMENTAL: 0, MODE_REBUILD: 1}


def mode_covers(mode: str, other: str) -> bool:
    """Whether an active job in ``mode`` already covers a ``other`` request."""
    return _MODE_STRENGTH[mode] >= _MODE_STRENGTH[other]


def new_job_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Job:
    """One indexing run; mutated only by its owning JobManager."""

    job_id: str
    repo: str
    mode: str
    state: str = JOB_QUEUED
    phase: str | None = None
    created_at: float = 0.0
    heartbeat_at: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    lexical_ready: bool = False
    dense_ready: bool = False
    epoch: int = 0
    cancel_requested: bool = False
    paths: list[str] | None = None

    def snapshot(self) -> dict:
        """Immutable client view of the job."""
        return {
            "job_id": self.job_id,
            "repo": self.repo,
            "mode": self.mode,
            "state": self.state,
            "phase": self.phase,
            "created_at": self.created_at,
            "heartbeat_at": self.heartbeat_at,
            "counters": dict(self.counters),
            "warnings": list(self.warnings),
            "error": self.error,
            "lexical_ready": self.lexical_ready,
            "dense_ready": self.dense_ready,
            "index_epoch": self.epoch,
            "cancel_requested": self.cancel_requested,
            "paths": list(self.paths) if self.paths is not None else None,
        }

    @classmethod
    def from_snapshot(cls, payload: dict) -> Job:
        """Rebuild the client view of a job from its snapshot."""
        return cls(
            job_id=payload["job_id"],
            repo=payload["repo"],
            mode=payload["mode"],
            state=payload["state"],
            phase=payload.get("phase"),
            created_at=payload.get("created_at", 0.0),
            heartbeat_at=payload.get("heartbeat_at", 0.0),
            counters=dict(payload.get("counters", {})),
            warnings=list(payload.get("warnings", [])),
            error=payload.get("error"),
            lexical_ready=payload.get("lexical_ready", False),
            dense_ready=payload.get("dense_ready", False),
            epoch=payload.get("index_epoch", 0),
            cancel_requested=payload.get("cancel_requested", False),
            paths=payload.get("paths"),
        )
