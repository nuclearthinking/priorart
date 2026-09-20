"""Core layer: error taxonomy, service config, workspace resolution,
response envelopes, application identity and the job lifecycle vocabulary.

Modules outside this package import only the names re-exported here.
"""

from __future__ import annotations

from .app import APP_VERSION
from .config import Config
from .contracts import Envelope
from .errors import (
    AMBIGUOUS_WORKSPACE,
    DAEMON_MISMATCH,
    HANDLE_CLOSED,
    INDEX_NOT_READY,
    INDEX_PROFILE_MISMATCH,
    JOB_NOT_FOUND,
    JOB_REPOSITORY_MISMATCH,
    REPOSITORY_NOT_FOUND,
    REPOSITORY_NOT_SELECTED,
    WORKSPACE_SCOPE_MISMATCH,
    PriorartError,
)
from .git import git_ls_files, git_output
from .jobs import (
    ACTIVE_STATES,
    FINAL_STATES,
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_DEGRADED,
    JOB_FAILED,
    JOB_INTERRUPTED,
    JOB_QUEUED,
    JOB_RUNNING,
    MODE_INCREMENTAL,
    MODE_REBUILD,
    Job,
    mode_covers,
    new_job_id,
)
from .resolver import ResolvedRepo, canonical_root, resolve_repo
from .text import bounded_body

__all__ = [
    "ACTIVE_STATES",
    "AMBIGUOUS_WORKSPACE",
    "APP_VERSION",
    "DAEMON_MISMATCH",
    "FINAL_STATES",
    "HANDLE_CLOSED",
    "INDEX_NOT_READY",
    "INDEX_PROFILE_MISMATCH",
    "JOB_CANCELLED",
    "JOB_COMPLETED",
    "JOB_DEGRADED",
    "JOB_FAILED",
    "JOB_INTERRUPTED",
    "JOB_NOT_FOUND",
    "JOB_QUEUED",
    "JOB_REPOSITORY_MISMATCH",
    "JOB_RUNNING",
    "MODE_INCREMENTAL",
    "MODE_REBUILD",
    "REPOSITORY_NOT_FOUND",
    "REPOSITORY_NOT_SELECTED",
    "WORKSPACE_SCOPE_MISMATCH",
    "Config",
    "Envelope",
    "Job",
    "PriorartError",
    "ResolvedRepo",
    "bounded_body",
    "canonical_root",
    "git_ls_files",
    "git_output",
    "mode_covers",
    "new_job_id",
    "resolve_repo",
]
