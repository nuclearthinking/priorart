"""Stable error taxonomy shared by every layer.

Domain failures cross layer boundaries as ``PriorartError``; adapters render
them as MCP tool errors (``is_error=true``) or CLI exit messages. Unknown
methods and invalid protocol requests stay protocol errors and never use
these codes.
"""

from __future__ import annotations

REPOSITORY_NOT_SELECTED = "REPOSITORY_NOT_SELECTED"
REPOSITORY_NOT_FOUND = "REPOSITORY_NOT_FOUND"
AMBIGUOUS_WORKSPACE = "AMBIGUOUS_WORKSPACE"
INDEX_NOT_READY = "INDEX_NOT_READY"
INDEX_PROFILE_MISMATCH = "INDEX_PROFILE_MISMATCH"
JOB_NOT_FOUND = "JOB_NOT_FOUND"
JOB_REPOSITORY_MISMATCH = "JOB_REPOSITORY_MISMATCH"
WORKSPACE_SCOPE_MISMATCH = "WORKSPACE_SCOPE_MISMATCH"
DAEMON_MISMATCH = "DAEMON_MISMATCH"
DAEMON_PROFILE_MISMATCH = "DAEMON_PROFILE_MISMATCH"
DAEMON_STOP_FAILED = "DAEMON_STOP_FAILED"
HANDLE_CLOSED = "HANDLE_CLOSED"
INVALID_ARGUMENT = "INVALID_ARGUMENT"

_NEXT_ACTIONS = {
    REPOSITORY_NOT_SELECTED: "Select a repository and repeat this call with repo=<absolute path>.",
    REPOSITORY_NOT_FOUND: "Check the path; the repository root must exist and be a git worktree.",
    AMBIGUOUS_WORKSPACE: "Repeat the tool call with an explicit repo.",
    INDEX_NOT_READY: "Call refresh_index for this repo and retry once the job reports lexical_ready.",
    INDEX_PROFILE_MISMATCH: "Run refresh_index to rebuild the index with the current profile.",
    JOB_NOT_FOUND: "Check the job id, or call refresh_index to start a new job.",
    JOB_REPOSITORY_MISMATCH: "Repeat the call with the repo that owns the job.",
    WORKSPACE_SCOPE_MISMATCH: "Repeat the call with the repo fixed for this session.",
    DAEMON_MISMATCH: (
        "Restart the priorart daemon (priorart daemon restart) and retry; "
        "if the daemon is already current, restart the MCP client session instead."
    ),
    DAEMON_PROFILE_MISMATCH: (
        "Use the same config for this socket, choose another socket, or restart the daemon."
    ),
    DAEMON_STOP_FAILED: (
        "Wait for the daemon to exit and run priorart daemon restart again; "
        "if it keeps running, stop its process manually first."
    ),
    HANDLE_CLOSED: "Retry with a fresh handle (restart the session if daemon-backed).",
    INVALID_ARGUMENT: "Correct the argument and retry the call.",
}

_DEFAULT_NEXT_ACTION = "Inspect the error message and adjust the call."

# Error detail keys that are safe to render to clients.
_RENDERABLE_DETAILS = frozenset(
    {
        "accepted",
        "candidates",
        "daemon_app_version",
        "daemon_code_id",
        "daemon_protocol",
        "field",
        "input",
        "job_id",
        "repo",
        "stored_profile",
        "requested_profile",
        "violations",
    }
)


class PriorartError(Exception):
    """A domain failure with a stable wire code."""

    def __init__(self, code: str, message: str, **details) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {key: value for key, value in details.items() if value is not None}

    def payload(self) -> dict:
        """Machine-readable error body for the response envelope."""
        body: dict = {"code": self.code, "message": self.message}
        for key in _RENDERABLE_DETAILS:
            if key in self.details:
                body[key] = self.details[key]
        body["next_action"] = self.details.get("next_action") or _NEXT_ACTIONS.get(
            self.code, _DEFAULT_NEXT_ACTION
        )
        return body
