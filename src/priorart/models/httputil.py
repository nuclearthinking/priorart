from __future__ import annotations

import threading
import time

import httpx


class BudgetExhaustedError(Exception):
    """The shared search deadline passed before this request could start."""


# Single source for client catch-alls: a network or payload-shape failure must
# degrade to a warning, never to an uncaught exception. A budget exhaustion is
# the same kind of recoverable condition for the caller.
REQUEST_ERRORS = (
    httpx.HTTPError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    BudgetExhaustedError,
)


_local = threading.local()


def set_deadline(deadline: float | None) -> None:
    """Arm (or clear) the per-thread monotonic deadline for model calls."""
    _local.deadline = deadline


def _clamp_timeout(timeout: float) -> float:
    deadline = getattr(_local, "deadline", None)
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExhaustedError("search deadline exhausted before the model call")
    return min(timeout, remaining)


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    response = httpx.post(
        url, json=body, headers=auth_headers(api_key), timeout=_clamp_timeout(timeout)
    )
    response.raise_for_status()
    return response.json()
