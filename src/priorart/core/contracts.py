"""Response envelope shared by the MCP and CLI adapters.

Every repository-scoped response carries ``ok``, ``repo``, ``index``,
``data``, ``warnings`` and ``timings``. Error responses replace ``data``
with an ``error`` body and set ``ok=false``. The envelope is a pure machine
contract: adapters render their own human text from the typed results and
never store presentation in ``data``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import PriorartError


@dataclass
class Envelope:
    ok: bool
    repo: str | None = None
    index: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @classmethod
    def success(
        cls,
        repo: str | None,
        data: dict[str, Any],
        *,
        index: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        timings: dict[str, float] | None = None,
    ) -> Envelope:
        return cls(
            ok=True,
            repo=repo,
            index=index,
            data=data,
            warnings=list(warnings or []),
            timings=dict(timings or {}),
        )

    @classmethod
    def failure(cls, error: PriorartError, *, repo: str | None = None) -> Envelope:
        return cls(ok=False, repo=repo, error=error.payload())

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"ok": self.ok, "repo": self.repo}
        if self.error is not None:
            body["error"] = self.error
            return body
        body["index"] = self.index
        body["data"] = self.data
        body["warnings"] = self.warnings
        body["timings"] = {key: round(value, 6) for key, value in self.timings.items()}
        return body
