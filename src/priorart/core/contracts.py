"""Response envelope shared by the MCP and CLI adapters.

Every repository-scoped response carries ``ok``, ``repo``, ``index``,
``data``, ``warnings`` and ``timings``. Error responses replace ``data``
with an ``error`` body and set ``ok=false``. Adapters render the same
envelope as text (MCP ``content``) and as a machine contract (MCP
``structuredContent``).
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

    def text(self) -> str:
        lines: list[str] = []
        if self.error is not None:
            lines.append(self.error["message"])
            lines.extend(
                f"{key}: {self.error[key]}"
                for key in ("candidates", "input", "job_id")
                if key in self.error
            )
            lines.append(self.error["next_action"])
            return "\n".join(lines)
        lines.extend(f"warning: {warning}" for warning in self.warnings)
        if self.data.get("text") is not None:
            lines.append(self.data["text"].rstrip())
        return "\n".join(lines)
