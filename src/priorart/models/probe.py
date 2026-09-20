"""Endpoint health probes mirroring the model clients.

The URL and payload construction lives next to the clients it mirrors, so
an endpoint change cannot leave the doctor probing a stale path.
"""

from __future__ import annotations

import httpx

from priorart.core import Config

from .httputil import REQUEST_ERRORS, post_json

PROBE_TIMEOUT_SECONDS = 5.0


def probe_endpoints(config: Config) -> dict[str, str]:
    """One status line per model endpoint: configured/reachable/unreachable.

    Any HTTP answer proves reachability; a fully valid probe payload for
    every endpoint would duplicate each client's contract.
    """
    report: dict[str, str] = {}
    for name in ("embed", "rerank", "llm"):
        url = getattr(config, f"{name}_base_url")
        key = getattr(config, f"{name}_api_key")
        auth = "key set" if key else "no key"
        if not url:
            report[name] = "not configured"
            continue
        try:
            post_json(*_probe_request(name, config, url), key, timeout=PROBE_TIMEOUT_SECONDS)
            report[name] = f"reachable ({auth})"
        except httpx.HTTPStatusError:
            report[name] = f"reachable ({auth}; probe rejected, endpoint answered)"
        except REQUEST_ERRORS as err:
            report[name] = f"unreachable ({auth}): {type(err).__name__}"
    return report


def _probe_request(name: str, config: Config, url: str) -> tuple[str, dict]:
    base = url.rstrip("/")
    if name == "embed":
        return f"{base}/embeddings", {"model": config.embed_model, "input": ["doctor"]}
    if name == "rerank":
        return f"{base}/rerank", {}
    return f"{base}/chat/completions", {"model": config.llm_model, "messages": []}
