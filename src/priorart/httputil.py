from __future__ import annotations

import httpx


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    response = httpx.post(url, json=body, headers=auth_headers(api_key), timeout=timeout)
    response.raise_for_status()
    return response.json()
