from __future__ import annotations

import json
import re

from priorart.core import Config

from .httputil import REQUEST_ERRORS, post_json

EXPAND_PROMPT = (
    "You turn a developer's feature request into code search queries for one "
    "repository. Reply with a JSON array of 5 to 8 short strings: likely existing "
    "symbol names (snake_case and camelCase variants), key technical nouns, and "
    "short English phrases. Reply with the JSON array only."
)


def make_expander(config: Config):
    if not config.llm_base_url or not config.llm_model:
        return None
    url = f"{config.llm_base_url.rstrip('/')}/chat/completions"

    def expand(query: str) -> tuple[list[str], str | None]:
        body = {
            "model": config.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": EXPAND_PROMPT},
                {"role": "user", "content": query},
            ],
        }
        try:
            payload = post_json(url, body, config.llm_api_key, timeout=60)
            content = payload["choices"][0]["message"]["content"]
        except REQUEST_ERRORS as err:
            return [query], f"query expansion failed ({err}); used raw query"
        if not isinstance(content, str):
            return [query], "query expansion returned no content; used raw query"
        queries = _parse_queries(content)
        if not queries:
            return [query], "query expansion returned unparseable output; used raw query"
        return queries, None

    return expand


def _parse_queries(content: str) -> list[str] | None:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    queries = [str(item).strip() for item in parsed if str(item).strip()]
    return queries or None
