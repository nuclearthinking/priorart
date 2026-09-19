from __future__ import annotations

import httpx

from .config import Config
from .httputil import post_json

RERANK_INSTRUCTION = (
    "Given a developer's feature request, retrieve the existing code symbols "
    "that can be imported, extended or refactored for it. A symbol is relevant "
    "when its responsibility matches the request, not merely when its name or "
    "path looks similar."
)


def make_reranker(config: Config):
    if not config.rerank_base_url or not config.rerank_model:
        return None
    url = f"{config.rerank_base_url.rstrip('/')}/rerank"

    def rerank(
        query: str, documents: list[str]
    ) -> tuple[list[tuple[int, float]] | None, str | None]:
        formatted_query = f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}"
        try:
            payload = post_json(
                url,
                {
                    "model": config.rerank_model,
                    "query": formatted_query,
                    "documents": documents,
                    "top_n": len(documents),
                    "return_documents": False,
                },
                config.rerank_api_key,
                timeout=60,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as err:
            return None, f"rerank failed ({err}); kept hybrid order"
        return _parse_results(payload.get("results"))

    return rerank


def _parse_results(results) -> tuple[list[tuple[int, float]] | None, str | None]:
    if not isinstance(results, list) or not results:
        return None, "rerank returned no results; kept hybrid order"
    order = []
    malformed = 0
    for item in results:
        if not isinstance(item, dict):
            malformed += 1
            continue
        try:
            index = item["index"]
            score = float(item.get("relevance_score", item.get("score", 0.0)))
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        if not isinstance(index, int) or isinstance(index, bool):
            malformed += 1
            continue
        order.append((index, score))
    if not order:
        return None, "rerank results contained no usable items; kept hybrid order"
    if malformed:
        return order, f"rerank response had {malformed} malformed items"
    return order, None
