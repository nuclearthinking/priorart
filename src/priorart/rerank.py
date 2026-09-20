from __future__ import annotations

import math
import re

from .config import Config
from .httputil import REQUEST_ERRORS, post_json

RERANK_INSTRUCTION = (
    "Given a developer's feature request, retrieve the existing code symbols "
    "that can be imported, extended or refactored for it. A symbol is relevant "
    "when its responsibility matches the request, not merely when its name or "
    "path looks similar."
)

# Qwen3-Reranker judge template (official usage from the model card): the model
# answers "yes" or "no" after a skipped thinking block, and the rerank score is
# P(yes) / (P(yes) + P(no)) read from the first generated token's logprobs.
RERANK_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
    f"<Instruct>: {RERANK_INSTRUCTION}\n"
    "<Query>: {query}\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

RERANK_YES = "yes"
RERANK_NO = "no"
RERANK_N_PROBS = 20

_TEMPLATE_PLACEHOLDER = re.compile(r"\{(query|document)\}")


def _fill_template(template: str, query: str, document: str) -> str:
    """Substitute both placeholders in a single pass.

    ``re.sub`` with a function does not rescan the replacement, so a literal
    ``{document}`` inside the query (or vice versa) survives as text instead of
    being substituted twice.
    """
    return _TEMPLATE_PLACEHOLDER.sub(
        lambda match: query if match.group(1) == "query" else document, template
    )


def make_reranker(config: Config):
    if not config.rerank_base_url or not config.rerank_model:
        return None
    if config.rerank_protocol == "llama-completion":
        return _make_llama_completion_reranker(config)
    return _make_openai_reranker(config)


def _make_openai_reranker(config: Config):
    url = f"{config.rerank_base_url.rstrip('/')}/rerank"

    def rerank(
        query: str, documents: list[str]
    ) -> tuple[list[tuple[int, float]] | None, str | None]:
        formatted_query = (
            query
            if config.rerank_query_format == "raw"
            else f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {query}"
        )
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
                timeout=300,
            )
        except REQUEST_ERRORS as err:
            return None, f"rerank failed ({err}); kept hybrid order"
        return _parse_results(payload.get("results"))

    return rerank


def _make_llama_completion_reranker(config: Config):
    root = config.rerank_base_url.rstrip("/")
    root = root.removesuffix("/v1")
    url = f"{root}/completion"

    def rerank(
        query: str, documents: list[str]
    ) -> tuple[list[tuple[int, float]] | None, str | None]:
        order = []
        for index, document in enumerate(documents):
            prompt = _fill_template(RERANK_PROMPT_TEMPLATE, query, document)
            try:
                payload = post_json(
                    url,
                    {
                        "prompt": prompt,
                        "n_predict": 1,
                        "temperature": 0,
                        "n_probs": RERANK_N_PROBS,
                    },
                    config.rerank_api_key,
                    timeout=300,
                )
            except REQUEST_ERRORS as err:
                return None, f"rerank failed ({err}); kept hybrid order"
            score, _ = _score_from_completion(payload)
            if score is None:
                return None, "rerank completion had no yes/no logprobs; kept hybrid order"
            order.append((index, score))
        return order, None

    return rerank


def _score_from_completion(payload) -> tuple[float | None, float | None]:
    try:
        probs = payload["completion_probabilities"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None, None
    p_yes = None
    p_no = None
    floor_logprob = None
    for entry in probs:
        try:
            token = entry["token"]
            logprob = float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        if floor_logprob is None or logprob < floor_logprob:
            floor_logprob = logprob
        if token == RERANK_YES and p_yes is None:
            p_yes = math.exp(logprob)
        elif token == RERANK_NO and p_no is None:
            p_no = math.exp(logprob)
    if p_yes is None and p_no is None:
        return None, None
    floor = math.exp(floor_logprob) if floor_logprob is not None else 0.0
    p_yes = floor if p_yes is None else p_yes
    p_no = floor if p_no is None else p_no
    return p_yes / (p_yes + p_no), floor_logprob


def _parse_results(results) -> tuple[list[tuple[int, float]] | None, str | None]:
    if not isinstance(results, list) or not results:
        return None, "rerank returned no results; kept hybrid order"
    order = []
    malformed = 0
    for item in results:
        if not isinstance(item, dict):
            malformed += 1
            continue
        raw = item.get("relevance_score", item.get("score"))
        if raw is None:
            malformed += 1
            continue
        try:
            index = item["index"]
            score = float(raw)
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
