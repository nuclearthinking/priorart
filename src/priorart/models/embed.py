from __future__ import annotations

import hashlib
import json
import math

import sqlite_vec

from priorart.core import APP_VERSION, Config

from .httputil import REQUEST_ERRORS, post_json

BATCH = 32

DOCUMENT_INSTRUCTION = (
    "Represent this source code symbol for reuse search: given a developer's "
    "feature request, retrieve existing functions, classes and methods that "
    "could be imported, extended or refactored instead of writing new code."
)
QUERY_INSTRUCTION = (
    "Given a developer's feature request, retrieve existing functions, classes "
    "and methods in this repository that could be reused or extended instead "
    "of writing new code."
)


def embedding_space_id(config: Config) -> str:
    """Identity of the embedding space one configuration produces.

    The model name and dimension alone do not define a space: the endpoint,
    input formatting, instructions and normalization change the meaning of
    vectors. Every input that shapes the produced bytes is hashed here, so a
    template or instruction edit invalidates cached vectors automatically.
    """

    profile = {
        "app_version": APP_VERSION,
        "base_url": config.embed_base_url,
        "model": config.embed_model,
        "dim": config.embed_dim,
        "input_format": config.embed_input_format,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "query_instruction": QUERY_INSTRUCTION,
        "normalized": config.embed_input_format == "qwen3",
    }
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _instruct(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nText: {text}"


def _qwen3_query(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nQuery: {text}"


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _in_input_order(data) -> list:
    """Follow the OpenAI ``index`` field when the endpoint provides it.

    Providers are allowed to return ``data`` out of order; trusting the list
    order would silently attach the wrong vectors to the wrong texts.
    """
    try:
        return sorted(data, key=lambda item: item["index"])
    except (KeyError, TypeError, ValueError):
        return data


def make_embedder(config: Config):
    if not config.embed_base_url or not config.embed_model:
        return None
    url = f"{config.embed_base_url.rstrip('/')}/embeddings"

    def embed(texts: list[str], *, query: bool = False) -> tuple[list[bytes] | None, str | None]:
        if config.embed_input_format == "qwen3":
            formatted = (
                [_qwen3_query(QUERY_INSTRUCTION, text) for text in texts] if query else list(texts)
            )
        else:
            instruction = QUERY_INSTRUCTION if query else DOCUMENT_INSTRUCTION
            formatted = [_instruct(instruction, text) for text in texts]
        vectors: list[bytes] = []
        for start in range(0, len(formatted), BATCH):
            batch = formatted[start : start + BATCH]
            try:
                payload = post_json(
                    url,
                    {
                        "model": config.embed_model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                    config.embed_api_key,
                    timeout=120,
                )
                batch_vectors = [
                    sqlite_vec.serialize_float32(
                        _normalize(item["embedding"])
                        if config.embed_input_format == "qwen3"
                        else item["embedding"]
                    )
                    for item in _in_input_order(payload["data"])
                ]
            except REQUEST_ERRORS as err:
                return None, f"embedding request failed ({err}); dense search skipped"
            if len(batch_vectors) != len(batch):
                return None, "embedding endpoint returned wrong payload; dense search skipped"
            for vector in batch_vectors:
                if len(vector) != config.embed_dim * 4:
                    return (
                        None,
                        (
                            f"embedding endpoint returned dimension {len(vector) // 4}, "
                            f"expected {config.embed_dim}; dense search skipped"
                        ),
                    )
            vectors.extend(batch_vectors)
        return vectors, None

    return embed
