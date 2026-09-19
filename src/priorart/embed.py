from __future__ import annotations

import httpx
import sqlite_vec

from .config import Config
from .httputil import post_json

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


def _instruct(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nText: {text}"


def make_embedder(config: Config):
    if not config.embed_base_url or not config.embed_model:
        return None
    url = f"{config.embed_base_url.rstrip('/')}/embeddings"

    def embed(texts: list[str], *, query: bool = False) -> tuple[list[bytes] | None, str | None]:
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
                    sqlite_vec.serialize_float32(item["embedding"]) for item in payload["data"]
                ]
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as err:
                return None, f"embedding request failed ({err}); dense search skipped"
            if len(batch_vectors) != len(batch):
                return None, "embedding endpoint returned wrong payload; dense search skipped"
            vectors.extend(batch_vectors)
        return vectors, None

    return embed
