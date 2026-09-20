"""Model layer: HTTP plumbing plus embedder, query-expander and reranker
clients built from the service config.

Model clients are shared per service profile, not recreated per worktree.
"""

from __future__ import annotations

from priorart.core import Config

from .embed import embedding_space_id, make_embedder
from .embed_cache import EmbedCache, input_hash
from .expand import make_expander
from .httputil import REQUEST_ERRORS, post_json, set_deadline
from .rerank import make_reranker


class ModelClients:
    """Embedding, expansion and reranking collaborators of one config."""

    def __init__(self, config: Config) -> None:
        self.embed = make_embedder(config)
        self.expand = make_expander(config)
        self.rerank = make_reranker(config)

    @property
    def dense_enabled(self) -> bool:
        return self.embed is not None


__all__ = [
    "REQUEST_ERRORS",
    "Config",
    "EmbedCache",
    "ModelClients",
    "embedding_space_id",
    "input_hash",
    "make_embedder",
    "make_expander",
    "make_reranker",
    "post_json",
    "set_deadline",
]
