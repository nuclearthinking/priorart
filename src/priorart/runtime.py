from __future__ import annotations

from pathlib import Path

from . import embed as embed_mod
from . import expand as expand_mod
from . import rerank as rerank_mod
from .config import Config
from .indexer import index_repo
from .search import SearchReport, search
from .store import connect


class Runtime:
    def __init__(self, repo: Path, config: Config | None = None):
        self.config = config or Config.from_env()
        self.repo = Path(repo).resolve()
        self.conn = connect(self.config.db_path, self.config.embed_dim)
        self.embed = embed_mod.make_embedder(self.config)
        self.expand = expand_mod.make_expander(self.config)
        self.rerank = rerank_mod.make_reranker(self.config)

    def search(self, query: str, k: int = 10) -> SearchReport:
        return search(
            self.conn,
            str(self.repo),
            query,
            k=k,
            expand_fn=self.expand,
            embed_fn=self.embed,
            rerank_fn=self.rerank,
        )

    def reindex(self, rebuild: bool = False) -> dict:
        return index_repo(self.conn, self.repo, embed_fn=self.embed, rebuild=rebuild)
