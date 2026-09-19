from __future__ import annotations

import threading
from pathlib import Path

from . import embed as embed_mod
from . import expand as expand_mod
from . import rerank as rerank_mod
from .config import Config
from .indexer import index_repo
from .search import SearchReport, map_symbols_text, search, status_text
from .store import connect


class Runtime:
    """Single owner of the database connection; serializes search and refresh."""

    def __init__(self, repo: Path, config: Config | None = None):
        self.config = config or Config.from_env()
        self.repo = Path(repo).resolve()
        self.conn = connect(self.config.db_path, self.config.embed_dim)
        self.embed = embed_mod.make_embedder(self.config)
        self.expand = expand_mod.make_expander(self.config)
        self.rerank = rerank_mod.make_reranker(self.config)
        self._lock = threading.Lock()

    def search(self, query: str, k: int = 10) -> SearchReport:
        with self._lock:
            return search(
                self.conn,
                str(self.repo),
                query,
                k=k,
                expand_fn=self.expand,
                embed_fn=self.embed,
                rerank_fn=self.rerank,
            )

    def reindex(self, *, rebuild: bool = False) -> dict:
        with self._lock:
            return index_repo(self.conn, self.repo, embed_fn=self.embed, rebuild=rebuild)

    def status(self) -> str:
        with self._lock:
            return status_text(self.conn, str(self.repo), dense=self.embed is not None)

    def map_symbols(self, path_glob: str, limit: int = 200) -> str:
        with self._lock:
            return map_symbols_text(self.conn, str(self.repo), path_glob, limit)

    def symbol_count(self) -> int:
        with self._lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE repo = ?", (str(self.repo),)
            ).fetchone()[0]
