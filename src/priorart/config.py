from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    llm_base_url: str | None
    llm_api_key: str | None
    embed_base_url: str | None
    embed_api_key: str | None
    rerank_base_url: str | None
    rerank_api_key: str | None
    embed_model: str
    embed_dim: int
    rerank_model: str
    llm_model: str
    db_path: Path

    @classmethod
    def from_env(cls) -> Config:
        db = os.environ.get("PRIORART_DB")
        shared_base_url = _optional_env("PRIORART_BASE_URL")
        shared_api_key = _optional_env("PRIORART_API_KEY")
        return cls(
            llm_base_url=_optional_env("PRIORART_LLM_BASE_URL") or shared_base_url,
            llm_api_key=_optional_env("PRIORART_LLM_API_KEY") or shared_api_key,
            embed_base_url=_optional_env("PRIORART_EMBED_BASE_URL") or shared_base_url,
            embed_api_key=_optional_env("PRIORART_EMBED_API_KEY") or shared_api_key,
            rerank_base_url=_optional_env("PRIORART_RERANK_BASE_URL") or shared_base_url,
            rerank_api_key=_optional_env("PRIORART_RERANK_API_KEY") or shared_api_key,
            embed_model=os.environ.get("PRIORART_EMBED_MODEL", ""),
            embed_dim=int(os.environ.get("PRIORART_EMBED_DIM", "1024")),
            rerank_model=os.environ.get("PRIORART_RERANK_MODEL", ""),
            llm_model=os.environ.get("PRIORART_LLM_MODEL", ""),
            db_path=Path(db).expanduser() if db else Path.home() / ".priorart" / "index.db",
        )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
