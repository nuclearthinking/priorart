from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HOME_ENV_FILE = Path.home() / ".priorart" / "priorart.env"

DEFAULT_CANDIDATE_LIMIT = 50


class Config(BaseSettings):
    """Service configuration, loaded once from explicit sources.

    Values come from PRIORART_* environment variables, then
    ``~/.priorart/priorart.env``, then an explicit ``--config`` file; real
    environment variables win over both files, and the explicit file wins
    over the home file. There is deliberately no project-local ``.env``:
    resolving the service config must not depend on the process working
    directory or on any indexed repository.

    ``index_dir`` is the root of the per-worktree store tree
    (``<index_dir>/v<layout>/<worktree-id>/<profile-id>.db``); there is no
    single shared index database anymore.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRIORART_",
        env_file=HOME_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    base_url: str | None = Field(
        default=None, description="Shared provider base URL for every service"
    )
    api_key: str | None = Field(default=None, description="Shared provider API key")
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    embed_model: str = Field(default="", description="Embedding model id sent to the provider")
    embed_dim: int = Field(
        default=1024,
        gt=0,
        description="Embedding vector dimension; must match the embedding model output",
    )
    embed_input_format: Literal["instruct-text", "qwen3"] = Field(
        default="instruct-text",
        description=(
            "Embedding input contract: 'instruct-text' wraps queries and documents in "
            "Instruct/Text tags, 'qwen3' follows the official Qwen3-Embedding usage "
            "(Instruct/Query prefix on queries only, raw documents, L2-normalized vectors)"
        ),
    )
    rerank_model: str = Field(default="", description="Reranker model id sent to the provider")
    watch_interval: float = Field(
        default=2.0,
        ge=0,
        description=(
            "Background freshness polling interval in seconds; 0 disables the auto-refresh watcher"
        ),
    )
    watch_debounce: float = Field(
        default=0.5,
        ge=0,
        description="Quiet period in seconds before detected changes trigger a refresh",
    )
    watch_content_interval: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Period in seconds for content-hash reconciliation of files whose "
            "size and mtime did not change"
        ),
    )
    search_deadline_seconds: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Monotonic end-to-end budget of one search: expansion, embedding "
            "and reranking are clamped to the remaining time and degrade to "
            "warnings instead of overrunning"
        ),
    )
    daemon_socket: str | None = Field(
        default=None,
        description=(
            "Unix socket of the shared coordinator daemon; when set, MCP serve "
            "becomes a thin client of that process instead of hosting its own registry"
        ),
    )
    rerank_protocol: Literal["openai", "llama-completion"] = Field(
        default="openai",
        description=(
            "Rerank client protocol: 'openai' posts to the /v1/rerank endpoint, "
            "'llama-completion' scores documents one by one through a llama-server "
            "/completion call and reads P(yes) from the first generated token's "
            "logprobs (Qwen3-Reranker judge template)"
        ),
    )
    rerank_query_format: Literal["instruct", "raw"] = Field(
        default="instruct",
        description=(
            "Rerank query contract: 'instruct' wraps the query in manual "
            "<Instruct>/<Query> tags, 'raw' passes the query unchanged for endpoints "
            "whose template already formats it (llama-server /v1/rerank)"
        ),
    )
    llm_model: str = Field(default="", description="LLM model id sent to the provider")
    pool_expansion: bool = Field(
        default=True,
        description="Add suitable owner symbols from files the fused ranking found to the rerank pool",
    )
    candidate_limit: int = Field(
        default=DEFAULT_CANDIDATE_LIMIT,
        ge=1,
        le=500,
        description="How many fused (FTS + dense, RRF) symbols enter the rerank pool",
    )
    index_dir: Path = Field(
        default=Path.home() / ".priorart" / "indexes",
        validation_alias="PRIORART_INDEX_DIR",
        description="Root of the per-worktree SQLite store tree",
    )

    @field_validator(
        "base_url",
        "api_key",
        "llm_base_url",
        "llm_api_key",
        "embed_base_url",
        "embed_api_key",
        "rerank_base_url",
        "rerank_api_key",
        mode="before",
    )
    @classmethod
    def _blank_means_absent(cls, value):
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("pool_expansion", mode="before")
    @classmethod
    def _blank_means_default(cls, value):
        return True if value == "" else value

    @field_validator("index_dir", mode="before")
    @classmethod
    def _expand_user(cls, value):
        return Path(value).expanduser() if isinstance(value, (str, Path)) else value

    @model_validator(mode="before")
    @classmethod
    def _apply_shared_provider(cls, data):
        if not isinstance(data, dict):
            return data
        for service in ("llm", "embed", "rerank"):
            if not data.get(f"{service}_base_url") and data.get("base_url"):
                data[f"{service}_base_url"] = data["base_url"]
            if not data.get(f"{service}_api_key") and data.get("api_key"):
                data[f"{service}_api_key"] = data["api_key"]
        return data
