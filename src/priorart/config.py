from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HOME_ENV_FILE = Path.home() / ".priorart" / "priorart.env"


class Config(BaseSettings):
    """Runtime configuration.

    Values come from PRIORART_* environment variables, then a project-local
    .env, then ~/.priorart/priorart.env; real environment variables win over
    both files, and the project-local .env wins over the home file.
    """

    model_config = SettingsConfigDict(
        env_prefix="PRIORART_",
        env_file=(HOME_ENV_FILE, ".env"),
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
    rerank_model: str = Field(default="", description="Reranker model id sent to the provider")
    llm_model: str = Field(default="", description="LLM model id sent to the provider")
    pool_expansion: bool = Field(
        default=True,
        description="Add suitable owner symbols from files the fused ranking found to the rerank pool",
    )
    db_path: Path = Field(
        default=Path.home() / ".priorart" / "index.db",
        validation_alias="PRIORART_DB",
        description="SQLite index location",
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

    @field_validator("db_path", mode="before")
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
