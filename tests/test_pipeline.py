from pathlib import Path

import pytest
from pydantic import ValidationError

from priorart.config import Config
from priorart.indexer import index_repo, parse_source
from priorart.search import rrf, search
from priorart.store import connect

SAMPLE = '''\
def parse_diff_patch(raw: bytes) -> list[str]:
    """Split a unified diff into per-file patches."""
    return raw.decode().split("---")


class DiffViewer:
    """Render diffs for the dashboard."""

    def render(self, patch: str) -> str:
        return patch
'''


def test_parse_source_extracts_symbols():
    result = parse_source(SAMPLE.encode(), "python", "sample.py")
    assert result.status == "ok"
    assert [symbol.qualname for symbol in result.symbols] == [
        "parse_diff_patch",
        "DiffViewer",
        "DiffViewer.render",
    ]
    top = result.symbols[0]
    assert top.kind == "function"
    assert top.line == 1
    assert "Split a unified diff" in top.docstring


MULTILINE = '''\
def connect(
    host: str,
    port: int = 5432,
):
    """Connect."""
    return host, port
'''


def test_full_signature_spans_multiline_definition():
    result = parse_source(MULTILINE.encode(), "python", "sample.py")
    symbol = result.symbols[0]
    assert symbol.signature == "def connect("
    assert symbol.full_signature == "def connect(\n    host: str,\n    port: int = 5432,\n):"


def test_full_signature_falls_back_for_bodyless_definitions():
    result = parse_source(b"type Server struct {\n\tName string\n}\n", "go", "server.go")
    symbol = result.symbols[0]
    assert symbol.kind == "struct"
    assert symbol.full_signature == symbol.signature == "Server struct {"


def test_full_signature_reaches_rerank_documents(tmp_path):
    (tmp_path / "sample.py").write_text(MULTILINE)
    repo = tmp_path.resolve()
    conn = connect(tmp_path / "test.db", embed_dim=8)
    index_repo(conn, repo, embed_fn=None)
    captured = {}

    def rerank(query, documents):
        captured["documents"] = documents
        return None, None

    report = search(conn, str(repo), "connect host port", k=5, rerank_fn=rerank)

    top = report.candidates[0]
    assert top.qualname == "connect"
    assert top.full_signature == "def connect(\n    host: str,\n    port: int = 5432,\n):"
    assert captured["documents"][0] == (
        "sample.py :: connect (function)\n"
        "def connect(\n    host: str,\n    port: int = 5432,\n):\n"
        "Connect."
    )


def test_index_and_lexical_search(tmp_path):
    (tmp_path / "sample.py").write_text(SAMPLE)
    repo = tmp_path.resolve()
    conn = connect(tmp_path / "test.db", embed_dim=8)
    stats = index_repo(conn, repo, embed_fn=None)
    assert stats["symbols"] == 3
    report = search(conn, str(repo), "split unified diff into patches", k=5)
    assert report.candidates[0].qualname == "parse_diff_patch"
    assert any("dense search skipped" in warning for warning in report.warnings)


def test_rrf_favors_overlap():
    scores = rrf([[1, 5], [1, 9]])
    assert scores[1] > scores[5]
    assert scores[5] == scores[9]


def test_parse_source_reports_empty_and_unsupported():
    empty = parse_source(b"VALUE = 1\n", "python", "consts.py")
    assert empty.status == "empty"
    assert empty.symbols == []

    unsupported = parse_source(b"x = 1\n", "no-such-lang", "sample.noext")
    assert unsupported.status == "unsupported"
    assert unsupported.symbols == []


def test_config_supports_shared_provider_with_service_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIORART_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("PRIORART_API_KEY", "shared-key")
    monkeypatch.setenv("PRIORART_EMBED_BASE_URL", "http://127.0.0.1:8091/v1")
    monkeypatch.setenv("PRIORART_EMBED_API_KEY", "local-key")
    monkeypatch.setenv("PRIORART_LLM_MODEL", "query-expander")
    monkeypatch.setenv("PRIORART_DB", str(tmp_path / "index.db"))

    config = Config(_env_file=None)

    assert config.llm_base_url == "https://provider.example/v1"
    assert config.llm_api_key == "shared-key"
    assert config.embed_base_url == "http://127.0.0.1:8091/v1"
    assert config.embed_api_key == "local-key"
    assert config.rerank_base_url == "https://provider.example/v1"
    assert config.rerank_api_key == "shared-key"
    assert config.llm_model == "query-expander"
    assert config.db_path == tmp_path / "index.db"


def test_config_reads_env_file(tmp_path):
    env_file = tmp_path / "priorart.env"
    env_file.write_text(
        "PRIORART_BASE_URL=https://file.example/v1\n"
        "PRIORART_RERANK_MODEL=file-reranker\n"
        "PRIORART_EMBED_DIM=2048\n"
    )

    config = Config(_env_file=env_file)

    assert config.base_url == "https://file.example/v1"
    assert config.rerank_base_url == "https://file.example/v1"
    assert config.rerank_model == "file-reranker"
    assert config.embed_dim == 2048


def test_config_env_vars_beat_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "priorart.env"
    env_file.write_text("PRIORART_LLM_MODEL=from-file\n")
    monkeypatch.setenv("PRIORART_LLM_MODEL", "from-env")

    config = Config(_env_file=env_file)

    assert config.llm_model == "from-env"


def test_config_blank_values_mean_absent(monkeypatch):
    monkeypatch.setenv("PRIORART_BASE_URL", "   ")

    config = Config(_env_file=None)

    assert config.base_url is None
    assert config.llm_base_url is None


def test_config_rejects_nonpositive_embed_dim(monkeypatch):
    monkeypatch.setenv("PRIORART_EMBED_DIM", "0")

    with pytest.raises(ValidationError):
        Config(_env_file=None)


def test_config_expands_db_path(monkeypatch):
    monkeypatch.setenv("PRIORART_DB", "~/.priorart/x/index.db")

    assert Config(_env_file=None).db_path == Path.home() / ".priorart" / "x" / "index.db"
