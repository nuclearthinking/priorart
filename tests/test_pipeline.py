from pathlib import Path

import pytest
from pydantic import ValidationError

from priorart.core.config import Config
from priorart.indexing.parser import parse_source
from priorart.indexing.pipeline import index_repo
from priorart.retrieval import rrf, search
from priorart.storage import StoreProfile, initialize_writer
from tests.helpers import init_repo

SAMPLE = '''\
def parse_diff_patch(raw: bytes) -> list[str]:
    """Split a unified diff into per-file patches."""
    return raw.decode().split("---")


class DiffViewer:
    """Render diffs for the dashboard."""

    def render(self, patch: str) -> str:
        return patch
'''


def _connect(tmp_path: Path):
    return initialize_writer(tmp_path / "test.db", StoreProfile(embed_dim=8))


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


def test_full_signature_and_body_reach_rerank_documents(tmp_path):
    repo = init_repo(tmp_path)
    (repo / "sample.py").write_text(MULTILINE)
    conn = _connect(tmp_path)
    index_repo(conn, repo, embed_fn=None)
    captured = {}

    def rerank(query, documents):
        captured["documents"] = documents
        return None, None

    report = search(conn, str(repo), "connect host port", k=5, rerank_fn=rerank)

    top = report.candidates[0]
    assert top.qualname == "connect"
    assert top.full_signature == "def connect(\n    host: str,\n    port: int = 5432,\n):"
    assert top.body == (
        "def connect(\n    host: str,\n    port: int = 5432,\n):\n"
        '    """Connect."""\n    return host, port'
    )
    assert captured["documents"][0] == (
        "sample.py :: connect (function)\n"
        "def connect(\n    host: str,\n    port: int = 5432,\n):\n"
        "Connect.\n"
        "def connect(\n    host: str,\n    port: int = 5432,\n):\n"
        '    """Connect."""\n    return host, port'
    )


def test_candidate_limit_controls_fused_pool_size(tmp_path):
    repo = init_repo(tmp_path)
    files = {
        f"mod_{index}.py": f"def handler_{index}():\n    return {index}\n" for index in range(8)
    }
    for name, text in files.items():
        (repo / name).write_text(text)
    conn = _connect(tmp_path)
    index_repo(conn, repo, embed_fn=None)
    calls = []

    def rerank(query, documents):
        calls.append(len(documents))
        return None, None

    search(conn, str(repo), "handler", k=5, rerank_fn=rerank, candidate_limit=3)
    search(conn, str(repo), "handler", k=5, rerank_fn=rerank)

    assert calls == [3, 8]


def test_pool_is_independent_of_output_k(tmp_path):
    repo = init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    conn = _connect(tmp_path)
    index_repo(conn, repo, embed_fn=None)
    calls = []

    def rerank(query, documents):
        calls.append([document.splitlines()[0] for document in documents])
        return None, None

    first = search(conn, str(repo), "split unified diff into patches", k=2, rerank_fn=rerank)
    second = search(conn, str(repo), "split unified diff into patches", k=10, rerank_fn=rerank)

    assert calls[0] == calls[1]
    assert len(first.pool) == len(second.pool) == 3
    assert [candidate.qualname for candidate in first.candidates] == [
        candidate.qualname for candidate in first.pool[:2]
    ]
    assert second.candidates == second.pool


def test_index_and_lexical_search(tmp_path):
    repo = init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    conn = _connect(tmp_path)
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
    monkeypatch.setenv("PRIORART_INDEX_DIR", str(tmp_path / "indexes"))

    config = Config(_env_file=None)

    assert config.llm_base_url == "https://provider.example/v1"
    assert config.llm_api_key == "shared-key"
    assert config.embed_base_url == "http://127.0.0.1:8091/v1"
    assert config.embed_api_key == "local-key"
    assert config.rerank_base_url == "https://provider.example/v1"
    assert config.rerank_api_key == "shared-key"
    assert config.llm_model == "query-expander"
    assert config.index_dir == tmp_path / "indexes"


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


def test_config_expands_index_dir(monkeypatch):
    monkeypatch.setenv("PRIORART_INDEX_DIR", "~/.priorart/x/indexes")

    assert Config(_env_file=None).index_dir == Path.home() / ".priorart" / "x" / "indexes"
