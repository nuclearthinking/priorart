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

    config = Config.from_env()

    assert config.llm_base_url == "https://provider.example/v1"
    assert config.llm_api_key == "shared-key"
    assert config.embed_base_url == "http://127.0.0.1:8091/v1"
    assert config.embed_api_key == "local-key"
    assert config.rerank_base_url == "https://provider.example/v1"
    assert config.rerank_api_key == "shared-key"
    assert config.llm_model == "query-expander"
    assert config.db_path == tmp_path / "index.db"
