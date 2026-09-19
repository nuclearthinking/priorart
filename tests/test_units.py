from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from priorart import embed as embed_mod
from priorart import expand as expand_mod
from priorart import httputil
from priorart.cli import app
from priorart.config import Config
from priorart.runtime import Runtime
from priorart.store import connect

SAMPLE = 'def cli_target():\n    """Used by cli and runtime tests."""\n    pass\n'


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        env={
            "PATH": subprocess.os.environ["PATH"],
            "HOME": str(Path.home()),
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _config(tmp_path: Path, **overrides) -> Config:
    fields = {
        "llm_base_url": None,
        "llm_api_key": None,
        "embed_base_url": None,
        "embed_api_key": None,
        "rerank_base_url": None,
        "rerank_api_key": None,
        "embed_model": "",
        "embed_dim": 4,
        "rerank_model": "",
        "llm_model": "",
        "db_path": tmp_path / "runtime.db",
    }
    fields.update(overrides)
    return Config(**fields)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_post_json_sends_bearer_and_parses_response(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(httputil.httpx, "post", fake_post)
    result = httputil.post_json("http://unit.example/v1", {"a": 1}, "secret", timeout=7)
    assert result == {"ok": True}
    assert seen["url"] == "http://unit.example/v1"
    assert seen["body"] == {"a": 1}
    assert seen["timeout"] == 7
    assert seen["headers"]["authorization"] == "Bearer secret"
    assert seen["headers"]["content-type"] == "application/json"


def test_post_json_without_key_has_no_auth_header():
    headers = httputil.auth_headers(None)
    assert "authorization" not in headers


def test_post_json_propagates_http_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httputil.httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPError):
        httputil.post_json("http://unit.example", {}, None, timeout=1)


def test_make_embedder_requires_config():
    assert embed_mod.make_embedder(_config(Path("/nonexistent"), embed_base_url=None)) is None


def test_embed_serializes_vectors_in_input_order(monkeypatch):
    config = _config(
        Path("/nonexistent"),
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    monkeypatch.setattr(
        embed_mod,
        "post_json",
        lambda *args, **kwargs: {"data": [{"embedding": [0.5, 0.5, 0.5, 0.5]} for _ in range(3)]},
    )
    vectors, warning = embed(["one", "two", "three"])
    assert warning is None
    assert len(vectors) == 3
    assert all(isinstance(vector, bytes) and len(vector) == 16 for vector in vectors)


def test_embed_batches_requests(monkeypatch):
    config = _config(
        Path("/nonexistent"),
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    monkeypatch.setattr(embed_mod, "BATCH", 2)
    calls = []

    def fake_post_json(url, body, api_key, timeout):
        calls.append(len(body["input"]))
        return {"data": [{"embedding": [0.5, 0.5, 0.5, 0.5]} for _ in body["input"]]}

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    vectors, warning = embed(["a", "b", "c", "d", "e"])
    assert warning is None
    assert len(vectors) == 5
    assert calls == [2, 2, 1]


def test_embed_http_failure_returns_warning(monkeypatch):
    config = _config(
        Path("/nonexistent"),
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)

    def fake_post_json(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    vectors, warning = embed(["a"])
    assert vectors is None
    assert "embedding request failed" in warning


def test_embed_wrong_payload_length_returns_warning(monkeypatch):
    config = _config(
        Path("/nonexistent"),
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    monkeypatch.setattr(embed_mod, "post_json", lambda *args, **kwargs: {"data": []})
    vectors, warning = embed(["a", "b"])
    assert vectors is None
    assert "wrong payload" in warning


def test_make_expander_requires_config():
    assert expand_mod.make_expander(_config(Path("/nonexistent"), llm_base_url=None)) is None


def test_expand_parses_chat_response(monkeypatch):
    config = _config(Path("/nonexistent"), llm_base_url="http://llm.example/v1", llm_model="x")
    expand = expand_mod.make_expander(config)
    monkeypatch.setattr(
        expand_mod,
        "post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": '["one", "two"]'}}]},
    )
    queries, warning = expand("feature request")
    assert queries == ["one", "two"]
    assert warning is None


def test_expand_failure_falls_back_to_raw_query(monkeypatch):
    config = _config(Path("/nonexistent"), llm_base_url="http://llm.example/v1", llm_model="x")
    expand = expand_mod.make_expander(config)

    def fake_post_json(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(expand_mod, "post_json", fake_post_json)
    queries, warning = expand("feature request")
    assert queries == ["feature request"]
    assert "expansion failed" in warning


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('["fenced"]', ["fenced"]),
        ('```json\n["a", " b ", ""]\n```', ["a", "b"]),
        ("not json at all", None),
        ('{"object": "not a list"}', None),
        ("[]", None),
        ("[1, true]", ["1", "True"]),
    ],
)
def test_parse_queries_variants(content, expected):
    assert expand_mod._parse_queries(content) == expected


def test_runtime_search_reindex_status_and_map(tmp_path):
    repo = _init_repo(tmp_path)
    runtime = Runtime(repo, config=_config(tmp_path))
    stats = runtime.reindex()
    assert stats["symbols"] == 1
    assert stats["files"] == 1
    assert runtime.symbol_count() == 1

    stats = runtime.reindex()
    assert stats["files"] == 0
    stats = runtime.reindex(rebuild=True)
    assert stats["files"] == 1

    report = runtime.search("cli_target", k=3)
    assert report.candidates[0].qualname == "cli_target"
    assert report.warnings == ["dense search skipped: embedding endpoint is not configured"]

    assert "repo:" in runtime.status()
    assert "cli_target" in runtime.map_symbols("*")
    assert "cli_target" in runtime.map_symbols("sample.py")


def test_cli_index_search_status(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "cli")
    monkeypatch.setenv("PRIORART_DB", str(tmp_path / "cli.db"))
    runner = CliRunner()

    result = runner.invoke(app, ["index", str(repo)])
    assert result.exit_code == 0
    assert "indexed 1 changed files, 1 symbols" in result.output

    result = runner.invoke(app, ["index", str(repo)])
    assert result.exit_code == 0
    assert "indexed 0 changed files" in result.output

    result = runner.invoke(app, ["search", "cli_target", "--repo", str(repo), "--k", "1"])
    assert result.exit_code == 0
    assert "cli_target" in result.output

    result = runner.invoke(app, ["status", "--repo", str(repo)])
    assert result.exit_code == 0
    assert "stale: no" in result.output


def test_cli_index_reports_removal_and_warnings(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "cli")
    (repo / "extra.py").write_text("def extra(): pass\n")
    _git(repo, "add", "extra.py")
    _git(repo, "commit", "-q", "-m", "extra")
    monkeypatch.setenv("PRIORART_DB", str(tmp_path / "cli.db"))
    runner = CliRunner()
    runner.invoke(app, ["index", str(repo)])

    (repo / "extra.py").unlink()
    _git(repo, "rm", "-q", "extra.py")
    _git(repo, "commit", "-q", "-m", "remove")
    result = runner.invoke(app, ["index", str(repo)])
    assert result.exit_code == 0
    assert "removed 1 deleted files" in result.output


def test_server_tools_outside_git_repo(tmp_path, monkeypatch):
    from priorart import server as server_mod

    monkeypatch.chdir(tmp_path)
    mcp = server_mod.build_server(None)
    payloads = {"search_codebase": {"query": "anything"}}
    for tool in ("search_codebase", "map_symbols", "refresh_index", "status"):
        content = asyncio.run(mcp.call_tool(tool, payloads.get(tool, {}))).content
        assert "no git repository detected" in content[0].text


def test_server_tools_serve_indexed_repo(tmp_path, monkeypatch):
    from priorart import server as server_mod

    repo = _init_repo(tmp_path / "srv")
    monkeypatch.setenv("PRIORART_DB", str(tmp_path / "srv.db"))
    mcp = server_mod.build_server(repo)

    content = asyncio.run(mcp.call_tool("status", {})).content
    assert "not indexed" in content[0].text

    content = asyncio.run(mcp.call_tool("refresh_index", {})).content
    assert "index total 1 symbols" in content[0].text

    content = asyncio.run(mcp.call_tool("search_codebase", {"query": "cli_target", "k": 1})).content
    assert "cli_target" in content[0].text

    content = asyncio.run(mcp.call_tool("map_symbols", {"path_glob": "*"})).content
    assert "cli_target" in content[0].text

    tools = asyncio.run(mcp.list_tools())
    assert sorted(tool.name for tool in tools) == [
        "map_symbols",
        "refresh_index",
        "search_codebase",
        "status",
    ]


def test_git_root_detects_repo_and_rejects_plain_dir(tmp_path):
    from priorart.server import _git_root

    repo = _init_repo(tmp_path / "root")
    assert _git_root(repo) == repo.resolve()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _git_root(plain) is None


def test_legacy_files_table_is_dropped_on_connect(tmp_path):
    db = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db)
    legacy.execute(
        "CREATE TABLE files (repo TEXT NOT NULL, path TEXT NOT NULL, mtime REAL NOT NULL, "
        "size INTEGER NOT NULL, PRIMARY KEY (repo, path))"
    )
    legacy.execute("INSERT INTO files VALUES ('r', 'p', 1.0, 1)")
    legacy.commit()
    legacy.close()

    conn = connect(db, embed_dim=4)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    assert "mtime_ns" in columns
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
