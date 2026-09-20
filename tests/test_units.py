from __future__ import annotations

import asyncio
import sqlite3
import struct
import time
from pathlib import Path

import httpx
import pytest
import sqlite_vec
from typer.testing import CliRunner

from priorart.cli import app
from priorart.core.config import Config
from priorart.core.jobs import FINAL_STATES
from priorart.models import embed as embed_mod
from priorart.models import expand as expand_mod
from priorart.models import httputil
from priorart.registry import RuntimeRegistry
from priorart.storage import StoreProfile, initialize_writer, store
from tests.helpers import git, make_config

SAMPLE = 'def cli_target():\n    """Used by cli and runtime tests."""\n    pass\n'


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def _wait_job(registry: RuntimeRegistry, job_id: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _handle, job = registry.get_job(job_id)
        if job.state in FINAL_STATES:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


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


def test_post_json_without_key_has_no_auth_header(monkeypatch):
    monkeypatch.setattr(
        httputil.httpx,
        "post",
        lambda url, json=None, headers=None, timeout=None: _FakeResponse({"ok": True}),
    )
    result = httputil.post_json("http://unit.example/v1", {}, None, timeout=1)
    assert result == {"ok": True}
    assert httputil.auth_headers(None) == {"content-type": "application/json"}


def test_post_json_propagates_http_error(monkeypatch):
    def failing_post(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httputil.httpx, "post", failing_post)
    with pytest.raises(httpx.ConnectError):
        httputil.post_json("http://unit.example/v1", {}, None, timeout=1)


def test_make_embedder_requires_config(tmp_path):
    assert make_config(tmp_path) is not None
    assert embed_mod.make_embedder(make_config(tmp_path)) is None


def test_embed_serializes_vectors_in_input_order(monkeypatch, tmp_path):
    def fake_post_json(url, body, api_key, timeout):
        assert [item.rsplit("Text: ", 1)[1] for item in body["input"]] == ["text-a", "text-b"]
        return {
            "data": [
                {"index": 1, "embedding": [0.2, 0.2, 0.2, 0.2]},
                {"index": 0, "embedding": [0.1, 0.1, 0.1, 0.1]},
            ]
        }

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed(["text-a", "text-b"])
    assert warning is None
    assert vectors == [
        sqlite_vec.serialize_float32([0.1, 0.1, 0.1, 0.1]),
        sqlite_vec.serialize_float32([0.2, 0.2, 0.2, 0.2]),
    ]


def test_embed_batches_requests(monkeypatch, tmp_path):
    calls = []

    def fake_post_json(url, body, api_key, timeout):
        calls.append(body["input"])
        return {
            "data": [
                {"index": i, "embedding": [0.1, 0.1, 0.1, 0.1]} for i in range(len(body["input"]))
            ]
        }

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed([f"t{i}" for i in range(70)])
    assert warning is None
    assert len(vectors) == 70
    assert [len(batch) for batch in calls] == [32, 32, 6]


def test_embed_http_failure_returns_warning(monkeypatch, tmp_path):
    def failing_post(url, body, api_key, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(embed_mod, "post_json", failing_post)
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed(["text"])
    assert vectors is None
    assert "embedding request failed" in warning


def test_embed_wrong_payload_length_returns_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        embed_mod,
        "post_json",
        lambda *args, **kwargs: {"data": []},
    )
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed(["text"])
    assert vectors is None
    assert "wrong payload" in warning


def test_embed_qwen3_format_instructs_queries_and_normalizes(monkeypatch, tmp_path):
    seen = {}

    def fake_post_json(url, body, api_key, timeout):
        seen["input"] = body["input"]
        return {"data": [{"index": 0, "embedding": [3.0, 0.0, 0.0, 0.0]}]}

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
        embed_input_format="qwen3",
    )
    embed = embed_mod.make_embedder(config)
    vectors, _warning = embed(["find the handler"], query=True)
    assert seen["input"] == [
        embed_mod._qwen3_query(embed_mod.QUERY_INSTRUCTION, "find the handler")
    ]
    assert vectors == [sqlite_vec.serialize_float32([1.0, 0.0, 0.0, 0.0])]

    vectors, _warning = embed(["def handler(): pass"])
    assert seen["input"] == ["def handler(): pass"]
    assert vectors == [sqlite_vec.serialize_float32([1.0, 0.0, 0.0, 0.0])]


def test_embed_legacy_format_wraps_documents(monkeypatch, tmp_path):
    seen = {}

    def fake_post_json(url, body, api_key, timeout):
        seen["input"] = body["input"]
        return {"data": [{"index": 0, "embedding": [0.1] * 4}]}

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    embed(["def handler(): pass"])
    assert seen["input"] == [
        embed_mod._instruct(embed_mod.DOCUMENT_INSTRUCTION, "def handler(): pass")
    ]


def test_embed_dimension_mismatch_returns_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(
        embed_mod,
        "post_json",
        lambda *args, **kwargs: {"data": [{"index": 0, "embedding": [0.1] * 8}]},
    )
    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed(["text"])
    assert vectors is None
    assert "dimension 8, expected 4" in warning


def test_make_expander_requires_config(tmp_path):
    assert expand_mod.make_expander(make_config(tmp_path)) is None


def test_expand_parses_chat_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        expand_mod,
        "post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": '["a", "b"]'}}]},
    )
    config = make_config(
        tmp_path,
        llm_base_url="http://llm.example/v1",
        llm_model="chat",
    )
    expand = expand_mod.make_expander(config)
    queries, warning = expand("find the handler")
    assert warning is None
    assert queries == ["a", "b"]


def test_expand_failure_falls_back_to_raw_query(monkeypatch, tmp_path):
    def failing_post(url, body, api_key, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(expand_mod, "post_json", failing_post)
    config = make_config(
        tmp_path,
        llm_base_url="http://llm.example/v1",
        llm_model="chat",
    )
    expand = expand_mod.make_expander(config)
    queries, warning = expand("find the handler")
    assert queries == ["find the handler"]
    assert "query expansion failed" in warning


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('```json\n["a", "b"]\n```', ["a", "b"]),
        ('["", "  "]', None),
        ("not json", None),
        ('{"a": 1}', None),
        ("[]", None),
    ],
)
def test_parse_queries_variants(content, expected):
    assert expand_mod._parse_queries(content) == expected


def test_handle_search_refresh_status_and_map(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)
    job = _wait_job(registry, registry.submit_refresh(handle).job_id)
    assert job.state == "completed"
    assert job.counters["files"] == 1
    assert job.counters["symbols"] == 1
    assert job.lexical_ready

    job = _wait_job(registry, registry.submit_refresh(handle).job_id)
    assert job.counters["files"] == 0
    job = _wait_job(registry, registry.submit_refresh(handle, rebuild=True).job_id)
    assert job.counters["files"] == 1

    report = handle.search("cli_target", k=3)
    assert report.candidates[0].qualname == "cli_target"
    # an exact identifier match answers without any model or warning
    assert report.stages_used == ["exact"]

    report = handle.search("the cli target function", k=3)
    assert report.candidates[0].qualname == "cli_target"
    assert report.warnings == ["dense search skipped: embedding endpoint is not configured"]

    assert "repo:" in handle.status_text()
    rows, _cursor = handle.map_symbols("*")
    assert [row["qualname"] for row in rows] == ["cli_target"]
    rows, _cursor = handle.map_symbols("sample.py")
    assert [row["qualname"] for row in rows] == ["cli_target"]


def test_cli_index_search_status(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path / "cli")
    monkeypatch.setenv("PRIORART_INDEX_DIR", str(tmp_path / "cli-indexes"))
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
    git(repo, "add", "extra.py")
    git(repo, "commit", "-q", "-m", "extra")
    monkeypatch.setenv("PRIORART_INDEX_DIR", str(tmp_path / "cli-indexes"))
    runner = CliRunner()
    runner.invoke(app, ["index", str(repo)])

    (repo / "extra.py").unlink()
    git(repo, "rm", "-q", "extra.py")
    git(repo, "commit", "-q", "-m", "remove")
    result = runner.invoke(app, ["index", str(repo)])
    assert result.exit_code == 0
    assert "removed 1 deleted files" in result.output


def test_cli_index_emits_json_progress(tmp_path, monkeypatch):
    import json as json_mod

    repo = _init_repo(tmp_path / "cli")
    monkeypatch.setenv("PRIORART_INDEX_DIR", str(tmp_path / "cli-indexes"))
    runner = CliRunner()

    result = runner.invoke(app, ["index", str(repo), "--json-progress"])
    assert result.exit_code == 0
    events = [json_mod.loads(line) for line in result.output.splitlines() if line.startswith("{")]
    assert events[0]["state"] in ("queued", "running")
    assert events[-1]["state"] == "completed"
    assert events[-1]["counters"]["symbols"] == 1


def test_server_tools_outside_git_repo(tmp_path, monkeypatch):
    from priorart import server as server_mod

    monkeypatch.chdir(tmp_path)
    mcp = server_mod.build_server(None)
    result = asyncio.run(mcp.call_tool("search_codebase", {"query": "anything"}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "REPOSITORY_NOT_SELECTED"
    assert "repo=<absolute path>" in result.structured_content["error"]["next_action"]


def test_server_tools_serve_indexed_repo(tmp_path, monkeypatch):
    from priorart import server as server_mod

    repo = _init_repo(tmp_path / "srv")
    monkeypatch.setenv("PRIORART_INDEX_DIR", str(tmp_path / "srv-indexes"))
    mcp = server_mod.build_server(repo, config=make_config(tmp_path))

    result = asyncio.run(mcp.call_tool("status", {}))
    assert result.is_error is False
    assert result.structured_content["index"]["state"] == "absent"

    result = asyncio.run(mcp.call_tool("search_codebase", {"query": "cli_target", "k": 1}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "INDEX_NOT_READY"

    result = asyncio.run(mcp.call_tool("refresh_index", {}))
    assert result.is_error is False
    job_id = result.structured_content["data"]["job"]["job_id"]
    while True:
        result = asyncio.run(mcp.call_tool("get_index_job", {"job_id": job_id}))
        if result.structured_content["data"]["job"]["state"] in FINAL_STATES:
            break
    assert result.structured_content["data"]["job"]["state"] == "completed"

    result = asyncio.run(mcp.call_tool("search_codebase", {"query": "cli_target", "k": 1}))
    assert result.is_error is False
    assert "cli_target" in result.content[0].text
    assert result.structured_content["ok"] is True
    assert result.structured_content["repo"] == str(repo)
    assert result.structured_content["index"]["index_epoch"] >= 1

    result = asyncio.run(mcp.call_tool("map_symbols", {"path_glob": "*"}))
    assert result.is_error is False
    assert result.structured_content["data"]["symbols"][0]["qualname"] == "cli_target"

    result = asyncio.run(mcp.call_tool("list_workspaces", {}))
    assert result.is_error is False
    assert str(repo) in result.structured_content["data"]["workspaces"]["configured"]

    tools = asyncio.run(mcp.list_tools())
    assert sorted(tool.name for tool in tools) == [
        "cancel_index_job",
        "get_index_job",
        "list_workspaces",
        "map_symbols",
        "refresh_index",
        "search_codebase",
        "status",
    ]


def test_server_rejects_relative_and_wrong_repo(tmp_path):
    from priorart import server as server_mod

    repo = _init_repo(tmp_path / "srv")
    mcp = server_mod.build_server(repo, config=make_config(tmp_path))

    result = asyncio.run(mcp.call_tool("search_codebase", {"query": "x", "repo": "relative/path"}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "REPOSITORY_NOT_FOUND"

    result = asyncio.run(mcp.call_tool("search_codebase", {"query": "x", "repo": str(tmp_path)}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "REPOSITORY_NOT_FOUND"

    result = asyncio.run(mcp.call_tool("get_index_job", {"job_id": "missing"}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "JOB_NOT_FOUND"


def test_git_toplevel_detects_repo_and_rejects_plain_dir(tmp_path):
    from priorart.core.git import git_toplevel

    repo = _init_repo(tmp_path / "root")
    assert git_toplevel(repo) == repo.resolve()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert git_toplevel(plain) is None


def _connect(tmp_path: Path, dim: int = 4, **profile) -> sqlite3.Connection:
    return initialize_writer(tmp_path / "unit.db", StoreProfile(embed_dim=dim, **profile))


def test_legacy_database_is_reset_from_scratch(tmp_path):
    db = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db)
    legacy.execute(
        "CREATE TABLE files (repo TEXT NOT NULL, path TEXT NOT NULL, mtime REAL NOT NULL, "
        "size INTEGER NOT NULL, PRIMARY KEY (repo, path))"
    )
    legacy.execute("INSERT INTO files VALUES ('r', 'p', 1.0, 1)")
    legacy.enable_load_extension(True)  # noqa: FBT003 - sqlite3 positional-only API
    sqlite_vec.load(legacy)
    legacy.enable_load_extension(False)  # noqa: FBT003 - sqlite3 positional-only API
    legacy.execute(
        "CREATE VIRTUAL TABLE symbols_vec USING vec0("
        "symbol_id INTEGER PRIMARY KEY, embedding float[4])"
    )
    legacy.commit()
    legacy.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=4))

    assert not conn.in_transaction
    columns = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    assert "mtime_ns" in columns
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'app_version'").fetchone()[0]
        == store.APP_VERSION
    )
    assert (
        conn.execute("SELECT sql FROM sqlite_master WHERE name = 'symbols_vec'")
        .fetchone()[0]
        .find("partition key")
        != -1
    )


def test_foreign_database_user_version_is_not_overwritten(tmp_path):
    db = tmp_path / "foreign.db"
    foreign = sqlite3.connect(db)
    foreign.execute("CREATE TABLE app_data (payload TEXT)")
    foreign.execute("INSERT INTO app_data VALUES ('owned by another application')")
    foreign.execute("PRAGMA user_version = 42")
    foreign.commit()
    foreign.close()

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        initialize_writer(db, StoreProfile(embed_dim=4))

    raw = sqlite3.connect(db)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 42
    assert raw.execute("SELECT COUNT(*) FROM app_data").fetchone()[0] == 1
    raw.close()


def test_non_database_file_is_a_loud_runtime_error(tmp_path):
    db = tmp_path / "garbage.db"
    db.write_bytes(b"this is not a sqlite database at all")

    with pytest.raises(RuntimeError, match="failed to initialize"):
        initialize_writer(db, StoreProfile(embed_dim=4))


def test_newer_schema_version_is_reset(tmp_path):
    db = tmp_path / "future.db"
    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    raw = sqlite3.connect(db)
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_app_version_change_resets_index(tmp_path, monkeypatch):
    db = tmp_path / "reindex.db"
    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(store, "APP_VERSION", "9.9.9-test")
    conn = initialize_writer(db, StoreProfile(embed_dim=4))

    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'app_version'").fetchone()[0]
        == "9.9.9-test"
    )


def test_embed_dim_change_resets_index(tmp_path):
    db = tmp_path / "dim.db"
    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=8))

    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()[0] == "8"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_schema_creates_repo_path_index(tmp_path):
    from priorart.indexing.pipeline import index_repo

    repo = _init_repo(tmp_path / "repo")
    conn = _connect(tmp_path)
    index_repo(conn, repo)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'symbols_repo_path'"
    ).fetchone()
    assert row is not None
    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT id FROM symbols WHERE repo = ? AND path = ?", ("r", "p")
    ).fetchall()
    assert all("SCAN symbols" not in step[3] for step in plan)


def test_connect_reopen_preserves_index_and_is_noop(tmp_path):
    db = tmp_path / "reopen.db"
    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_failed_schema_creation_is_atomic(tmp_path, monkeypatch):
    db = tmp_path / "partial.db"
    monkeypatch.setattr(
        store,
        "SCHEMA_STATEMENTS",
        ("CREATE TABLE repos (repo TEXT PRIMARY KEY)", "CREATE TABLE bad ("),
    )
    with pytest.raises(RuntimeError, match="failed to initialize"):
        initialize_writer(db, StoreProfile(embed_dim=4))

    raw = sqlite3.connect(db)
    assert raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == []
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.close()

    monkeypatch.undo()
    conn = initialize_writer(db, StoreProfile(embed_dim=4))
    assert not conn.in_transaction
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_rollback_without_transaction_is_noop(tmp_path):
    conn = _connect(tmp_path)
    assert not conn.in_transaction
    store._rollback(conn)


def test_rollback_discards_open_transaction(tmp_path):
    conn = _connect(tmp_path)
    conn.execute("BEGIN")
    conn.execute("CREATE TABLE stray (x INTEGER)")
    store._rollback(conn)
    assert not conn.in_transaction
    assert (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'stray'"
        ).fetchall()
        == []
    )


def test_index_line_flags_parse_problems_not_clean_empties():
    from priorart.retrieval.search import SearchReport, _index_line

    report = SearchReport(
        candidates=[],
        warnings=[],
        symbol_count=5,
        head=None,
        age_seconds=None,
        parse_coverage={"ok": 10, "empty": 3, "partial": 1, "error": 2},
    )

    line = _index_line(report)

    assert "parse issues: 2 error, 1 partial" in line
    assert "empty" not in line


def _expand_db(tmp_path, files):
    conn = _connect(tmp_path)
    conn.execute("INSERT INTO repos (repo, head, indexed_at) VALUES ('r', NULL, NULL)")
    ids = {}
    for path, symbols in files.items():
        for order, (qualname, body, vector) in enumerate(symbols, 1):
            cur = conn.execute(
                "INSERT INTO symbols (repo, path, name, qualname, kind, lang, line, end_line, "
                "signature, full_signature, docstring, body, search_text, embed_text) "
                "VALUES ('r', ?, ?, ?, 'function', 'python', ?, ?, '', ?, '', ?, ?, ?)",
                (path, qualname, qualname, order, order + 1, qualname, body, qualname, qualname),
            )
            ids[(path, qualname)] = cur.lastrowid
            if vector is not None:
                conn.execute(
                    "INSERT INTO symbols_vec (symbol_id, repo, embedding) VALUES (?, 'r', ?)",
                    (cur.lastrowid, vector),
                )
    conn.commit()
    return conn, ids


def _vec4(*values):
    return struct.pack("4f", *values)


def test_expand_pool_ranks_term_hits_before_cosine(tmp_path):
    from priorart.retrieval.search import _expand_pool

    query_vector = _vec4(1.0, 0.0, 0.0, 0.0)
    conn, ids = _expand_db(
        tmp_path,
        {
            "a.py": [
                ("pool_hit", "def pool_hit(): pass", None),
                ("term_owner", "INSERT INTO events ON CONFLICT", _vec4(0.0, 1.0, 0.0, 0.0)),
                ("cosy_neighbour", "def cosy_neighbour(): pass", _vec4(1.0, 0.0, 0.0, 0.0)),
            ]
        },
    )
    pool = [(ids[("a.py", "pool_hit")], 0.9)]

    expanded = _expand_pool(conn, "r", "сделать INSERT ON CONFLICT", query_vector, pool)

    assert next(row[3] for _sid, row in expanded) == "term_owner"


def test_expand_pool_breaks_ties_by_cosine(tmp_path):
    from priorart.retrieval.search import _expand_pool

    query_vector = _vec4(1.0, 0.0, 0.0, 0.0)
    conn, ids = _expand_db(
        tmp_path,
        {
            "a.py": [
                ("pool_hit", "def pool_hit(): pass", None),
                ("aligned", "INSERT INTO events", _vec4(0.9, 0.1, 0.0, 0.0)),
                ("orthogonal", "INSERT INTO logs", _vec4(0.0, 0.9, 0.1, 0.0)),
            ]
        },
    )
    pool = [(ids[("a.py", "pool_hit")], 0.9)]

    expanded = _expand_pool(conn, "r", "INSERT", query_vector, pool)

    assert [row[3] for _sid, row in expanded] == ["aligned", "orthogonal"]


def test_expand_pool_respects_file_quota_order_and_limit(tmp_path):
    from priorart.retrieval.search import EXPANSION_FILE_QUOTA, EXPANSION_LIMIT, _expand_pool

    files = {}
    for index in range(20):
        files[f"f{index:02}.py"] = [("sibling", f"def s{index}(): pass", None) for _ in range(5)]
    conn, ids = _expand_db(tmp_path, files)
    top = [(ids[(f"f{index:02}.py", "sibling")], 0.9 - index * 0.01) for index in range(20)]

    expanded = _expand_pool(conn, "r", "anything", None, top)

    per_file = {}
    for _sid, row in expanded:
        per_file[row[1]] = per_file.get(row[1], 0) + 1
    assert len(expanded) == EXPANSION_LIMIT
    assert all(count == EXPANSION_FILE_QUOTA for count in list(per_file.values())[:-1])
    assert list(per_file.values())[-1] == 2
    assert set(per_file) == {f"f{index:02}.py" for index in range(17)}


def test_expand_pool_ignores_files_without_fused_symbols(tmp_path):
    from priorart.retrieval.search import _expand_pool

    conn, ids = _expand_db(
        tmp_path,
        {
            "found.py": [("pool_hit", "def pool_hit(): pass", None)],
            "unfound.py": [("loner", "INSERT INTO events", None)],
        },
    )
    pool = [(ids[("found.py", "pool_hit")], 0.9)]

    expanded = _expand_pool(conn, "r", "INSERT", None, pool)

    assert expanded == []


def test_search_pool_expansion_adds_owner_and_can_be_disabled(tmp_path):
    conn, _ids = _expand_db(
        tmp_path,
        {
            "a.py": [
                ("widget_keeper", "def keeper(): pass", None),
                ("owner", "widget_factory = register(widget)", None),
            ]
        },
    )
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'a.py', 0, 0)")
    conn.commit()

    from priorart.retrieval import search

    with_expansion = search(conn, "r", "widget", k=10, pool_expansion=True)
    assert [candidate.qualname for candidate in with_expansion.candidates] == [
        "widget_keeper",
        "owner",
    ]

    without_expansion = search(conn, "r", "widget", k=10, pool_expansion=False)
    assert [candidate.qualname for candidate in without_expansion.candidates] == ["widget_keeper"]
    assert without_expansion.trace.pool_expansion == []


def test_blank_pool_expansion_env_means_default():

    assert Config.model_validate({"pool_expansion": ""}).pool_expansion is True


def _insert_symbol(conn, repo: str = "r") -> None:
    conn.execute(
        "INSERT INTO symbols (repo, path, name, qualname, kind, lang, line, end_line, "
        "signature, full_signature, docstring, body, search_text, embed_text) "
        "VALUES (?, 'a.py', 'n', 'q', 'function', 'python', 1, 1, '', '', '', '', '', '')",
        (repo,),
    )
    conn.commit()


def test_embed_profile_change_resets_index(tmp_path):
    db = tmp_path / "profile.db"
    conn = initialize_writer(db, StoreProfile(embed_dim=4, embed_model="embed-a"))
    _insert_symbol(conn)
    conn.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=4, embed_model="embed-a"))
    assert conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 1
    conn.close()

    conn = initialize_writer(db, StoreProfile(embed_dim=4, embed_model="embed-b"))
    assert conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'embed_model'").fetchone()[0] == "embed-b"
    )
    _insert_symbol(conn)
    conn.close()

    conn = initialize_writer(
        db, StoreProfile(embed_dim=4, embed_model="embed-b", embed_input_format="qwen3")
    )
    assert conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'embed_input_format'").fetchone()[0]
        == "qwen3"
    )
    conn.close()


def test_foreign_database_zero_user_version_is_refused(tmp_path):
    db = tmp_path / "foreign-zero.db"
    foreign = sqlite3.connect(db)
    foreign.execute("CREATE TABLE app_data (payload TEXT)")
    foreign.execute("INSERT INTO app_data VALUES ('owned by another application')")
    foreign.commit()
    foreign.close()

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        initialize_writer(db, StoreProfile(embed_dim=4))

    raw = sqlite3.connect(db)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    assert raw.execute("SELECT COUNT(*) FROM app_data").fetchone()[0] == 1
    raw.close()
