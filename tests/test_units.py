from __future__ import annotations

import asyncio
import sqlite3
import struct
from pathlib import Path

import httpx
import pytest
import sqlite_vec
from typer.testing import CliRunner

from priorart import embed as embed_mod
from priorart import expand as expand_mod
from priorart import httputil, store
from priorart.cli import app
from priorart.config import Config
from priorart.runtime import Runtime
from priorart.store import connect
from tests.helpers import git, make_config

SAMPLE = 'def cli_target():\n    """Used by cli and runtime tests."""\n    pass\n'


def _init_repo(repo: Path) -> Path:
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


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


def test_make_embedder_requiresmake_config():
    assert embed_mod.make_embedder(make_config(Path("/nonexistent"), embed_base_url=None)) is None


def test_embed_serializes_vectors_in_input_order(monkeypatch):
    config = make_config(
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
    config = make_config(
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
    config = make_config(
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
    config = make_config(
        Path("/nonexistent"),
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    monkeypatch.setattr(embed_mod, "post_json", lambda *args, **kwargs: {"data": []})
    vectors, warning = embed(["a", "b"])
    assert vectors is None
    assert "wrong payload" in warning


def test_make_expander_requiresmake_config():
    assert expand_mod.make_expander(make_config(Path("/nonexistent"), llm_base_url=None)) is None


def test_expand_parses_chat_response(monkeypatch):
    config = make_config(Path("/nonexistent"), llm_base_url="http://llm.example/v1", llm_model="x")
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
    config = make_config(Path("/nonexistent"), llm_base_url="http://llm.example/v1", llm_model="x")
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
    runtime = Runtime(repo, config=make_config(tmp_path))
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
    git(repo, "add", "extra.py")
    git(repo, "commit", "-q", "-m", "extra")
    monkeypatch.setenv("PRIORART_DB", str(tmp_path / "cli.db"))
    runner = CliRunner()
    runner.invoke(app, ["index", str(repo)])

    (repo / "extra.py").unlink()
    git(repo, "rm", "-q", "extra.py")
    git(repo, "commit", "-q", "-m", "remove")
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

    conn = connect(db, embed_dim=4)

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
        connect(db, embed_dim=4)

    raw = sqlite3.connect(db)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 42
    assert raw.execute("SELECT COUNT(*) FROM app_data").fetchone()[0] == 1
    raw.close()


def test_non_database_file_is_a_loud_runtime_error(tmp_path):
    db = tmp_path / "garbage.db"
    db.write_bytes(b"this is not a sqlite database at all")

    with pytest.raises(RuntimeError, match="failed to initialize"):
        connect(db, embed_dim=4)


def test_newer_schema_version_is_reset(tmp_path):
    db = tmp_path / "future.db"
    conn = connect(db, embed_dim=4)
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    raw = sqlite3.connect(db)
    raw.execute("PRAGMA user_version = 99")
    raw.commit()
    raw.close()

    conn = connect(db, embed_dim=4)
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_app_version_change_resets_index(tmp_path, monkeypatch):
    db = tmp_path / "reindex.db"
    conn = connect(db, embed_dim=4)
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(store, "APP_VERSION", "9.9.9-test")
    conn = connect(db, embed_dim=4)

    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'app_version'").fetchone()[0]
        == "9.9.9-test"
    )


def test_embed_dim_change_resets_index(tmp_path):
    db = tmp_path / "dim.db"
    conn = connect(db, embed_dim=4)
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    conn = connect(db, embed_dim=8)

    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("SELECT value FROM meta WHERE key = 'embed_dim'").fetchone()[0] == "8"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_connect_reopen_preserves_index_and_is_noop(tmp_path):
    db = tmp_path / "reopen.db"
    conn = connect(db, embed_dim=4)
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    conn = connect(db, embed_dim=4)
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_failed_schema_creation_is_atomic(tmp_path, monkeypatch):
    from priorart import store as store_mod

    db = tmp_path / "partial.db"
    monkeypatch.setattr(
        store_mod,
        "SCHEMA_STATEMENTS",
        ("CREATE TABLE repos (repo TEXT PRIMARY KEY)", "CREATE TABLE bad ("),
    )
    with pytest.raises(RuntimeError, match="failed to initialize"):
        connect(db, embed_dim=4)

    raw = sqlite3.connect(db)
    assert raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == []
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
    raw.close()

    monkeypatch.undo()
    conn = connect(db, embed_dim=4)
    assert not conn.in_transaction
    assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION


def test_rollback_without_transaction_is_noop(tmp_path):
    from priorart import store as store_mod

    conn = connect(tmp_path / "noop.db", embed_dim=4)
    assert not conn.in_transaction
    store_mod._rollback(conn)


def test_rollback_discards_open_transaction(tmp_path):
    from priorart import store as store_mod

    db = tmp_path / "rollback.db"
    conn = connect(db, embed_dim=4)
    conn.execute("BEGIN")
    conn.execute("CREATE TABLE stray (x INTEGER)")
    store_mod._rollback(conn)
    assert not conn.in_transaction
    assert (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'stray'"
        ).fetchall()
        == []
    )


def test_index_line_flags_parse_problems_not_clean_empties():
    from priorart.search import SearchReport, _index_line

    report = SearchReport(
        candidates=[],
        warnings=[],
        symbol_count=5,
        head=None,
        age_seconds=None,
        parse_coverage={"ok": 10, "empty": 3, "partial": 1, "embed_failed": 2},
    )

    line = _index_line(report)

    assert "parse issues: 2 embed_failed, 1 partial" in line
    assert "empty" not in line


def _expand_db(tmp_path, files):
    conn = connect(tmp_path / "expand.db", embed_dim=4)
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
    from priorart.search import _expand_pool

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
    from priorart.search import _expand_pool

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
    from priorart.search import EXPANSION_FILE_QUOTA, EXPANSION_LIMIT, _expand_pool

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
    from priorart.search import _expand_pool

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
    conn, ids = _expand_db(
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

    from priorart.search import search

    with_expansion = search(conn, "r", "widget", k=10, pool_expansion=True)
    assert [candidate.qualname for candidate in with_expansion.candidates] == [
        "widget_keeper",
        "owner",
    ]
    assert with_expansion.trace.pool_expansion == [ids[("a.py", "owner")]]

    without_expansion = search(conn, "r", "widget", k=10, pool_expansion=False)
    assert [candidate.qualname for candidate in without_expansion.candidates] == ["widget_keeper"]
    assert without_expansion.trace.pool_expansion == []


def test_blank_pool_expansion_env_means_default():

    assert Config.model_validate({"pool_expansion": ""}).pool_expansion is True
