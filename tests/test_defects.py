from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from priorart.indexing.inventory import repo_languages
from priorart.indexing.parser import ParseResult, parse_source
from priorart.indexing.pipeline import index_repo
from priorart.models.embed import make_embedder  # noqa: F401 - imported for parity by units
from priorart.retrieval import format_report, search, status_text, valid_rerank_order
from priorart.retrieval.search import _fetch, fts_search
from priorart.storage import StoreProfile, initialize_writer
from tests.helpers import git, init_repo

SAMPLE = '''\
def parse_diff_patch(raw: bytes) -> list[str]:
    """Split a unified diff into per-file patches."""
    return raw.decode().split("---")


class DiffViewer:
    """Render diffs for the dashboard."""

    def render(self, patch: str) -> str:
        return patch
'''


def _connect(tmp_path: Path, **profile):
    return initialize_writer(
        tmp_path / "test.db", StoreProfile(embed_dim=profile.pop("embed_dim", 4), **profile)
    )


def _index(repo: Path, tmp_path: Path, embed_fn=None, *, rebuild=False):
    conn = _connect(tmp_path)
    return conn, index_repo(conn, repo, embed_fn=embed_fn, rebuild=rebuild)


def _file_rows(conn, repo: Path) -> set[str]:
    return {row[0] for row in conn.execute("SELECT path FROM files WHERE repo = ?", (str(repo),))}


def test_inventory_includes_untracked_and_excludes_ignored(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "tracked.py").write_text("def tracked(): pass\n")
    (repo / ".gitignore").write_text("ignored.py\n")
    git(repo, "add", "tracked.py", ".gitignore")
    git(repo, "commit", "-q", "-m", "init")
    (repo / "ignored.py").write_text("def ignored(): pass\n")
    (repo / "untracked.py").write_text("def untracked(): pass\n")

    worktree = tmp_path / "worktree"
    git(repo, "worktree", "add", "-q", str(worktree))
    (worktree / "untracked.py").write_text("def untracked(): pass\n")

    conn, _stats = _index(worktree, tmp_path)
    assert _file_rows(conn, worktree) == {"tracked.py", "untracked.py"}
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(worktree),))
    }
    assert names == {"tracked", "untracked"}


def test_symlink_outside_repo_is_skipped(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "tracked.py").write_text("def tracked(): pass\n")
    outside = tmp_path / "outside.py"
    outside.write_text("def secret(): '''s3cr3t doc'''\n    pass\n")
    (repo / "linked.py").symlink_to(outside.resolve())
    git(repo, "add", "tracked.py", "linked.py")
    git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert _file_rows(conn, repo) == {"tracked.py"}
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"tracked"}


def test_rebuild_removes_vanished_files(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "kept.py").write_text("def kept(): pass\n")
    (repo / "gone.py").write_text("def gone(): pass\n")
    git(repo, "add", "kept.py", "gone.py")
    git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert _file_rows(conn, repo) == {"kept.py", "gone.py"}

    (repo / "gone.py").unlink()
    git(repo, "rm", "-q", "gone.py")
    git(repo, "commit", "-q", "-m", "remove")

    conn, stats = _index(repo, tmp_path, rebuild=True)
    assert stats["removed"] == 1
    assert _file_rows(conn, repo) == {"kept.py"}


def test_embedding_failure_publishes_lexical_symbols(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def sample(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")

    conn, stats = _index(
        repo, tmp_path, embed_fn=lambda texts, *, query=False: (None, "embedding down")
    )

    assert stats["warnings"] == ["embedding down"]
    assert stats["symbols"] == 1
    assert stats["embed_failures"] == 1
    status, _detail = conn.execute(
        "SELECT status, detail FROM parse_state WHERE repo = ? AND path = 'sample.py'",
        (str(repo),),
    ).fetchone()
    assert status == "ok"
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(repo),)
    ).fetchone()[0]
    assert vectorized == 0
    # lexical search still finds the published symbol
    report = search(conn, str(repo), "sample", k=3)
    assert report.candidates[0].qualname == "sample"


def test_embedding_failure_is_retried(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")

    calls = {"n": 0}

    def flaky_embed(texts, *, query=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "embedding request failed (test)"
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    conn, stats = _index(repo, tmp_path, embed_fn=flaky_embed)
    assert stats["warnings"] == ["embedding request failed (test)"]
    assert stats["symbols"] == 3

    # the file stat did not change, but the missing vectors are retried
    stats = index_repo(conn, repo, embed_fn=flaky_embed)
    assert stats["warnings"] == []
    assert stats["files"] == 0
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(repo),)
    ).fetchone()[0]
    assert vectorized == 3


def test_expansion_keeps_raw_query_first(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def exact_unique_symbol(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "exact_unique_symbol",
        k=5,
        expand_fn=lambda q: (["something completely unrelated"], None),
    )
    assert report.candidates[0].qualname == "exact_unique_symbol"


def test_fts_search_handles_non_ascii_queries(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "retry.py").write_text(
        'def retry_request():\n    """повторить неудачный запрос"""\n    pass\n'
    )
    git(repo, "add", "retry.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    ids = fts_search(conn, str(repo), "повторить неудачный запрос")
    assert ids


def test_status_reports_staleness(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert "stale: no" in status_text(conn, str(repo), dense=False)

    (repo / "sample.py").write_text("def changed(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "edit")

    text = status_text(conn, str(repo), dense=False)
    assert "HEAD moved since indexing" in text
    assert "files changed since indexing" in text
    assert "working_tree_dirty: no" in text


def test_status_reports_dirty_working_tree(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    (repo / "sample.py").write_text("def edited(): pass\n")
    assert "working_tree_dirty: yes" in status_text(conn, str(repo), dense=False)


def test_docstring_taken_only_from_first_statement():
    source = '''\
def process(data):
    """Real docstring."""
    summary = """triple quoted but not a docstring"""
    return summary


class Widget:
    def render(self):
        """Render the widget."""
        return ""

    @property
    def size(self):
        return 0
'''
    symbols = {s.qualname: s for s in parse_source(source.encode(), "python", "w.py").symbols}
    assert "Real docstring" in symbols["process"].docstring
    assert "not a docstring" not in symbols["process"].docstring
    assert symbols["Widget"].docstring == ""
    assert "Render the widget" in symbols["Widget.render"].docstring


def test_language_coverage():
    cases = [
        (
            "c",
            "int add(int a, int b) { return a + b; }\n",
            [("add", "function")],
        ),
        (
            "cpp",
            (
                "struct Point { int x; };\nclass Engine { public: void start(); };\n"
                "void Engine::start() { run(); }\n"
            ),
            [("Point", "struct"), ("Engine", "class"), ("Engine::start", "function")],
        ),
        (
            "ruby",
            "class Greeter\n  def greet\n  end\nend\n",
            [("Greeter", "class"), ("Greeter.greet", "method")],
        ),
        (
            "go",
            "type Point struct { X int }\ntype Reader interface { Read() }\n",
            [("Point", "struct"), ("Reader", "interface")],
        ),
        (
            "typescript",
            "interface Opts { a: number }\ntype Alias = string\n",
            [("Opts", "interface"), ("Alias", "type_alias")],
        ),
        (
            "csharp",
            "struct Point { int X; }\ninterface I { void M(); }\n",
            [("Point", "struct"), ("I", "interface")],
        ),
    ]
    for lang, source, expected in cases:
        symbols = parse_source(source.encode(), lang, f"sample.{lang}").symbols
        actual = [(s.qualname, s.kind) for s in symbols]
        for pair in expected:
            assert pair in actual, f"{lang}: {pair} missing from {actual}"


def _vector(texts, *, query=False):
    return [b"\x00\x00\x80?" * 4] * len(texts), None


def test_file_changed_during_embedding_is_reprocessed(tmp_path):
    repo = init_repo(tmp_path / "repo")
    target = repo / "sample.py"
    target.write_text("def original(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")

    def embed_that_edits_file(texts, *, query=False):
        # the file changes while inference is in flight
        target.write_text("def replacement(): pass\n")
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    conn, _stats = _index(repo, tmp_path, embed_fn=embed_that_edits_file)
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"original"}

    # the captured stat (not the late one) was recorded, so the next
    # refresh must notice the change and reindex the file
    stats = index_repo(conn, repo, embed_fn=_vector)
    assert stats["warnings"] == []
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"replacement"}


def test_parse_error_keeps_previous_symbols_and_retries(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([], "error", "boom")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")

    stats = index_repo(conn, repo)
    assert stats["warnings"] == [
        "sample.py: error (boom); kept previous symbols as a stale fallback"
    ]
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"good"}

    # the file is not marked indexed: every refresh retries it
    stats = index_repo(conn, repo)
    assert stats["warnings"] == [
        "sample.py: error (boom); kept previous symbols as a stale fallback"
    ]


def test_unsupported_parser_keeps_previous_symbols(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([], "unsupported", "gone")
    )
    (repo / "sample.py").write_text("def other(): pass\n")

    stats = index_repo(conn, repo)
    assert stats["warnings"] == [
        "sample.py: unsupported (gone); kept previous symbols as a stale fallback"
    ]
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"good"}


def test_partial_parse_indexes_symbols_with_warning(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    symbol = parse_source(b"def partial(): pass\n", "python", "sample.py").symbols[0]
    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([symbol], "partial", "syntax")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")

    stats = index_repo(conn, repo)
    assert any("partial parse" in warning for warning in stats["warnings"])
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"partial"}

    # partial results are published and the file is marked indexed: no retry
    stats = index_repo(conn, repo)
    assert stats["warnings"] == []
    assert stats["files"] == 0


def _hybrid_order(conn, repo: Path) -> list[str]:
    report = search(conn, str(repo), "sample.py", k=3)
    return [candidate.qualname for candidate in report.candidates]


def test_rerank_incomplete_order_keeps_hybrid_order(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: ([(0, 9.0)], None),
    )
    assert [c.qualname for c in report.candidates] == _hybrid_order(conn, repo)
    assert any("invalid or incomplete (1/3" in warning for warning in report.warnings)


def test_rerank_duplicate_index_keeps_hybrid_order(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: ([(0, 5.0), (0, 6.0), (1, 1.0)], None),
    )
    assert [c.qualname for c in report.candidates] == _hybrid_order(conn, repo)
    assert any("invalid or incomplete" in warning for warning in report.warnings)


def test_valid_rerank_reorders_candidates(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)
    hybrid = _hybrid_order(conn, repo)

    def reverse_rerank(query, documents):
        return [(index, float(index)) for index in range(len(documents))], None

    report = search(conn, str(repo), "sample.py", k=3, rerank_fn=reverse_rerank)
    assert [c.qualname for c in report.candidates] == list(reversed(hybrid))


def test_fetch_is_scoped_to_repo(tmp_path):
    repo_a = init_repo(tmp_path / "repo_a")
    repo_b = init_repo(tmp_path / "repo_b")
    for repo in (repo_a, repo_b):
        (repo / "sample.py").write_text("def shared_name(): pass\n")
        git(repo, "add", "sample.py")
        git(repo, "commit", "-q", "-m", "init")

    conn = _connect(tmp_path)
    index_repo(conn, repo_a)
    index_repo(conn, repo_b)
    ids_b = [
        row[0] for row in conn.execute("SELECT id FROM symbols WHERE repo = ?", (str(repo_b),))
    ]
    assert ids_b
    assert _fetch(conn, ids_b, str(repo_a)) == {}
    assert _fetch(conn, ids_b, str(repo_b))


def test_symlinked_parent_directory_is_not_indexed(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "code.py").write_text("def inside(): pass\n")
    git(repo, "add", "pkg/code.py")
    git(repo, "commit", "-q", "-m", "init")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "code.py").write_text("def leaked(): pass\n")
    shutil.rmtree(repo / "pkg")
    (repo / "pkg").symlink_to(outside)

    conn, stats = _index(repo, tmp_path)
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == set()
    assert any("unreadable" in warning for warning in stats["warnings"])


def test_index_repo_crash_does_not_poison_connection(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    def exploding_drop(conn, ids):
        conn.execute("DELETE FROM files WHERE repo = 'nonexistent'")
        raise KeyboardInterrupt("simulated crash mid-index")

    monkeypatch.setattr(pipeline, "_drop_symbols", exploding_drop)
    (repo / "sample.py").write_text("def changed(): pass\n")
    with pytest.raises(KeyboardInterrupt):
        index_repo(conn, repo)
    assert not conn.in_transaction

    # the same connection must keep serving searches after the crash
    monkeypatch.undo()
    report = search(conn, str(repo), "good", k=3)
    assert report.symbol_count == 1


def test_embedding_failure_without_warning_uses_default_warning(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")

    conn, stats = _index(repo, tmp_path, embed_fn=lambda texts, *, query=False: (None, None))
    assert stats["files"] == 1
    assert stats["symbols"] == 1
    assert stats["warnings"] == ["embedding failed; dense coverage stays partial"]
    assert _file_rows(conn, repo) == {"sample.py"}


def test_rerank_ignores_non_dict_and_bad_index_items(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    monkeypatch.setattr(
        rerank_mod,
        "post_json",
        lambda *args, **kwargs: {
            "results": [
                None,
                {"index": "0", "relevance_score": 0.9},
                {"index": 0.7, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        },
    )
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example/v1",
        rerank_model="reranker",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("query", ["doc0", "doc1"])
    assert order == [(1, 0.5)]
    assert warning == "rerank response had 3 malformed items"


def test_rerank_raw_format_passes_query_unchanged(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    seen = {}

    def fake_post_json(url, body, api_key, timeout):
        seen["query"] = body["query"]
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    monkeypatch.setattr(rerank_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example/v1",
        rerank_model="reranker",
        rerank_query_format="raw",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("retry failed calls", ["doc0"])
    assert order == [(0, 0.9)]
    assert warning is None
    assert seen["query"] == "retry failed calls"


def test_rerank_instruct_format_wraps_query(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    seen = {}

    def fake_post_json(url, body, api_key, timeout):
        seen["query"] = body["query"]
        return {"results": [{"index": 0, "relevance_score": 0.9}]}

    monkeypatch.setattr(rerank_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example/v1",
        rerank_model="reranker",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("retry failed calls", ["doc0"])
    assert order == [(0, 0.9)]
    assert warning is None
    assert seen["query"].startswith("<Instruct>: ")
    assert seen["query"].endswith("\n<Query>: retry failed calls")


def test_valid_rerank_rejects_bool_and_nonfinite_scores():
    assert valid_rerank_order([(0, 1.0), (1, 0.5)], 2) is True
    assert valid_rerank_order([(0, True), (1, False)], 2) is False
    assert valid_rerank_order([(0, float("nan")), (1, 0.5)], 2) is False
    assert valid_rerank_order([(0, float("inf")), (1, 0.5)], 2) is False
    assert valid_rerank_order([("0", 1.0), (1, 0.5)], 2) is False


def test_search_dense_path_inside_read_transaction(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path, embed_fn=_vector)
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(repo),)
    ).fetchone()[0]
    assert vectorized == 3

    report = search(conn, str(repo), "split unified diff", k=3, embed_fn=_vector)
    assert report.candidates
    assert report.warnings == []
    assert not conn.in_transaction


def test_partial_parse_state_persists_across_refreshes(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    symbol = parse_source(b"def partial(): pass\n", "python", "sample.py").symbols[0]
    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([symbol], "partial", "syntax")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")

    stats = index_repo(conn, repo)
    assert any("partial parse" in warning for warning in stats["warnings"])

    # the warning disappears on the next unchanged refresh, but the state persists
    stats = index_repo(conn, repo)
    assert stats["warnings"] == []

    status, attempted_at = conn.execute(
        "SELECT status, attempted_at FROM parse_state WHERE repo = ? AND path = 'sample.py'",
        (str(repo),),
    ).fetchone()
    assert status == "partial"
    assert attempted_at > 0
    assert "1 partial" in status_text(conn, str(repo), dense=False)

    report = search(conn, str(repo), "partial symbol", k=3)
    assert report.parse_coverage == {"partial": 1}
    assert "parse issues: 1 partial" in format_report(report)

    # an exact identifier match answers without models and without the
    # full pipeline (parse coverage is not part of the exact report)
    exact = search(conn, str(repo), "partial", k=3)
    assert exact.stages_used == ["exact"]
    assert exact.candidates[0].qualname == "partial"


def test_parse_error_state_persists_and_retries(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([], "error", "boom")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")

    for _ in range(2):
        stats = index_repo(conn, repo)
        assert stats["warnings"] == [
            "sample.py: error (boom); kept previous symbols as a stale fallback"
        ]

    status = conn.execute(
        "SELECT status FROM parse_state WHERE repo = ? AND path = 'sample.py'", (str(repo),)
    ).fetchone()[0]
    assert status == "error"
    assert "1 error" in status_text(conn, str(repo), dense=False)


def test_unreadable_file_state_is_persisted(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(pipeline, "capture_file", lambda root, rel: None)
    (repo / "sample.py").write_text("def changed(): pass\n")

    stats = index_repo(conn, repo)
    assert any("unreadable" in warning for warning in stats["warnings"])
    status = conn.execute(
        "SELECT status FROM parse_state WHERE repo = ? AND path = 'sample.py'", (str(repo),)
    ).fetchone()[0]
    assert status == "unreadable"


def test_removed_file_cleans_parse_state(tmp_path, monkeypatch):
    from priorart.indexing import pipeline

    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text("def good(): pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    symbol = parse_source(b"def partial(): pass\n", "python", "sample.py").symbols[0]
    monkeypatch.setattr(
        pipeline, "parse_source", lambda data, lang, rel: ParseResult([symbol], "partial", "syntax")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")
    index_repo(conn, repo)
    assert (
        conn.execute("SELECT COUNT(*) FROM parse_state WHERE repo = ?", (str(repo),)).fetchone()[0]
        == 1
    )

    (repo / "sample.py").unlink()
    git(repo, "rm", "-q", "sample.py")
    git(repo, "commit", "-q", "-m", "remove")
    index_repo(conn, repo)

    assert (
        conn.execute("SELECT COUNT(*) FROM parse_state WHERE repo = ?", (str(repo),)).fetchone()[0]
        == 0
    )
    assert "parse:" not in status_text(conn, str(repo), dense=False)


def test_preflight_parsers_reports_original_error(monkeypatch):
    from priorart.indexing import parser

    monkeypatch.setattr(parser, "_parsers", {})
    monkeypatch.setattr(parser, "_parser_errors", {})

    def refusing(lang):
        raise ConnectionError("connection refused while downloading grammar")

    monkeypatch.setattr(parser, "get_parser", refusing)
    failures = parser.preflight_parsers(["go", "python"])
    assert set(failures) == {"go", "python"}
    assert all(reason.startswith("ConnectionError:") for reason in failures.values())

    result = parser.parse_source(b"x = 1", "python", "x.py")
    assert result.status == "unsupported"
    assert "ConnectionError: connection refused while downloading grammar" in result.detail


def test_preflight_parsers_passes_when_parsers_load(monkeypatch):
    from priorart.indexing import parser

    monkeypatch.setattr(parser, "_parsers", {})
    monkeypatch.setattr(parser, "_parser_errors", {})

    assert parser.preflight_parsers(["python"]) == {}


def test_repo_languages_lists_languages_of_tracked_files(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "code.py").write_text("def f(): pass\n")
    (repo / "tool.go").write_text("package main\n")
    git(repo, "add", "code.py", "tool.go")
    git(repo, "commit", "-q", "-m", "init")

    assert repo_languages(repo) == ["go", "python"]


def _qualname_to_id(conn, repo: Path) -> dict[str, int]:
    return {
        row[1]: row[0]
        for row in conn.execute("SELECT id, qualname FROM symbols WHERE repo = ?", (str(repo),))
    }


def test_search_records_stage_trace(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path, embed_fn=_vector)

    report = search(
        conn,
        str(repo),
        "split unified diff",
        k=3,
        expand_fn=lambda q: (["diff viewer"], None),
        embed_fn=_vector,
        rerank_fn=lambda query, documents: (
            [(index, float(len(documents) - index)) for index in range(len(documents))],
            None,
        ),
    )
    trace = report.trace
    assert trace is not None
    assert trace.queries == ["split unified diff", "diff viewer"]
    assert set(trace.stage_seconds) == {"expand", "embed", "retrieve", "rerank"}
    assert all(0 <= seconds < 60 for seconds in trace.stage_seconds.values())
    assert trace.fts_rankings
    assert all(trace.fts_rankings)
    assert trace.vec_rankings
    assert all(trace.vec_rankings)
    assert len(trace.fused) >= len(report.candidates)
    fused_scores = [score for _symbol_id, score in trace.fused]
    assert fused_scores == sorted(fused_scores, reverse=True)
    qualname_to_id = _qualname_to_id(conn, repo)
    assert trace.rerank_order == [qualname_to_id[c.qualname] for c in report.candidates]
    assert {symbol_id for symbol_id, _score in trace.fused} == {
        qualname_to_id[c.qualname] for c in report.candidates
    }


def test_search_trace_records_rerank_fallback(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: ([(0, 9.0)], None),
    )
    trace = report.trace
    assert trace is not None
    assert trace.queries == ["sample.py", "sample py"]
    assert trace.vec_rankings == []
    assert trace.rerank_order is None
    assert any("invalid or incomplete" in warning for warning in report.warnings)
    qualname_to_id = _qualname_to_id(conn, repo)
    assert [symbol_id for symbol_id, _score in trace.fused] == [
        qualname_to_id[c.qualname] for c in report.candidates
    ]


def _completion_payload(logprobs):
    top = [{"token": t, "logprob": lp} for t, lp in logprobs]
    return {"completion_probabilities": [{"top_logprobs": top}]}


def test_rerank_llama_completion_scores_from_yes_no_logprobs(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    seen = {"prompts": []}

    def fake_post_json(url, body, api_key, timeout):
        seen["url"] = url
        seen["n_predict"] = body["n_predict"]
        seen["prompts"].append(body["prompt"])
        if body["prompt"].count("doc-one") > 0:
            return _completion_payload([("yes", -0.2), ("Yes", -3.0), ("no", -4.0), ("true", -6.0)])
        return _completion_payload([("no", -0.1), ("No", -3.0), ("yes", -5.0), ("false", -6.0)])

    monkeypatch.setattr(rerank_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example/v1",
        rerank_model="priorart-rerank",
        rerank_protocol="llama-completion",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("retry failed calls", ["doc-one", "doc-two"])
    assert warning is None
    assert len(order) == 2
    assert order[0][0] == 0
    assert order[0][1] > 0.97
    assert order[1][0] == 1
    assert order[1][1] < 0.03
    assert seen["url"] == "http://rerank.example/completion"
    assert seen["n_predict"] == 1
    prompts = seen["prompts"]
    assert len(prompts) == 2
    for prompt in prompts:
        assert "retry failed calls" in prompt
        assert "Judge whether the Document meets the requirements" in prompt
        assert rerank_mod.RERANK_INSTRUCTION in prompt
        assert prompt.endswith("\n\n")
    assert "doc-one" in prompts[0]
    assert "doc-two" in prompts[1]


def test_rerank_llama_completion_missing_token_floors_score(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    def fake_post_json(url, body, api_key, timeout):
        return _completion_payload([("yes", -0.3), ("Yes", -3.0), ("true", -4.0)])

    monkeypatch.setattr(rerank_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example",
        rerank_model="priorart-rerank",
        rerank_protocol="llama-completion",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("query", ["doc"])
    assert warning is None
    assert order[0][1] > 0.9


def test_rerank_llama_completion_without_yes_no_keeps_hybrid_order(monkeypatch, tmp_path):
    from priorart.models import rerank as rerank_mod

    def fake_post_json(url, body, api_key, timeout):
        return _completion_payload([("true", -0.3), ("false", -1.0)])

    monkeypatch.setattr(rerank_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        rerank_base_url="http://rerank.example/v1",
        rerank_model="priorart-rerank",
        rerank_protocol="llama-completion",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("query", ["doc"])
    assert order is None
    assert warning == "rerank completion had no yes/no logprobs; kept hybrid order"


def test_rerank_prompt_template_matches_official_qwen3_reranker_usage():
    from priorart.models.rerank import RERANK_PROMPT_TEMPLATE

    template = RERANK_PROMPT_TEMPLATE
    assert "<|im_start|>system\n" in template
    assert '"yes" or "no"' in template
    assert "<Instruct>: " in template
    assert "<Query>: {query}" in template
    assert "<Document>: {document}" in template
    assert "assistant\n" + chr(60) + "think" + chr(62) in template
    assert chr(60) + "/think" + chr(62) in template


def test_query_expansion_null_content_falls_back_to_raw_query(monkeypatch, tmp_path):
    from priorart.models import expand as expand_mod

    monkeypatch.setattr(
        expand_mod,
        "post_json",
        lambda *args, **kwargs: {"choices": [{"message": {"content": None}}]},
    )
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        llm_base_url="http://llm.example/v1",
        llm_model="chat",
    )
    expand = expand_mod.make_expander(config)
    queries, warning = expand("find the handler")
    assert queries == ["find the handler"]
    assert "query expansion returned no content" in warning


def test_embedding_payload_data_order_follows_index_field(monkeypatch, tmp_path):
    from priorart.models import embed as embed_mod

    def fake_post_json(url, body, api_key, timeout):
        assert [item.rsplit("Text: ", 1)[1] for item in body["input"]] == ["text-a", "text-b"]
        return {
            "data": [
                {"index": 1, "embedding": [0.2, 0.2, 0.2, 0.2]},
                {"index": 0, "embedding": [0.1, 0.1, 0.1, 0.1]},
            ]
        }

    monkeypatch.setattr(embed_mod, "post_json", fake_post_json)
    from tests.helpers import make_config

    config = make_config(
        tmp_path,
        embed_base_url="http://embed.example/v1",
        embed_model="embedder",
    )
    embed = embed_mod.make_embedder(config)
    vectors, warning = embed(["text-a", "text-b"])
    assert warning is None
    import sqlite_vec

    assert vectors == [
        sqlite_vec.serialize_float32([0.1, 0.1, 0.1, 0.1]),
        sqlite_vec.serialize_float32([0.2, 0.2, 0.2, 0.2]),
    ]


def test_rerank_prompt_substitution_is_single_pass():
    from priorart.models.rerank import RERANK_PROMPT_TEMPLATE, _fill_template

    prompt = _fill_template(RERANK_PROMPT_TEMPLATE, "weird {document} query", "<body>")
    assert "weird {document} query" in prompt
    assert "<body>" in prompt
    assert prompt.count("{document}") == 1


def test_rerank_missing_score_counts_as_malformed():
    from priorart.models.rerank import _parse_results

    order, warning = _parse_results([{"index": 0}, {"index": 1, "relevance_score": 0.4}])
    assert order == [(1, 0.4)]
    assert warning == "rerank response had 1 malformed items"


def test_search_warns_when_dense_channel_is_empty(tmp_path):
    repo = init_repo(tmp_path / "repo")
    (repo / "sample.py").write_text(SAMPLE)
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    conn, _stats = _index(repo, tmp_path)

    def embed_fn(texts, *, query=False):
        import sqlite_vec

        return [sqlite_vec.serialize_float32([0.1, 0.2, 0.3, 0.4])] * len(texts), None

    report = search(conn, str(repo), "parse diff patch", embed_fn=embed_fn)
    assert any("dense channel is empty" in warning for warning in report.warnings)

    _index(repo, tmp_path, embed_fn=embed_fn, rebuild=True)
    report = search(conn, str(repo), "parse diff patch", embed_fn=embed_fn)
    assert not any("dense channel is empty" in warning for warning in report.warnings)
