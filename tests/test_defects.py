from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from priorart.indexer import ParseResult, index_repo, parse_source
from priorart.search import _fetch, _valid_rerank, fts_search, search, status_text
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


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo


def _index(repo: Path, tmp_path: Path, embed_fn=None, *, rebuild=False):
    conn = connect(tmp_path / "test.db", embed_dim=4)
    return conn, index_repo(conn, repo, embed_fn=embed_fn, rebuild=rebuild)


def _file_rows(conn, repo: Path) -> set[str]:
    return {row[0] for row in conn.execute("SELECT path FROM files WHERE repo = ?", (str(repo),))}


def test_worktree_indexes_only_tracked_files(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.py").write_text("def tracked(): pass\n")
    (repo / ".gitignore").write_text("ignored.py\n")
    _git(repo, "add", "tracked.py", ".gitignore")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "ignored.py").write_text("def ignored(): pass\n")
    (repo / "untracked.py").write_text("def untracked(): pass\n")

    worktree = tmp_path / "worktree"
    _git(repo, "worktree", "add", "-q", str(worktree))

    conn, _stats = _index(worktree, tmp_path)
    assert _file_rows(conn, worktree) == {"tracked.py"}
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(worktree),))
    }
    assert names == {"tracked"}


def test_symlink_outside_repo_is_skipped(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.py").write_text("def tracked(): pass\n")
    outside = tmp_path / "outside.py"
    outside.write_text("def secret(): '''s3cr3t doc'''\n    pass\n")
    (repo / "linked.py").symlink_to(outside.resolve())
    _git(repo, "add", "tracked.py", "linked.py")
    _git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert _file_rows(conn, repo) == {"tracked.py"}
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"tracked"}


def test_rebuild_removes_vanished_files(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "kept.py").write_text("def kept(): pass\n")
    (repo / "gone.py").write_text("def gone(): pass\n")
    _git(repo, "add", "kept.py", "gone.py")
    _git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert _file_rows(conn, repo) == {"kept.py", "gone.py"}

    (repo / "gone.py").unlink()
    _git(repo, "rm", "-q", "gone.py")
    _git(repo, "commit", "-q", "-m", "remove")

    conn, stats = _index(repo, tmp_path, rebuild=True)
    assert stats["removed"] == 1
    assert _file_rows(conn, repo) == {"kept.py"}


def test_embedding_failure_is_retried(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")

    calls = {"n": 0}

    def flaky_embed(texts, *, query=False):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, "embedding request failed (test)"
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    conn, stats = _index(repo, tmp_path, embed_fn=flaky_embed)
    assert stats["warnings"] == ["embedding request failed (test)"]
    assert _file_rows(conn, repo) == set()

    stats = index_repo(conn, repo, embed_fn=flaky_embed)
    assert stats["warnings"] == []
    assert stats["symbols"] == 3
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(repo),)
    ).fetchone()[0]
    assert vectorized == 3


def test_expansion_keeps_raw_query_first(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def exact_unique_symbol(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
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
    repo = _init_repo(tmp_path)
    (repo / "retry.py").write_text(
        'def retry_request():\n    """повторить неудачный запрос"""\n    pass\n'
    )
    _git(repo, "add", "retry.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    ids = fts_search(conn, str(repo), "повторить неудачный запрос")
    assert ids


def test_status_reports_staleness(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")

    conn, _ = _index(repo, tmp_path)
    assert "stale: no" in status_text(conn, str(repo), dense=False)

    (repo / "sample.py").write_text("def changed(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "edit")

    text = status_text(conn, str(repo), dense=False)
    assert "HEAD moved since indexing" in text
    assert "files changed since indexing" in text
    assert "working_tree_dirty: no" in text


def test_status_reports_dirty_working_tree(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
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
    repo = _init_repo(tmp_path)
    target = repo / "sample.py"
    target.write_text("def original(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")

    def embed_that_edits_file(texts, *, query=False):
        # the file changes while inference is in flight
        target.write_text("def replacement(): pass\n")
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    conn, stats = _index(repo, tmp_path, embed_fn=embed_that_edits_file)
    assert any("changed during indexing" in warning for warning in stats["warnings"])
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
    from priorart import indexer

    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def good(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(
        indexer, "parse_source", lambda data, lang, rel: ParseResult([], "error", "boom")
    )
    (repo / "sample.py").write_text("def changed(): pass\n")

    stats = index_repo(conn, repo)
    assert stats["warnings"] == ["sample.py: parse error (boom); kept previous symbols"]
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"good"}

    # the file is not marked indexed: every refresh retries it
    stats = index_repo(conn, repo)
    assert stats["warnings"] == ["sample.py: parse error (boom); kept previous symbols"]


def test_unsupported_parser_keeps_previous_symbols(tmp_path, monkeypatch):
    from priorart import indexer

    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def good(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    monkeypatch.setattr(
        indexer, "parse_source", lambda data, lang, rel: ParseResult([], "unsupported", "gone")
    )
    (repo / "sample.py").write_text("def other(): pass\n")

    stats = index_repo(conn, repo)
    assert stats["warnings"] == ["sample.py: parse unsupported (gone); kept previous symbols"]
    names = {
        row[0] for row in conn.execute("SELECT name FROM symbols WHERE repo = ?", (str(repo),))
    }
    assert names == {"good"}


def test_partial_parse_indexes_symbols_with_warning(tmp_path, monkeypatch):
    from priorart import indexer

    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def good(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    symbol = parse_source(b"def partial(): pass\n", "python", "sample.py").symbols[0]
    monkeypatch.setattr(
        indexer, "parse_source", lambda data, lang, rel: ParseResult([symbol], "partial", "syntax")
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


def test_rerank_incomplete_order_keeps_hybrid_order(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: ([(0, 9.0)], None),
    )
    assert [c.qualname for c in report.candidates] == [
        "DiffViewer",
        "DiffViewer.render",
        "parse_diff_patch",
    ]
    assert any("invalid or incomplete (1/3" in warning for warning in report.warnings)


def test_rerank_duplicate_index_keeps_hybrid_order(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: ([(0, 5.0), (0, 6.0), (1, 1.0)], None),
    )
    assert [c.qualname for c in report.candidates] == [
        "DiffViewer",
        "DiffViewer.render",
        "parse_diff_patch",
    ]
    assert any("invalid or incomplete" in warning for warning in report.warnings)


def test_valid_rerank_reorders_candidates(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    report = search(
        conn,
        str(repo),
        "sample.py",
        k=3,
        rerank_fn=lambda query, documents: (
            [(1, 0.9), (0, 0.5), (2, 0.1)],
            None,
        ),
    )
    assert [c.qualname for c in report.candidates] == [
        "DiffViewer.render",
        "DiffViewer",
        "parse_diff_patch",
    ]


def test_fetch_is_scoped_to_repo(tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    for repo in (repo_a, repo_b):
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "sample.py").write_text("def shared_name(): pass\n")
        _git(repo, "add", "sample.py")
        _git(repo, "commit", "-q", "-m", "init")

    conn = connect(tmp_path / "test.db", embed_dim=4)
    index_repo(conn, repo_a)
    index_repo(conn, repo_b)
    ids_b = [
        row[0] for row in conn.execute("SELECT id FROM symbols WHERE repo = ?", (str(repo_b),))
    ]
    assert ids_b
    assert _fetch(conn, ids_b, str(repo_a)) == {}
    assert _fetch(conn, ids_b, str(repo_b))


def test_symlinked_parent_directory_is_not_indexed(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "code.py").write_text("def inside(): pass\n")
    _git(repo, "add", "pkg/code.py")
    _git(repo, "commit", "-q", "-m", "init")

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
    from priorart import indexer

    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def good(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path)

    def exploding_drop(conn, ids):
        conn.execute("DELETE FROM files WHERE repo = 'nonexistent'")
        raise KeyboardInterrupt("simulated crash mid-index")

    monkeypatch.setattr(indexer, "_drop_symbols", exploding_drop)
    (repo / "sample.py").write_text("def changed(): pass\n")
    with pytest.raises(KeyboardInterrupt):
        index_repo(conn, repo)
    assert not conn.in_transaction

    # the same connection must keep serving searches after the crash
    monkeypatch.undo()
    report = search(conn, str(repo), "good", k=3)
    assert report.symbol_count == 1


def test_embedding_failure_without_warning_is_not_counted(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text("def good(): pass\n")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")

    conn, stats = _index(repo, tmp_path, embed_fn=lambda texts, *, query=False: (None, None))
    assert stats["files"] == 0
    assert stats["warnings"] == ["embedding failed without a warning"]
    assert _file_rows(conn, repo) == set()


def test_rerank_ignores_non_dict_and_bad_index_items(monkeypatch, tmp_path):
    from priorart import rerank as rerank_mod
    from priorart.config import Config

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
    config = Config(
        llm_base_url=None,
        llm_api_key=None,
        embed_base_url=None,
        embed_api_key=None,
        rerank_base_url="http://rerank.example/v1",
        rerank_api_key=None,
        embed_model="",
        embed_dim=4,
        rerank_model="reranker",
        llm_model="",
        db_path=tmp_path / "rerank-test.db",
    )
    rerank = rerank_mod.make_reranker(config)
    order, warning = rerank("query", ["doc0", "doc1"])
    assert order == [(1, 0.5)]
    assert warning == "rerank response had 3 malformed items"


def test_valid_rerank_rejects_bool_and_nonfinite_scores():
    assert _valid_rerank([(0, 1.0), (1, 0.5)], 2) is True
    assert _valid_rerank([(0, True), (1, False)], 2) is False
    assert _valid_rerank([(0, float("nan")), (1, 0.5)], 2) is False
    assert _valid_rerank([(0, float("inf")), (1, 0.5)], 2) is False
    assert _valid_rerank([("0", 1.0), (1, 0.5)], 2) is False


def test_search_dense_path_inside_read_transaction(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "sample.py").write_text(SAMPLE)
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-q", "-m", "init")
    conn, _ = _index(repo, tmp_path, embed_fn=_vector)
    vectorized = conn.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(repo),)
    ).fetchone()[0]
    assert vectorized == 3

    report = search(conn, str(repo), "split unified diff", k=3, embed_fn=_vector)
    assert report.candidates
    assert report.warnings == []
    assert not conn.in_transaction
