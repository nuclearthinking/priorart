from __future__ import annotations

import subprocess
from pathlib import Path

from priorart.indexer import index_repo, parse_source
from priorart.search import fts_search, search, status_text
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
    subprocess.run(
        ["git", "-C", str(repo), *args],
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


def _index(repo: Path, tmp_path: Path, embed_fn=None, rebuild=False):
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
    symbols = {s.qualname: s for s in parse_source(source.encode(), "python", "w.py")}
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
        symbols = parse_source(source.encode(), lang, f"sample.{lang}")
        actual = [(s.qualname, s.kind) for s in symbols]
        for pair in expected:
            assert pair in actual, f"{lang}: {pair} missing from {actual}"
