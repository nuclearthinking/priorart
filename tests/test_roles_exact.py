"""Source roles, exact dispatch and intent contracts (deliveries D/F).

R01: an exact symbol name answers without any model. Exact test-function
queries stay findable in the default intent. intent=tests restricts the
pool; balanced keeps generative expansion off, deep enables it; production
preference reorders only comparable relevance.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from priorart.core.errors import INDEX_NOT_READY, PriorartError
from priorart.core.jobs import FINAL_STATES
from priorart.indexing.roles import source_role
from priorart.registry import RuntimeRegistry
from tests.helpers import git, init_repo, make_config

SAMPLE = "def production_owner():\n    pass\n"


def _repo_with_files(path: Path, files: dict[str, str]) -> Path:
    repo = init_repo(path)
    for rel, content in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        git(repo, "add", rel)
    git(repo, "commit", "-q", "-m", "init")
    return repo


def _refresh(registry: RuntimeRegistry, repo: Path):
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _handle, current = registry.get_job(job.job_id)
        if current.state in FINAL_STATES:
            return handle, current
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _make_registry(tmp_path: Path):
    return RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))


# --- role conventions --------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("src/priorart/server.py", "production"),
        ("tests/test_units.py", "test"),
        ("src/thing/test_thing.py", "test"),
        ("src/thing_test.go", "test"),
        ("src/thing.spec.ts", "test"),
        ("conftest.py", "fixture"),
        ("tests/fixtures/data.json", "fixture"),
        ("src/conftest.py", "fixture"),
        ("examples/run.py", "example"),
        ("src/generated/api.g.dart", "generated"),
        ("src/unknown.bin", "production"),
        ("", "unknown"),
    ],
)
def test_source_role_conventions(rel, expected):
    assert source_role(rel) == expected


def test_roles_are_stored_on_symbols(tmp_path):
    repo = _repo_with_files(
        tmp_path / "repo",
        {
            "src/app.py": "def owner():\n    pass\n",
            "tests/test_app.py": "def test_owner():\n    pass\n",
            "tests/conftest.py": "def fixture_helper():\n    pass\n",
        },
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        reader = handle.reader()
        try:
            roles = dict(
                reader.execute(
                    "SELECT qualname, source_role FROM symbols WHERE repo = ?",
                    (str(repo.resolve()),),
                ).fetchall()
            )
        finally:
            reader.close()
        assert roles == {
            "owner": "production",
            "test_owner": "test",
            "fixture_helper": "fixture",
        }
    finally:
        registry.close()


# --- exact dispatch (R01) ---------------------------------------------------


def test_exact_qualname_answers_without_models(tmp_path, monkeypatch):
    repo = _repo_with_files(
        tmp_path / "repo",
        {"src/app.py": "def GenerationRecovery():\n    pass\n"},
    )
    calls = {"embed": 0, "rerank": 0, "expand": 0}

    import priorart.models

    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: _counting(calls, "embed"))
    monkeypatch.setattr(priorart.models, "make_reranker", lambda config: _counting(calls, "rerank"))
    monkeypatch.setattr(priorart.models, "make_expander", lambda config: _counting(calls, "expand"))

    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        calls["embed"] = 0  # discard the indexing-phase document embeddings
        report = handle.search("GenerationRecovery", k=5)
        assert report.stages_used == ["exact"]
        assert [c.qualname for c in report.candidates] == ["GenerationRecovery"]
        assert calls == {"embed": 0, "rerank": 0, "expand": 0}
    finally:
        registry.close()


def _counting(calls, name):
    def fn(*args, **kwargs):
        calls[name] += 1
        return None, None

    return fn


def test_exact_name_of_test_function_found_in_default_intent(tmp_path):
    repo = _repo_with_files(
        tmp_path / "repo",
        {
            "src/app.py": "def handler():\n    pass\n",
            "tests/test_app.py": "def test_handler_retries():\n    pass\n",
        },
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("test_handler_retries", k=5)
        assert report.stages_used == ["exact"]
        assert report.candidates[0].qualname == "test_handler_retries"
        assert report.candidates[0].source_role == "test"
    finally:
        registry.close()


def test_multiple_definitions_return_bounded_variants(tmp_path):
    repo = _repo_with_files(
        tmp_path / "repo",
        {
            "src/a.py": "def duplicate():\n    pass\n",
            "src/b.py": "def duplicate():\n    pass\n",
        },
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("duplicate", k=5)
        assert report.stages_used == ["exact"]
        assert sorted(c.path for c in report.candidates) == ["src/a.py", "src/b.py"]
    finally:
        registry.close()


def test_natural_language_query_skips_exact_dispatch(tmp_path):
    repo = _repo_with_files(
        tmp_path / "repo",
        {"src/app.py": "def handler():\n    pass\n"},
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("where is the handler implemented", k=5)
        assert report.stages_used != ["exact"]
    finally:
        registry.close()


def test_exact_match_on_missing_index_still_raises_not_ready(tmp_path):
    repo = _repo_with_files(tmp_path / "repo", {"src/app.py": SAMPLE})
    registry = _make_registry(tmp_path)
    try:
        with pytest.raises(PriorartError, match="not built yet") as err:
            registry.resolve(repo).search("anything")
        assert err.value.code == INDEX_NOT_READY
    finally:
        registry.close()


# --- intent contracts -------------------------------------------------------


def test_intent_tests_restricts_pool_to_tests(tmp_path):
    repo = _repo_with_files(
        tmp_path / "repo",
        {
            "src/app.py": "def owner():\n    pass\n",
            "tests/test_app.py": "def test_owner():\n    pass\n",
        },
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("owner handler", k=5, intent="tests")
        assert report.candidates
        assert {c.source_role for c in report.pool} <= {"test", "fixture"}
        assert report.candidates[0].qualname == "test_owner"
    finally:
        registry.close()


def test_production_prior_orders_comparable_relevance(tmp_path, monkeypatch):
    repo = _repo_with_files(
        tmp_path / "repo",
        {
            "src/app.py": "def resolve_workspace():\n    pass\n",
            "tests/test_app.py": "def resolve_workspace_test_double():\n    pass\n",
        },
    )
    # a reranker that scores both candidates identically: the prior decides
    import priorart.models

    monkeypatch.setattr(
        priorart.models,
        "make_reranker",
        lambda config: lambda q, docs: ([(0, 0.5), (1, 0.5)], None),
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("resolve workspace", k=5, intent="implementation")
        roles = [c.source_role for c in report.candidates]
        assert roles[0] == "production"
        # neutral intent keeps the raw order
        neutral = handle.search("resolve workspace", k=5, intent="any")
        assert neutral.candidates == report.candidates or {
            c.qualname for c in neutral.candidates
        } == {c.qualname for c in report.candidates}
    finally:
        registry.close()


# --- mode semantics ----------------------------------------------------------


def test_balanced_mode_uses_no_generative_expansion(tmp_path, monkeypatch):
    repo = _repo_with_files(tmp_path / "repo", {"src/app.py": SAMPLE})
    expand_calls: list[str] = []
    import priorart.models

    monkeypatch.setattr(
        priorart.models,
        "make_expander",
        lambda config: lambda query: (expand_calls.append(query) or ["extra variant"], None),
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        handle.search("the production owner function", k=5, mode="balanced")
        assert expand_calls == []
        handle.search("the production owner function", k=5, mode="deep")
        assert expand_calls == ["the production owner function"]
    finally:
        registry.close()


def test_deep_mode_bypasses_exact_dispatch(tmp_path, monkeypatch):
    repo = _repo_with_files(tmp_path / "repo", {"src/app.py": SAMPLE})
    embed_calls: list[list[str]] = []
    import priorart.models

    monkeypatch.setattr(
        priorart.models,
        "make_embedder",
        lambda config: (
            lambda texts, *, query=False: (embed_calls.append(list(texts)) or [b"x" * 16], None)
        ),
    )
    registry = _make_registry(tmp_path)
    try:
        handle, _job = _refresh(registry, repo)
        report = handle.search("production_owner", k=5, mode="deep")
        assert report.stages_used != ["exact"]
        assert embed_calls  # deep ran the embedding stage on the raw query
    finally:
        registry.close()
