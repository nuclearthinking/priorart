"""Embedding cache: incremental vectorization, dedup and restart recovery.

Delivery E of the workspace architecture plan: one changed input is computed
once, identical inputs across symbols share inference, vectors survive
restarts and rebuilds through the shared cache, and query variants never
substitute each other's vectors.
"""

from __future__ import annotations

import time
from pathlib import Path

from priorart.core.jobs import FINAL_STATES
from priorart.models.embed import embedding_space_id
from priorart.models.embed_cache import EmbedCache, input_hash
from priorart.registry import RuntimeRegistry
from tests.helpers import git, init_repo, make_config

VECTOR = b"\x00\x00\x80?" * 4


def _counting_embed(calls: list[list[str]]):
    def embed(texts, *, query=False):
        calls.append(list(texts))
        return [VECTOR] * len(texts), None

    return embed


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "sample.py").write_text("def first():\n    pass\n\n\ndef second():\n    pass\n")
    git(repo, "add", "sample.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def _refresh(registry: RuntimeRegistry, repo: Path, **kwargs):
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle, **kwargs)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _handle, current = registry.get_job(job.job_id)
        if current.state in FINAL_STATES:
            return handle, current
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _patch_embed(monkeypatch, calls: list[list[str]]):
    import priorart.models

    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: _counting_embed(calls))


# --- unit level -------------------------------------------------------------


def test_cache_roundtrip_and_role_isolation(tmp_path):
    cache = EmbedCache(tmp_path / "embed-cache.db")
    try:
        cache.store("space", "document", [(input_hash("text one"), VECTOR)], dim=4)
        hits = cache.lookup("space", "document", ["text one"])
        assert hits == {input_hash("text one"): VECTOR}
        # the same input under a different role or space is a miss
        assert cache.lookup("space", "query", ["text one"]) == {}
        assert cache.lookup("other", "document", ["text one"]) == {}
        assert cache.lookup("space", "document", ["text two"]) == {}
    finally:
        cache.close()


def test_input_hash_keys_the_exact_text():
    assert input_hash("a") != input_hash("b")
    assert input_hash("a") == input_hash("a")


def test_corrupt_vector_is_ignored_on_lookup(tmp_path):
    cache = EmbedCache(tmp_path / "embed-cache.db")
    try:
        key = input_hash("real")
        cache.store("space", "document", [(key, VECTOR)], dim=4)
        with cache._mutex:  # sabotage the stored blob directly
            cache._conn.execute("UPDATE vectors SET vector = X'00' WHERE input_hash = ?", (key,))
            cache._conn.commit()
        assert cache.lookup("space", "document", ["real"]) == {}
    finally:
        cache.close()


def test_space_identity_changes_with_model_config(tmp_path):
    base = make_config(tmp_path, embed_model="m1")
    other = make_config(tmp_path, embed_model="m2")
    same = make_config(tmp_path, embed_model="m1")
    assert embedding_space_id(base) == embedding_space_id(same)
    assert embedding_space_id(base) != embedding_space_id(other)


# --- pipeline level ---------------------------------------------------------


def test_unchanged_inputs_come_from_cache_across_restart(tmp_path, monkeypatch):
    repo = _repo_with(tmp_path / "repo")
    calls: list[list[str]] = []
    _patch_embed(monkeypatch, calls)

    registry = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _handle, first = _refresh(registry, repo)
    assert first.counters["embedded"] >= 2
    assert first.counters["cache_hits"] == 0
    first_calls = len(calls)

    # a fresh service instance over the same index root: the cache survives
    registry.close()
    registry2 = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    (repo / "sample.py").write_text(
        "def first():\n    pass\n\n\ndef second():\n    pass\n\n\ndef third():\n    pass\n"
    )
    _handle2, second = _refresh(registry2, repo)
    assert second.counters["cache_hits"] >= 2
    assert second.counters["embedded"] >= 1
    # only the genuinely new input reached the model
    flat = [text for batch in calls[first_calls:] for text in batch]
    assert len(flat) == 1
    assert "third" in flat[0]
    registry2.close()


def test_rebuild_reuses_cached_vectors_without_inference(tmp_path, monkeypatch):
    repo = _repo_with(tmp_path / "repo")
    calls: list[list[str]] = []
    _patch_embed(monkeypatch, calls)

    registry = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _refresh(registry, repo)
    calls.clear()

    _handle2, rebuilt = _refresh(registry, repo, rebuild=True)
    assert rebuilt.counters["cache_hits"] >= 2
    assert calls == []
    registry.close()


def test_identical_inputs_are_embedded_once(tmp_path, monkeypatch):
    repo = init_repo(tmp_path / "repo")
    # one file with a duplicated definition: both captures share the exact
    # embed input (same path, qualname and body)
    (repo / "a.py").write_text("def same_body():\n    pass\n\n\ndef same_body():\n    pass\n")
    git(repo, "add", "a.py")
    git(repo, "commit", "-q", "-m", "init")

    calls: list[list[str]] = []
    _patch_embed(monkeypatch, calls)
    registry = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _handle, job = _refresh(registry, repo)
    flat = [text for batch in calls for text in batch]
    assert len(flat) == 1
    assert "same_body" in flat[0]
    assert job.counters["embedded"] == 2  # both symbols still get vectors
    registry.close()


def test_query_variants_keep_separate_cache_entries(tmp_path, monkeypatch):
    repo = _repo_with(tmp_path / "repo")
    query_calls: list[list[str]] = []

    def embed(texts, *, query=False):
        if query:
            query_calls.append(list(texts))
        return [VECTOR] * len(texts), None

    import priorart.models

    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: embed)

    registry = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _refresh(registry, repo)

    handle = registry.resolve(repo)
    # not an exact identifier: goes through the embedding stage
    first = handle.search("the first function", k=3)
    second = handle.search("the first function", k=3)
    assert first.candidates
    assert second.candidates
    # the same query text is embedded exactly once across both searches
    assert query_calls == [["the first function"]]
    registry.close()


def test_embedding_failure_keeps_cache_of_earlier_batches(tmp_path, monkeypatch):
    repo = init_repo(tmp_path / "repo")
    lines = [f"def fn_{index:03d}():\n    pass\n\n" for index in range(70)]
    (repo / "generated.py").write_text("".join(lines))
    git(repo, "add", "generated.py")
    git(repo, "commit", "-q", "-m", "init")

    state = {"calls": 0}

    def flaky(texts, *, query=False):
        state["calls"] += 1
        if state["calls"] > 1:
            return None, "endpoint down"
        return [VECTOR] * len(texts), None

    import priorart.models

    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: flaky)

    registry = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _handle, job = _refresh(registry, repo)
    # one batch of 64 succeeded, the rest failed: dense coverage is partial
    assert job.state in {"degraded", "completed"}
    registry.close()

    # restart with a working embedder: the successful batch must be served
    # from the cache, not recomputed
    calls: list[list[str]] = []
    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: _counting_embed(calls))
    registry2 = RuntimeRegistry(make_config(tmp_path, embed_model="test-model"))
    _handle2, second = _refresh(registry2, repo, rebuild=True)
    assert second.counters["cache_hits"] >= 64
    flat = [text for batch in calls for text in batch]
    assert len(flat) == 70 - second.counters["cache_hits"]
    registry2.close()
