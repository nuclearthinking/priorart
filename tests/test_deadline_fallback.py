"""Search deadline budget and reranker-unavailable fallback (delivery F).

The end-to-end deadline is monotonic: expansion, embedding and reranking are
clamped to the remaining budget and degrade to warnings. Without a reranker
the pool is ranked by deterministic evidence with pool-expansion owners on
equal terms (R03) instead of tail-ordering at score 0.
"""

from __future__ import annotations

import time
from pathlib import Path

from priorart.registry import RuntimeRegistry
from priorart.retrieval import search
from tests.helpers import git, init_repo, make_config

SAMPLE = "def owner_handler():\n    pass\n"


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "src_app.py").write_text(SAMPLE)
    git(repo, "add", "src_app.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def _wait_jobs(registry: RuntimeRegistry, repo: Path):
    from priorart.core.jobs import FINAL_STATES

    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        _handle, current = registry.get_job(job.job_id)
        if current.state in FINAL_STATES:
            return handle
        time.sleep(0.01)
    raise AssertionError("job did not finish")


# --- deadline budget --------------------------------------------------------


def test_deadline_skips_dense_channel_with_warning(tmp_path, monkeypatch):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))

    def slow_embed(texts, *, query=False):
        time.sleep(0.05)
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    import priorart.models

    monkeypatch.setattr(priorart.models, "make_embedder", lambda config: slow_embed)
    handle = _wait_jobs(registry, repo)
    reader = handle.reader()
    try:
        report = search(
            reader,
            str(handle.root),
            "owner handler",
            k=5,
            embed_fn=slow_embed,
            deadline_seconds=0.001,
        )
    finally:
        reader.close()
    assert report.stages_used == ["lexical"]
    assert any("deadline exhausted" in warning for warning in report.warnings)


# --- R03 fallback ranking ---------------------------------------------------


def test_fallback_ranks_expansion_owners_with_fused_candidates(tmp_path):
    repo = init_repo(tmp_path / "repo")
    # one file with many symbols: the fused top is crowded by filler, the
    # owner of the relevant file enters the pool only via expansion
    (repo / "service.py").write_text(
        "def unrelated_alpha():\n    pass\n\n\ndef unrelated_beta():\n    pass\n\n\n"
        "def acceptor():\n    pass\n"
    )
    (repo / "canonicalize.py").write_text(
        "def canonicalize_owner():\n    pass\n\n\ndef other():\n    pass\n"
    )
    git(repo, "add", "service.py", "canonicalize.py")
    git(repo, "commit", "-q", "-m", "init")

    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = _wait_jobs(registry, repo)
        reader = handle.reader()
        try:
            report = search(
                reader,
                str(handle.root),
                "canonicalize owner",
                k=5,
                embed_fn=None,
                rerank_fn=None,
            )
        finally:
            reader.close()
        assert report.stages_used == ["lexical"]
        qualnames = [candidate.qualname for candidate in report.candidates]
        assert "canonicalize_owner" in qualnames
        # term evidence puts the owner above filler even though hybrid
        # order alone had expansion candidates at score 0
        assert qualnames[0] == "canonicalize_owner"
    finally:
        registry.close()


def test_fallback_survives_reranker_failure(tmp_path):
    repo = _repo_with(tmp_path / "repo")

    def broken_rerank(query, documents):
        return None, "reranker endpoint down"

    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = _wait_jobs(registry, repo)
        reader = handle.reader()
        try:
            report = search(reader, str(handle.root), "owner handler", k=5, rerank_fn=broken_rerank)
        finally:
            reader.close()
        assert report.candidates
        assert any("reranker endpoint down" in warning for warning in report.warnings)
        assert report.candidates[0].qualname == "owner_handler"
    finally:
        registry.close()
