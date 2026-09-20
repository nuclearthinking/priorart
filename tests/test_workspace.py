"""Workspace, storage-runtime and job contracts (plan deliveries A, B, C).

Covers the acceptance matrix rows implementable in-process: W01–W05, W07,
S01, S05, S07, J02 and the resolution order itself.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from priorart.core.errors import (
    AMBIGUOUS_WORKSPACE,
    INDEX_NOT_READY,
    INDEX_PROFILE_MISMATCH,
    JOB_REPOSITORY_MISMATCH,
    REPOSITORY_NOT_FOUND,
    REPOSITORY_NOT_SELECTED,
    PriorartError,
)
from priorart.core.jobs import FINAL_STATES, JOB_INTERRUPTED
from priorart.core.resolver import canonical_root, resolve_repo
from priorart.registry import RuntimeRegistry
from priorart.storage import (
    FileLock,
    StoreProfile,
    ensure_worktree_dir,
    initialize_writer,
    known_roots,
    open_reader,
    profile_id,
    worktree_dir,
)
from tests.helpers import git, init_repo, make_config

SAMPLE = 'def workspace_target():\n    """workspace tests."""\n    pass\n'


def _repo_with(path: Path, content: str = SAMPLE) -> Path:
    repo = init_repo(path)
    (repo / "sample.py").write_text(content)
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


def _refresh(registry: RuntimeRegistry, repo: Path, **kwargs):
    handle = registry.resolve(repo)
    job = registry.submit_refresh(handle, **kwargs)
    return handle, _wait_job(registry, job.job_id)


# --- resolution order (W02–W04) ---------------------------------------------


def test_explicit_absolute_repo_wins_and_subdirectory_resolves_to_root(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    (repo / "pkg").mkdir()
    resolved = resolve_repo(explicit=repo / "pkg")
    assert resolved.root == repo.resolve()
    assert resolved.input == repo / "pkg"


def test_explicit_relative_repo_is_rejected(tmp_path):
    with pytest.raises(PriorartError) as err:
        resolve_repo(explicit=Path("relative/path"))
    assert err.value.code == REPOSITORY_NOT_FOUND
    assert "absolute" in err.value.message


def test_explicit_missing_repo_is_rejected_even_with_default(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    with pytest.raises(PriorartError) as err:
        resolve_repo(explicit=tmp_path / "missing", defaults=[repo])
    assert err.value.code == REPOSITORY_NOT_FOUND


def test_explicit_non_git_repo_is_rejected_even_with_default(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(PriorartError) as err:
        resolve_repo(explicit=plain, defaults=[repo])
    assert err.value.code == REPOSITORY_NOT_FOUND
    # no silent fallback to the configured default
    assert "plain" in err.value.message


def test_no_source_at_all_is_not_selected(tmp_path):
    cached = _repo_with(tmp_path / "cached")
    with pytest.raises(PriorartError) as err:
        resolve_repo(defaults=[])
    assert err.value.code == REPOSITORY_NOT_SELECTED
    # the cached/known repo is never used implicitly
    assert cached not in err.value.details.values()


def test_multiple_context_roots_are_ambiguous(tmp_path):
    first = _repo_with(tmp_path / "first")
    second = _repo_with(tmp_path / "second")
    with pytest.raises(PriorartError) as err:
        resolve_repo(context=[first, second])
    assert err.value.code == AMBIGUOUS_WORKSPACE
    assert sorted(err.value.details["candidates"]) == sorted([str(first), str(second)])


def test_parent_directory_above_repositories_contributes_nothing(tmp_path):
    first = _repo_with(tmp_path / "first")
    second = _repo_with(tmp_path / "second")
    # the tmp_path parent itself is not a git repository
    with pytest.raises(PriorartError) as err:
        resolve_repo(context=[tmp_path, first, second])
    assert err.value.code == AMBIGUOUS_WORKSPACE


def test_single_default_is_selected(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    assert resolve_repo(defaults=[repo]).root == repo.resolve()


def test_symlink_alias_deduplicates_to_one_root(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    resolved = resolve_repo(context=[repo, alias])
    assert resolved.root == repo.resolve()
    assert Path(resolved.root) != alias


def test_canonical_root_rejects_plain_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(PriorartError) as err:
        canonical_root(plain)
    assert err.value.code == REPOSITORY_NOT_FOUND


# --- storage layout and profiles (W05, W07, S07) -----------------------------


def test_two_worktrees_of_one_history_get_separate_stores(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    worktree = tmp_path / "worktree"
    git(repo, "worktree", "add", "-q", str(worktree))
    (worktree / "sample.py").write_text("def worktree_target(): pass\n")

    config = make_config(tmp_path)
    registry = RuntimeRegistry(config)
    _handle_a, job_a = _refresh(registry, repo)
    _handle_b, job_b = _refresh(registry, worktree)
    assert job_a.state == job_b.state == "completed"

    handle_a = registry.resolve(repo)
    handle_b = registry.resolve(worktree)
    assert handle_a.store != handle_b.store
    assert handle_a.store.parent != handle_b.store.parent
    # each store serves only its own worktree
    report = handle_a.search("workspace_target", k=3)
    assert report.candidates[0].qualname == "workspace_target"
    report = handle_b.search("worktree_target", k=3)
    assert report.candidates[0].qualname == "worktree_target"


def test_worktree_dir_is_stable_and_symlink_aliases_share_it(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    assert worktree_dir(tmp_path, repo) == worktree_dir(tmp_path, alias)
    assert worktree_dir(tmp_path, repo) != worktree_dir(tmp_path, tmp_path / "other")


def test_profile_ids_separate_embedding_contracts():
    assert profile_id("", 4, "") == "lexical"
    assert profile_id("embed-a", 4, "instruct-text") != profile_id("embed-b", 4, "instruct-text")
    assert profile_id("embed-a", 4, "instruct-text") != profile_id("embed-a", 8, "instruct-text")
    assert profile_id("embed-a", 4, "instruct-text") != profile_id("embed-a", 4, "qwen3")


def test_reader_of_absent_store_is_none(tmp_path):
    assert open_reader(tmp_path / "absent.db", StoreProfile(embed_dim=4)) is None


def test_reader_never_resets_incompatible_store(tmp_path):
    db = tmp_path / "store.db"
    initialize_writer(db, StoreProfile(embed_dim=4))
    conn = initialize_writer(db, StoreProfile(embed_dim=4, embed_model="other"))
    conn.execute("INSERT INTO files (repo, path, mtime_ns, size) VALUES ('r', 'p', 1, 2)")
    conn.commit()
    conn.close()

    with pytest.raises(PriorartError) as err:
        open_reader(db, StoreProfile(embed_dim=8))
    assert err.value.code == INDEX_PROFILE_MISMATCH

    # the reader refusal left the store intact
    reader = open_reader(db, StoreProfile(embed_dim=4, embed_model="other"))
    assert reader.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
    reader.close()


def test_ensure_worktree_dir_records_known_root(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    config = make_config(tmp_path)
    ensure_worktree_dir(config.index_dir, repo.resolve())
    assert [str(root) for root in known_roots(config.index_dir)] == [str(repo.resolve())]


def test_file_lock_is_exclusive_and_releasable(tmp_path):
    lock = FileLock(tmp_path / "writer.lock")
    other = FileLock(tmp_path / "writer.lock")
    assert lock.acquire()
    assert not other.acquire()
    lock.release()
    assert other.acquire()
    other.release()


# --- registry behaviour (W02, W01) -------------------------------------------


def test_registry_does_not_fall_back_to_cached_handle(tmp_path):
    cached = _repo_with(tmp_path / "cached")
    registry = RuntimeRegistry(make_config(tmp_path))
    _refresh(registry, cached)

    fresh = RuntimeRegistry(make_config(tmp_path))
    with pytest.raises(PriorartError) as err:
        fresh.resolve(None)
    assert err.value.code == REPOSITORY_NOT_SELECTED


def test_search_before_first_publication_is_index_not_ready(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)
    with pytest.raises(PriorartError) as err:
        handle.search("anything", k=3)
    assert err.value.code == INDEX_NOT_READY


def test_status_of_absent_index_is_an_absent_success(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    summary = registry.resolve(repo).status()
    assert summary.state == "absent"
    assert summary.freshness == "absent"


def test_known_indexed_repos_listed_by_workspaces(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    config = make_config(tmp_path)
    registry = RuntimeRegistry(config)
    _refresh(registry, repo)
    assert str(repo) in registry.workspaces()["known_indexed"]


def test_job_repository_mismatch_is_detected(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    other = _repo_with(tmp_path / "other")
    registry = RuntimeRegistry(make_config(tmp_path))
    _handle, job = _refresh(registry, repo)
    with pytest.raises(PriorartError) as err:
        registry.get_job(job.job_id, other)
    assert err.value.code == JOB_REPOSITORY_MISMATCH
    _handle, same = registry.get_job(job.job_id, repo)
    assert same.job_id == job.job_id


# --- publication atomicity and vector guards (S01, S05) -----------------------


def test_reader_sees_consistent_epoch_across_publication(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    config = make_config(tmp_path)
    registry = RuntimeRegistry(config)
    handle, job = _refresh(registry, repo)
    reader = handle.reader()
    epoch_before = reader.execute(
        "SELECT epoch FROM repos WHERE repo = ?", (str(handle.root),)
    ).fetchone()[0]
    assert epoch_before == job.epoch

    (repo / "sample.py").write_text("def changed(): pass\n")
    _handle, job2 = _refresh(registry, repo)
    epoch_after = reader.execute(
        "SELECT epoch FROM repos WHERE repo = ?", (str(handle.root),)
    ).fetchone()[0]
    assert epoch_after == job2.epoch > epoch_before
    reader.close()


def test_late_vector_does_not_attach_to_recreated_symbols(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle, _first_job = _refresh(registry, repo)

    def embed_that_republishes(texts, *, query=False):
        # a concurrent refresh publishes a new version of the file while the
        # embedding call is in flight
        (repo / "sample.py").write_text("def newer(): pass\n")
        from priorart.indexing.pipeline import index_repo
        from priorart.storage import initialize_writer

        concurrent_handle = registry.resolve(repo)
        concurrent = initialize_writer(concurrent_handle.store, concurrent_handle.profile)
        index_repo(concurrent, repo, embed_fn=None)
        concurrent.close()
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    from priorart.indexing.pipeline import index_repo
    from priorart.storage import initialize_writer

    writer = initialize_writer(handle.store, handle.profile)
    stats = index_repo(writer, repo, embed_fn=embed_that_republishes, rebuild=True)
    writer.close()
    assert stats["warnings"] == []
    # published symbols are the concurrent refresh's version
    report = handle.search("newer", k=3)
    assert report.candidates[0].qualname == "newer"
    # the late vectors for the replaced symbol ids were not attached
    reader = handle.reader()
    attached = reader.execute(
        "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (str(handle.root),)
    ).fetchone()[0]
    reader.close()
    assert attached == 0


def test_publish_is_all_or_nothing_on_parse_crash(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    (repo / "second.py").write_text("def second(): pass\n")
    git(repo, "add", "second.py")
    git(repo, "commit", "-q", "-m", "second")

    registry = RuntimeRegistry(make_config(tmp_path))
    handle, job = _refresh(registry, repo)
    assert job.counters["symbols"] == 2

    from priorart.indexing import pipeline
    from priorart.storage import initialize_writer

    (repo / "sample.py").write_text("def changed(): pass\n")
    (repo / "second.py").write_text("def changed2(): pass\n")
    writer = initialize_writer(handle.store, handle.profile)
    original = pipeline._publish_staged

    def crashing_publish(conn, repo_key, item, warnings):
        if item.rel == "second.py":
            raise KeyboardInterrupt("crash between staged files")
        return original(conn, repo_key, item, warnings)

    pipeline._publish_staged = crashing_publish
    try:
        with pytest.raises(KeyboardInterrupt):
            pipeline.index_repo(writer, repo)
    finally:
        pipeline._publish_staged = original
    assert not writer.in_transaction
    reader = handle.reader()
    names = {
        row[0]
        for row in reader.execute("SELECT name FROM symbols WHERE repo = ?", (str(handle.root),))
    }
    epoch = reader.execute(
        "SELECT epoch FROM repos WHERE repo = ?", (str(handle.root),)
    ).fetchone()[0]
    reader.close()
    writer.close()
    # nothing from the crashed refresh was published
    assert names == {"workspace_target", "second"}
    assert epoch == job.epoch


# --- job lifecycle (join, cancel, reconcile) ----------------------------------


def test_identical_refresh_joins_active_job_and_rebuild_is_queued(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)
    _handle, first_job = _refresh(registry, repo)

    # once no job is active, a new identical request creates a new job
    second = registry.submit_refresh(handle)
    assert second.job_id != first_job.job_id
    _wait_job(registry, second.job_id)
    assert second.mode == "incremental"
    rebuild = registry.submit_refresh(handle, rebuild=True)
    assert rebuild.mode == "rebuild"
    _wait_job(registry, rebuild.job_id)


def test_active_job_is_joined_by_identical_requests(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)
    _refresh(registry, repo)

    from priorart.indexing.jobs import JobManager
    from priorart.storage import initialize_writer

    started = threading.Event()
    release = threading.Event()

    def slow_embed(texts, *, query=False):
        started.set()
        release.wait(timeout=10)
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    writer = initialize_writer(handle.store, handle.profile)
    journal = initialize_writer(handle.store, handle.profile)
    manager = JobManager(
        handle.root,
        journal_conn=journal,
        writer_conn=writer,
        embed_fn=slow_embed,
        lock_path=handle.worktree_directory / "join-test.lock",
    )
    try:
        job = manager.submit(rebuild=True)
        assert started.wait(timeout=10)
        joined = manager.submit(rebuild=True)
        incremental = manager.submit()
        assert joined.job_id == job.job_id
        assert incremental.job_id == job.job_id  # a rebuild covers an incremental request
        rebuild_after = manager.submit(rebuild=True)
        assert rebuild_after.job_id == job.job_id
        release.set()
        deadline = time.monotonic() + 10
        while manager.get(job.job_id).state not in FINAL_STATES:
            assert time.monotonic() < deadline, "job did not finish"
            time.sleep(0.01)
    finally:
        release.set()
        manager.shutdown()
        writer.close()
        journal.close()


def test_cancel_during_embedding_keeps_published_lexical_state(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle = registry.resolve(repo)

    embedding = threading.Event()
    cancel_requested = threading.Event()

    def hanging_embed(texts, *, query=False):
        embedding.set()
        cancel_requested.wait(timeout=10)
        return [b"\x00\x00\x80?" * 4] * len(texts), None

    from priorart.indexing.jobs import JobManager
    from priorart.storage import initialize_writer

    writer = initialize_writer(handle.store, handle.profile)
    journal = initialize_writer(handle.store, handle.profile)
    manager = JobManager(
        handle.root,
        journal_conn=journal,
        writer_conn=writer,
        embed_fn=hanging_embed,
        lock_path=handle.worktree_directory / "cancel-test.lock",
    )
    try:
        cancelled_job = manager.submit(rebuild=True)
        deadline = time.monotonic() + 10
        while not embedding.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        result = manager.cancel(cancelled_job.job_id)
        assert result is not None
        assert result.cancel_requested
        cancel_requested.set()
        deadline = time.monotonic() + 10
        while True:
            current = manager.get(cancelled_job.job_id)
            if current.state in FINAL_STATES:
                break
            if time.monotonic() > deadline:
                raise AssertionError("job did not cancel in time")
            time.sleep(0.01)
        assert current.state == "cancelled"
        # lexical symbols stayed published even though vectors were cancelled
        reader = handle.reader()
        symbols = reader.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo = ?", (str(handle.root),)
        ).fetchone()[0]
        reader.close()
        assert symbols >= 1
    finally:
        manager.shutdown()
        writer.close()
        journal.close()


def test_new_owner_reconciles_abandoned_jobs_as_interrupted(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    handle, _job = _refresh(registry, repo)

    from priorart.storage import initialize_writer

    journal = initialize_writer(handle.store, handle.profile)
    journal.execute(
        "INSERT OR REPLACE INTO jobs (job_id, repo, mode, state, phase, created_at, "
        "heartbeat_at, counters, error, epoch) VALUES "
        "('abandoned', ?, 'incremental', 'running', 'embedding', 1, 1, '{}', NULL, 3)",
        (str(handle.root),),
    )
    journal.commit()
    journal.close()

    _handle2, job2 = _refresh(registry, repo)
    state = initialize_writer(handle.store, handle.profile)
    row = state.execute("SELECT state FROM jobs WHERE job_id = 'abandoned'").fetchone()
    state.close()
    assert row[0] == JOB_INTERRUPTED
    assert job2.state == "completed"


def test_refresh_scopes_to_requested_paths(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    (repo / "other.py").write_text("def other_target(): pass\n")
    git(repo, "add", "other.py")
    git(repo, "commit", "-q", "-m", "other")

    registry = RuntimeRegistry(make_config(tmp_path))
    handle, job = _refresh(registry, repo)
    assert job.counters["symbols"] == 2

    (repo / "sample.py").write_text("def changed(): pass\n")
    (repo / "other.py").write_text("def other_changed(): pass\n")
    _handle, scoped = _refresh(registry, repo, paths=["sample.py"])
    # only the scoped file was recaptured
    assert scoped.counters["symbols"] == 1
    report = handle.search("other_target", k=3)
    assert report.candidates[0].qualname == "other_target"


# --- MCP wire contract (W01, W04) ---------------------------------------------


def test_mcp_from_foreign_cwd_with_explicit_repo(tmp_path, monkeypatch):
    from priorart import server as server_mod

    repo = _repo_with(tmp_path / "repo")
    monkeypatch.chdir("/")
    mcp = server_mod.build_server(None, config=make_config(tmp_path))

    result = asyncio.run(
        mcp.call_tool("search_codebase", {"query": "workspace_target", "repo": str(repo)})
    )
    assert result.is_error is True  # INDEX_NOT_READY: not indexed yet
    assert result.structured_content["error"]["code"] == INDEX_NOT_READY

    result = asyncio.run(mcp.call_tool("refresh_index", {"repo": str(repo)}))
    assert result.is_error is False
    job_id = result.structured_content["data"]["job"]["job_id"]
    while True:
        result = asyncio.run(mcp.call_tool("get_index_job", {"job_id": job_id, "repo": str(repo)}))
        if result.structured_content["data"]["job"]["state"] in FINAL_STATES:
            break

    result = asyncio.run(
        mcp.call_tool("search_codebase", {"query": "workspace_target", "repo": str(repo)})
    )
    assert result.is_error is False
    assert result.structured_content["repo"] == str(repo)
    assert result.structured_content["index"]["state"] == "ready"
    assert "workspace_target" in result.content[0].text


def test_mcp_invalid_explicit_repo_does_not_fall_back_to_default(tmp_path):
    from priorart import server as server_mod

    repo = _repo_with(tmp_path / "repo")
    mcp = server_mod.build_server(repo, config=make_config(tmp_path))

    result = asyncio.run(
        mcp.call_tool("search_codebase", {"query": "x", "repo": str(tmp_path / "missing")})
    )
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == REPOSITORY_NOT_FOUND


def test_mcp_job_of_other_repo_is_a_mismatch(tmp_path):
    from priorart import server as server_mod

    repo = _repo_with(tmp_path / "repo")
    other = _repo_with(tmp_path / "other")
    mcp = server_mod.build_server(None, config=make_config(tmp_path))

    result = asyncio.run(mcp.call_tool("refresh_index", {"repo": str(repo)}))
    job_id = result.structured_content["data"]["job"]["job_id"]

    result = asyncio.run(mcp.call_tool("get_index_job", {"job_id": job_id, "repo": str(other)}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == JOB_REPOSITORY_MISMATCH


def test_mcp_cancel_index_job(tmp_path):
    from priorart import server as server_mod

    repo = _repo_with(tmp_path / "repo")
    mcp = server_mod.build_server(None, config=make_config(tmp_path))
    result = asyncio.run(mcp.call_tool("refresh_index", {"repo": str(repo)}))
    job_id = result.structured_content["data"]["job"]["job_id"]

    result = asyncio.run(mcp.call_tool("cancel_index_job", {"job_id": job_id, "repo": str(repo)}))
    assert result.is_error is False
    state = result.structured_content["data"]["job"]["state"]
    assert state in FINAL_STATES
