"""Repository runtime registry.

A handle exists per canonical worktree root and index profile; the registry
lock protects only lookup/creation and never spans git, SQL, parsing or
inference. Model clients are shared across handles (one service profile),
and every reader call opens its own connection — there is no shared
mutable connection whose transaction one call could roll back for another.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from pathlib import Path

from .core import (
    HANDLE_CLOSED,
    INDEX_NOT_READY,
    JOB_NOT_FOUND,
    JOB_REPOSITORY_MISMATCH,
    Config,
    Job,
    PriorartError,
    resolve_repo,
)
from .indexing.jobs import JobManager, journal_job
from .models import ModelClients, embedding_space_id
from .models.embed_cache import EmbedCache
from .retrieval import (
    IndexSummary,
    map_symbols_rows,
    published_epoch,
    published_file_state,
    search,
    status_summary,
    status_text,
)
from .storage import (
    StoreProfile,
    embed_cache_path,
    ensure_worktree_dir,
    initialize_writer,
    known_roots,
    open_reader,
    profile_id,
    store_path,
    worktree_dir,
)


class RepoHandle:
    """One repository: its store, its jobs, its model collaborators."""

    def __init__(
        self,
        root: Path,
        config: Config,
        models: ModelClients,
        embed_cache: EmbedCache | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.models = models
        self.embed_cache = embed_cache
        self.embed_space_id = (
            embedding_space_id(config) if embed_cache is not None and models.dense_enabled else ""
        )
        self.profile = StoreProfile(
            embed_dim=config.embed_dim,
            embed_model=config.embed_model,
            embed_input_format=config.embed_input_format,
        )
        self.worktree_directory = worktree_dir(config.index_dir, self.root)
        self.store = store_path(
            self.worktree_directory,
            profile_id(config.embed_model, config.embed_dim, config.embed_input_format),
        )
        self._jobs: JobManager | None = None
        self._init_mutex = threading.Lock()
        self._watcher_stop = None
        self._closed = False

    def enable_auto_refresh(self, submit) -> None:
        """Start the background watcher if auto-freshness is configured."""
        with self._init_mutex:
            if self._watcher_stop is not None or self.config.watch_interval <= 0:
                return
            from .indexing.watcher import start_watcher

            self._watcher_stop = start_watcher(
                self.root,
                str(self.root),
                self._published_file_state,
                submit,
                interval=self.config.watch_interval,
                debounce=self.config.watch_debounce,
                content_interval=self.config.watch_content_interval,
            )

    def _published_file_state(self) -> dict[str, tuple[int, int, str]] | None:
        """File state of the last publication, or ``None`` when absent."""
        reader = open_reader(self.store, self.profile)
        if reader is None:
            return None
        try:
            if not published_epoch(reader, str(self.root)):
                return None
            return published_file_state(reader, str(self.root))
        finally:
            reader.close()

    # -- readers -----------------------------------------------------------

    def reader(self):
        """Open the published store for one call, or ``None`` when absent."""
        return open_reader(self.store, self.profile)

    def search(
        self, query: str, k: int = 10, mode: str = "balanced", intent: str = "implementation"
    ):
        _require_choice(mode, _MODES, "mode")
        _require_choice(intent, _INTENTS, "intent")
        reader = self.reader()
        if reader is None or not published_epoch(reader, str(self.root)):
            if reader is not None:
                reader.close()
            raise PriorartError(
                INDEX_NOT_READY,
                f"the index of {self.root} is not built yet; run refresh_index first",
                repo=str(self.root),
            )
        fast = mode == "fast"
        deep = mode == "deep"
        # balanced keeps generative expansion off by default: one raw query
        # embedding plus cheap lexical variants; deep enables the bounded
        # generative expansion and embeds all variants
        try:
            return search(
                reader,
                str(self.root),
                query,
                k=k,
                expand_fn=self.models.expand if deep else None,
                embed_fn=None if fast else self.models.embed,
                rerank_fn=None if fast else self.models.rerank,
                pool_expansion=self.config.pool_expansion,
                candidate_limit=self.config.candidate_limit,
                query_cache=self.embed_cache,
                query_space_id=self.embed_space_id,
                intent=intent,
                exact_dispatch=not deep,
                deadline_seconds=self.config.search_deadline_seconds,
                # fast mode skips models on purpose: no dense channel is
                # expected, so its absence is neither a warning nor degraded
                dense_expected=not fast,
            )
        finally:
            reader.close()

    def status(self) -> IndexSummary:
        reader = self.reader()
        if reader is None:
            return IndexSummary(repo=str(self.root), state="absent")
        try:
            return status_summary(reader, str(self.root), dense=self.models.dense_enabled)
        finally:
            reader.close()

    def status_text(self) -> str:
        reader = self.reader()
        if reader is None:
            return f"repo: {self.root}\nstate: absent\nrun refresh_index to build it"
        try:
            return status_text(reader, str(self.root), dense=self.models.dense_enabled)
        finally:
            reader.close()

    def map_symbols(self, path_glob: str, limit: int = 100, offset: int = 0):
        reader = self.reader()
        if reader is None:
            raise PriorartError(
                INDEX_NOT_READY,
                f"the index of {self.root} is not built yet; run refresh_index first",
                repo=str(self.root),
            )
        try:
            return map_symbols_rows(reader, str(self.root), path_glob, limit, offset)
        finally:
            reader.close()

    def index_metadata(self) -> dict:
        """Light published-state block for response envelopes."""
        summary = self.status()
        return summary.payload()

    # -- writer and jobs ---------------------------------------------------

    def refresh(self, *, rebuild: bool = False, paths: list[str] | None = None) -> Job:
        """Submit an indexing job; returns immediately with the job."""
        with self._init_mutex:
            if self._closed:
                raise PriorartError(
                    HANDLE_CLOSED, f"this priorart handle for {self.root} is closed"
                )
            if self._jobs is None:
                ensure_worktree_dir(self.config.index_dir, self.root)
                writer = initialize_writer(self.store, self.profile)
                try:
                    journal = initialize_writer(self.store, self.profile)
                except BaseException:
                    writer.close()
                    raise
                self._jobs = JobManager(
                    self.root,
                    journal_conn=journal,
                    writer_conn=writer,
                    embed_fn=self.models.embed,
                    lock_path=self.worktree_directory / "writer.lock",
                    embed_cache=self.embed_cache,
                    embed_space_id=self.embed_space_id,
                )
        return self._jobs.submit(rebuild=rebuild, paths=paths)

    def _journal_job(self, job_id: str) -> Job | None:
        """Job row from the on-disk journal (for restarted services)."""
        reader = self.reader()
        if reader is None:
            return None
        try:
            return journal_job(reader, job_id)
        finally:
            reader.close()

    def job(self, job_id: str) -> Job | None:
        manager = self._jobs
        return manager.get(job_id) if manager is not None else None

    def cancel_job(self, job_id: str) -> Job | None:
        manager = self._jobs
        return manager.cancel(job_id) if manager is not None else None

    def close(self) -> None:
        # snapshot under the mutex, act outside it: holding _init_mutex
        # across the 30s watcher join would deadlock a tick that is itself
        # blocked on the mutex inside refresh(), and the join would always
        # time out
        with self._init_mutex:
            stop, jobs = self._watcher_stop, self._jobs
            self._watcher_stop, self._jobs = None, None
            self._closed = True
        # the watcher must stop first: a late tick submitting into an
        # already-shut-down manager would strand the job at "queued"
        if stop is not None:
            stop()
        if jobs is not None:
            jobs.shutdown()


_MODES = ("fast", "balanced", "deep")
_INTENTS = ("implementation", "tests", "any")


def _require_choice(value: str, choices: tuple[str, ...], field: str) -> None:
    if value not in choices:
        raise ValueError(f"{field} must be one of {', '.join(choices)}; got {value!r}")


def _job_not_found(job_id: str) -> None:
    raise PriorartError(JOB_NOT_FOUND, f"no job {job_id} in this service.", job_id=job_id)


class RuntimeRegistry:
    """Resolves repositories to handles; never guesses the workspace."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        default_repos: Iterable[Path] = (),
        context_repos: Iterable[Path] = (),
    ) -> None:
        self.config = config or Config()
        self.models = ModelClients(self.config)
        self._defaults = [Path(repo) for repo in default_repos]
        self._context = [Path(repo) for repo in context_repos]
        self._handles: dict[Path, RepoHandle] = {}
        self._cache: EmbedCache | None = None
        self._job_owners: dict[str, RepoHandle] = {}
        self._mutex = threading.Lock()

    def resolve(self, repo: Path | str | None = None) -> RepoHandle:
        """Select exactly one repository for this call and return its handle."""
        resolved = resolve_repo(
            explicit=Path(repo) if repo is not None else None,
            context=self._context,
            defaults=self._defaults,
        )
        return self._handle(resolved.root)

    def _handle(self, root: Path) -> RepoHandle:
        with self._mutex:
            handle = self._handles.get(root)
            if handle is None:
                handle = RepoHandle(root, self.config, self.models, self._embed_cache())
                handle.enable_auto_refresh(
                    lambda paths, handle=handle: self.submit_refresh(handle, paths=paths)
                )
                self._handles[root] = handle
            return handle

    def _embed_cache(self) -> EmbedCache | None:
        """One shared cache file per service; opened lazily on first use."""
        if not self.models.dense_enabled:
            return None
        if self._cache is None:
            self._cache = EmbedCache(embed_cache_path(self.config.index_dir))
        return self._cache

    def submit_refresh(
        self, handle: RepoHandle, *, rebuild: bool = False, paths: list[str] | None = None
    ) -> Job:
        job = handle.refresh(rebuild=rebuild, paths=paths)
        with self._mutex:
            self._job_owners[job.job_id] = handle
        return job

    def get_job(self, job_id: str, repo: Path | str | None = None) -> tuple[RepoHandle, Job]:
        """Look up a job; an explicit repo must match the job's owner."""
        with self._mutex:
            handle = self._job_owners.get(job_id)
        if handle is None and repo is not None:
            # a restarted service has no in-memory owner: the journal row is
            # the honest answer (reconciled as interrupted when mid-run)
            fallback = self.resolve(repo)._journal_job(job_id)
            if fallback is not None:
                return self.resolve(repo), fallback
        if handle is None:
            _job_not_found(job_id)
        if repo is not None and str(Path(repo)) != str(handle.root):
            raise PriorartError(
                JOB_REPOSITORY_MISMATCH,
                f"job {job_id} belongs to {handle.root}, not {repo}.",
                job_id=job_id,
                repo=str(handle.root),
            )
        job = handle.job(job_id)
        if job is None:
            _job_not_found(job_id)
        return handle, job

    def cancel_job(self, job_id: str, repo: Path | str | None = None) -> Job:
        """Request cancellation of a job owned by this service."""
        handle, _job = self.get_job(job_id, repo)
        cancelled = handle.cancel_job(job_id)
        if cancelled is None:
            _job_not_found(job_id)
        return cancelled

    def workspaces(self) -> dict:
        """Workspace candidates by origin; never changes any selection."""
        return {
            "context": [str(Path(repo)) for repo in self._context],
            "configured": [str(Path(repo)) for repo in self._defaults],
            "known_indexed": [str(root) for root in known_roots(self.config.index_dir)],
        }

    def close(self) -> None:
        with self._mutex:
            handles = list(self._handles.values())
            cache, self._cache = self._cache, None
        for handle in handles:
            handle.close()
        if cache is not None:
            cache.close()
