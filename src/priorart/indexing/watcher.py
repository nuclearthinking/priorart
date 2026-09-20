"""Background freshness watcher: polling invalidation with debounce.

The watcher is an accelerator, not a correctness mechanism: it notices new,
changed and deleted files and submits scoped incremental refreshes after a
quiet period, so the published index follows the working tree without a
manual refresh. Explicit refresh_index stays available for verification.

Detection is stat-based per poll; periodically the content hashes of files
whose size and mtime did not change are re-checked, catching edits that
preserved both (atomic-save tricks, deliberate mtime resets). Saved bytes
are indexed; editor buffers are out of scope.

Stability rule of the loop: stat drift is stable (a drifted file stays
drifted until a refresh publishes it), but content suspects are only
visible on reconciliation ticks. Suspects are therefore sticky between
reconciliations, and a re-edited file (new stat while the previous refresh
is still in flight) re-arms the debounce through its changed stat
signature.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from pathlib import Path

from priorart.core import FINAL_STATES as _FINAL_JOB_STATES
from priorart.core import JOB_COMPLETED, JOB_DEGRADED

from .inventory import InventoryError, list_source_files

StopFn = Callable[[], None]
Stat = tuple[int, int, int]

# failure backoff cap: bounded retries against a held writer lock or a
# failing submit path instead of one new job per watch interval
_RETRY_CAP = 300.0


def start_watcher(  # noqa: PLR0913 - explicit collaborator set
    root: Path,
    repo: str,
    open_state: Callable[[], dict[str, tuple[int, int, str]] | None],
    submit: Callable[[list[str] | None], object],
    *,
    interval: float,
    debounce: float,
    content_interval: float,
) -> StopFn:
    """Start one daemon watcher thread; the returned callable stops it.

    Stopping joins the thread: an in-flight tick must finish before the
    caller tears down what that tick might still be using.
    """
    stop = threading.Event()
    thread = threading.Thread(
        target=_loop,
        args=(Path(root), repo, open_state, submit, interval, debounce, content_interval, stop),
        daemon=True,
        name=f"priorart-watch-{Path(root).name}",
    )
    thread.start()

    def stop_and_join() -> None:
        stop.set()
        thread.join(timeout=30)

    return stop_and_join


def _loop(  # noqa: C901, PLR0913, PLR0915, PLR0917 - thread body takes its explicit collaborators
    root, repo, open_state, submit, interval, debounce, content_interval, stop
):
    seen_stat: dict[str, Stat] = {}
    sticky_suspects: set[str] = set()
    deleted_seen = False
    last_change = 0.0
    submitted = False
    last_job = None
    submitted_stat: dict[str, Stat] = {}
    submitted_paths: set[str] = set()
    unpublishable: set[str] = set()
    failures = 0
    retry_ready = 0.0
    last_content = time.monotonic()
    while not stop.wait(interval):
        now = time.monotonic()
        try:
            known = open_state()
        except Exception:  # noqa: BLE001, S112 - one bad tick must not kill the watcher
            # a profile mismatch or damaged store until the next refresh;
            # retry on the next tick
            continue
        if not known:
            # nothing published yet: the first explicit refresh decides
            continue
        try:
            inventory = list_source_files(root)
        except InventoryError:
            # transient git failure: retry on the next tick
            continue
        content_due = now - last_content >= content_interval
        drift, suspects, deleted = _scan(root, inventory, known, content_due=content_due)
        changed_now = {rel for rel, stat in drift.items() if seen_stat.get(rel) != stat}
        grew = bool(changed_now)
        if deleted and not deleted_seen:
            grew = True
        deleted_seen = deleted
        if content_due:
            last_content = now
            grew = grew or bool(set(suspects) - sticky_suspects)
            sticky_suspects = set(suspects)
        if grew:
            # a new edit arrived: the quiet period starts over; the edited
            # paths are publishable again and a fresh edit must not wait
            # out a backoff armed by older, unrelated failures
            last_change = now
            submitted = False
            unpublishable -= changed_now
            failures = 0
            retry_ready = 0.0
        seen_stat = drift
        if submitted and _job_finished(last_job):
            state = getattr(last_job, "state", None)
            if state in (JOB_COMPLETED, JOB_DEGRADED):
                # paths whose stat did not change since the submission were
                # processed and left unpublished (unparseable, unreadable):
                # resubmitting them every tick is an infinite loop
                unpublishable |= {
                    rel for rel, stat in submitted_stat.items() if seen_stat.get(rel) == stat
                }
                # the force-recapture of the submitted suspects happened;
                # the next content scan re-adds any that still mismatch
                sticky_suspects -= submitted_paths
                failures = 0
            else:
                failures += 1
            submitted = False
            retry_ready = _retry_ready_at(now, interval, failures)
        pending = (set(seen_stat) | sticky_suspects) - unpublishable
        if (
            (pending or deleted)
            and not submitted
            and now >= retry_ready
            and now - last_change >= debounce
        ):
            try:
                last_job = submit(None if deleted else sorted(pending))
            except Exception:  # noqa: BLE001 - the watcher must survive submit errors
                # submit failed: back off, then re-arm for a later tick
                last_job = None
                failures += 1
                retry_ready = _retry_ready_at(now, interval, failures)
                continue
            submitted = True
            submitted_stat = dict(seen_stat)
            submitted_paths = set(pending)


def _retry_ready_at(now: float, interval: float, failures: int) -> float:
    """Bounded exponential backoff after repeated failures."""
    return now + min(interval * 2**failures, _RETRY_CAP)


def _job_finished(last_job) -> bool:
    """True when the submitted job reached a final state (or is unobservable).

    The finished-with-unpublished-drift case re-arms the latch so the
    drift is retried instead of silently ignored until the next edit.
    """
    state = getattr(last_job, "state", None)
    return state is None or state in _FINAL_JOB_STATES


def _scan(
    root: Path, inventory: list[str], known: dict, *, content_due: bool
) -> tuple[dict[str, Stat], list[str], bool]:
    """Compare the working tree with the published file state.

    Returns (changed, content-suspects, deleted). Changed files differ by
    stat; suspects match the stat but not the content hash. The stat keeps
    ctime: metadata-only changes (permissions) must re-arm drift even when
    mtime and size are unchanged.
    """
    drift: dict[str, Stat] = {}
    suspects: list[str] = []
    for rel in inventory:
        captured = known.get(rel)
        try:
            st = (root / rel).stat()
        except OSError:
            drift[rel] = (-1, -1, -1)
            continue
        if captured is None or (captured[0], captured[1]) != (st.st_mtime_ns, st.st_size):
            drift[rel] = (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
        elif content_due:
            try:
                data = (root / rel).read_bytes()
            except OSError:
                # unreadable between stat and read: one bad tick must not
                # kill the watcher thread (no auto-refresh afterwards)
                drift[rel] = (st.st_mtime_ns, st.st_size, st.st_ctime_ns)
                continue
            if hashlib.sha256(data).hexdigest() != captured[2]:
                suspects.append(rel)
    deleted = bool(set(known) - set(inventory))
    return drift, suspects, deleted
