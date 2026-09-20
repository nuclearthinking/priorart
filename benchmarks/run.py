from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath

from priorart.core.config import Config
from priorart.indexing.capture import capture_file
from priorart.indexing.inventory import repo_languages
from priorart.indexing.parser import (
    EMBED_TEXT_FORMAT,
    LANGS,
    SEARCH_TEXT_FORMAT,
    parse_source,
    preflight_parsers,
)
from priorart.models.expand import EXPAND_PROMPT
from priorart.models.rerank import make_reranker
from priorart.registry import RepoHandle, RuntimeRegistry
from priorart.retrieval.search import (
    BODY_MAX_CHARS,
    EXPANSION_FILE_QUOTA,
    EXPANSION_LIMIT,
    RERANK_DOCUMENT_FORMAT,
    RRF_K,
    Candidate,
    rerank_document,
    rerank_positions,
)

ROOT = Path(__file__).resolve().parents[1]


class _Runner:
    """Synchronous benchmark wrapper over the registry/handle API."""

    def __init__(self, repo: Path):
        self.registry = RuntimeRegistry(Config())
        self.handle: RepoHandle = self.registry.resolve(Path(repo).resolve())
        self.conn = None

    @property
    def repo(self) -> Path:
        return self.handle.root

    @property
    def config(self) -> Config:
        return self.handle.config

    def reindex(self, *, rebuild: bool = False) -> dict:
        from priorart.core.jobs import FINAL_STATES

        job = self.registry.submit_refresh(self.handle, rebuild=rebuild)
        while job.state not in FINAL_STATES:
            time.sleep(0.05)
            job = self.registry.get_job(job.job_id)[1]
        if job.state not in {"completed", "degraded"}:
            raise SystemExit(f"indexing job {job.state}: {job.error}")
        if self.conn is not None:
            self.conn.close()
        self.conn = self.handle.reader()
        return {
            "files": job.counters.get("files", 0),
            "symbols": job.counters.get("symbols", 0),
            "removed": job.counters.get("removed", 0),
            "warnings": list(job.warnings),
            "index_epoch": job.epoch,
            "job_state": job.state,
        }

    def search(self, query: str, k: int):
        return self.handle.search(query, k=k)

    def symbol_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE repo = ?", (str(self.repo),)
        ).fetchone()[0]


LEGACY_DOCUMENT_FORMAT = "signature-docstring-v1"
LEGACY_LOCATOR_FORMAT = "path-qualname-kind-signature-docstring-v1"


def _document_signature_docstring(candidate: dict) -> str:
    return f"{candidate['signature']}\n{candidate['docstring']}"


def _document_locator_signature_docstring(candidate: dict) -> str:
    header = f"{candidate['path']} :: {candidate['qualname']} ({candidate['kind']})"
    return f"{header}\n{candidate['full_signature']}\n{candidate['docstring']}"


def _document_current_format(candidate: dict, max_chars: int) -> str:
    # The current format must stay byte-identical to the live pipeline: build
    # the document through the same rerank_document() the search uses.
    return rerank_document(
        Candidate(
            path=candidate["path"],
            name="",
            qualname=candidate["qualname"],
            kind=candidate["kind"],
            lang=candidate.get("lang", "python"),
            line=candidate.get("line", 0),
            end_line=candidate.get("end_line", 0),
            signature=candidate.get("signature", ""),
            full_signature=candidate.get("full_signature", ""),
            docstring=candidate.get("docstring", ""),
            body=candidate.get("body", ""),
            source_role=candidate.get("source_role", "production"),
            score=candidate.get("score", 0.0),
        ),
        body_max_chars=max_chars,
    )


DOCUMENT_FORMAT_IDS = (
    RERANK_DOCUMENT_FORMAT,
    LEGACY_LOCATOR_FORMAT,
    LEGACY_DOCUMENT_FORMAT,
)


def _document_builder(format_id: str, body_max_chars: int):
    if format_id == RERANK_DOCUMENT_FORMAT:
        return partial(_document_current_format, max_chars=body_max_chars)
    if format_id == LEGACY_LOCATOR_FORMAT:
        return _document_locator_signature_docstring
    if format_id == LEGACY_DOCUMENT_FORMAT:
        return _document_signature_docstring
    raise SystemExit(f"unknown rerank format: {format_id}")


def _obfuscation():
    path = Path(__file__).resolve().parent / "obfuscate.py"
    spec = importlib.util.spec_from_file_location("priorart_benchmark_obfuscate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def expected_rank(results: list[dict], expected: dict) -> int | None:
    for rank, result in enumerate(results, 1):
        if (result["path"], result["qualname"]) == (
            expected["path"],
            expected["qualname"],
        ):
            return rank
    return None


def path_rank(results: list[dict], expected: dict) -> int | None:
    for rank, result in enumerate(results, 1):
        if result["path"] == expected["path"]:
            return rank
    return None


def loss_stage(gold_id: int, trace, rank: int | None, k: int) -> str | None:
    if rank is not None and rank <= k:
        return None
    in_pool = gold_id in {symbol_id for symbol_id, _score in trace.fused} or gold_id in (
        trace.pool_expansion or ()
    )
    if in_pool:
        if rank is None:
            return "not_fetched"
        return "ranked_deep"
    if any(gold_id in ranking for ranking in (*trace.fts_rankings, *trace.vec_rankings)):
        return "pool_cutoff"
    return "not_retrieved"


def retrieved_by(gold_id: int, trace) -> list[str]:
    sources = []
    if any(gold_id in ranking for ranking in trace.fts_rankings):
        sources.append("fts")
    if any(gold_id in ranking for ranking in trace.vec_rankings):
        sources.append("dense")
    if gold_id in (trace.pool_expansion or ()):
        sources.append("expansion")
    return sources


def summarize(cases: list[dict], k: int) -> dict:
    ranks = [case["rank"] for case in cases if case["rank"] is not None and case["rank"] <= k]
    latencies = sorted(case["latency_seconds"] for case in cases)
    file_ranks = [
        case["path_rank"]
        for case in cases
        if case.get("path_rank") is not None and case["path_rank"] <= k
    ]
    losses: dict[str, int] = {}
    for case in cases:
        stage = case.get("loss_stage")
        if stage:
            losses[stage] = losses.get(stage, 0) + 1
    return {
        f"recall_at_{k}": len(ranks) / len(cases),
        f"mrr_at_{k}": sum(1 / rank for rank in ranks) / len(cases),
        f"file_presence_at_{k}": len(file_ranks) / len(cases),
        "loss_stages": losses,
        "query_total_seconds": sum(latencies),
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "queries_with_warnings": sum(bool(case["warnings"]) for case in cases),
    }


def priorart_revision() -> dict:
    try:

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"cannot determine priorart revision via git: {err}") from err


def main() -> None:
    args = _parse_args()
    result = _replay_artifact(args) if args.replay else _run_benchmark(args)
    output = args.output or _default_output(result["suite"], result["label"], result["created_at"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved: {output}")

    if args.publish:
        published = _publish(result, output, ROOT / "benchmarks" / "results")
        print(f"published: {published}")


def _replay_artifact(args) -> dict:
    _require_replay_args(args)
    source = json.loads(args.replay.read_text())
    config = Config()
    rerank_fn = make_reranker(config)
    if rerank_fn is None:
        raise SystemExit(
            "replay needs a rerank endpoint: set PRIORART_RERANK_BASE_URL and PRIORART_RERANK_MODEL"
        )
    return _run_replay(source, rerank_fn, config.rerank_model, args)


def _require_replay_args(args) -> None:
    if args.suite:
        raise SystemExit("--replay cannot be combined with --suite")
    if args.rebuild:
        raise SystemExit("--replay cannot be combined with --rebuild")
    if args.save_depth is not None:
        raise SystemExit("--replay cannot be combined with --save-depth")
    if not args.rerank_format:
        raise SystemExit("--replay needs --rerank-format")


def _require_benchmark_args(args) -> None:
    if not args.suite:
        raise SystemExit("--suite is required (or use --replay)")
    if not args.repo:
        raise SystemExit("--repo is required")
    if args.rerank_format:
        raise SystemExit("--rerank-format requires --replay")
    if args.body_from is not None:
        raise SystemExit("--body-from requires --replay")


def _run_case(
    runtime,
    case,
    gold_id,
    k,
    save_depth,
) -> dict:
    started = time.perf_counter()
    report = runtime.search(case["query"], k=k)
    latency = time.perf_counter() - started
    # The pool is the untruncated rerank input, exactly what an MCP search with
    # the same profile sees; k only limits what is reported to the caller.
    results = [
        {
            "path": candidate.path,
            "qualname": candidate.qualname,
            "kind": candidate.kind,
            "line": candidate.line,
            "score": candidate.score,
            "signature": candidate.signature,
            "full_signature": candidate.full_signature,
            "docstring": candidate.docstring,
            "body": candidate.body,
        }
        for candidate in report.pool or []
    ]
    rank = expected_rank(results, case["expected"])
    trace = report.trace
    outcome = f"HIT {rank}" if rank is not None and rank <= k else f"OUT {rank or '-'}"
    print(f"{outcome:>6}  {case['id']}  {latency:.2f}s", flush=True)
    return {
        **case,
        "rank": rank,
        "path_rank": path_rank(results, case["expected"]),
        "loss_stage": loss_stage(gold_id, trace, rank, k),
        "retrieved_by": retrieved_by(gold_id, trace),
        "latency_seconds": latency,
        "warnings": report.warnings,
        "results": results if save_depth is None else results[:save_depth],
        "trace": asdict(trace),
    }


def _run_benchmark(args) -> dict:
    _require_benchmark_args(args)
    save_depth = args.save_depth
    suite = json.loads(args.suite.read_text())
    if not suite.get("cases"):
        raise SystemExit("suite has no cases")
    if not suite.get("revision"):
        raise SystemExit("suite has no revision")
    repo = args.repo.resolve()
    _verify_snapshot(repo, suite["revision"])
    failures = preflight_parsers(repo_languages(repo))
    if failures:
        details = "; ".join(f"{lang}: {reason}" for lang, reason in failures.items())
        raise SystemExit(f"parser preflight failed: {details}")

    config = Config()
    if "PRIORART_INDEX_DIR" not in os.environ and "index_dir" not in config.model_fields_set:
        # The index store is only valid for one embedding profile: model,
        # dimension and input format together define the vector space, so all
        # three belong in the key. Otherwise an A/B over embed_input_format
        # would reuse vectors embedded with the other contract.
        model_key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", config.embed_model or "lexical")
        format_key = config.embed_input_format or "default"
        os.environ["PRIORART_INDEX_DIR"] = str(
            ROOT / ".bench" / f"indexes-{model_key}-{config.embed_dim}-{format_key}"
        )
    if args.no_pool_expansion:
        os.environ["PRIORART_POOL_EXPANSION"] = "false"

    runtime = _Runner(repo)
    index_started = time.perf_counter()
    index_stats = runtime.reindex(rebuild=args.rebuild)
    index_seconds = time.perf_counter() - index_started
    gold_ids = _verify_expected_symbols(runtime, suite["cases"])
    cases = [
        _run_case(runtime, case, gold_ids[case["id"]], args.k, save_depth)
        for case in suite["cases"]
    ]

    created_at = datetime.now(UTC).isoformat()
    label = args.label or runtime.config.rerank_model
    return {
        "suite": suite.get("name") or "unnamed",
        "revision": suite["revision"],
        "created_at": created_at,
        "label": label,
        "priorart": priorart_revision(),
        "models": {
            "embedding": runtime.config.embed_model,
            "embedding_dimension": runtime.config.embed_dim,
            "reranker": runtime.config.rerank_model,
            "query_expansion": runtime.config.llm_model,
        },
        "manifest": {
            "representation": {
                "rerank_document": RERANK_DOCUMENT_FORMAT,
                "embed_text": EMBED_TEXT_FORMAT,
                "search_text": SEARCH_TEXT_FORMAT,
                "candidate_limit": runtime.config.candidate_limit,
                "rrf_k": RRF_K,
                "pool_expansion": {
                    "enabled": runtime.config.pool_expansion,
                    "file_quota": EXPANSION_FILE_QUOTA,
                    "limit": EXPANSION_LIMIT,
                },
                "embedding_input_format": runtime.config.embed_input_format,
                "rerank_query_format": runtime.config.rerank_query_format,
            },
            "expansion_prompt_sha256": hashlib.sha256(EXPAND_PROMPT.encode()).hexdigest(),
        },
        "k": args.k,
        "save_depth": save_depth,
        "index": index_stats,
        "coverage": _coverage(runtime),
        "index_seconds": index_seconds,
        "summary": summarize(cases, args.k),
        "cases": cases,
    }


def _run_replay(source: dict, rerank_fn, rerank_model: str, args) -> dict:
    cases_in = _replay_source_cases(source)
    format_id = args.rerank_format
    if format_id not in DOCUMENT_FORMAT_IDS:
        raise SystemExit(f"unknown rerank format: {format_id}")
    if args.body_from is not None:
        _attach_bodies(source, args.body_from.resolve())
    if format_id == RERANK_DOCUMENT_FORMAT:
        _require_bodies(cases_in)
    build_document = _document_builder(format_id, args.body_chars)
    cases = []
    for case in cases_in:
        replayed = _replay_case(case, rerank_fn, build_document, args.k)
        cases.append(replayed)
        rank = replayed["rank"]
        outcome = f"HIT {rank}" if rank is not None and rank <= args.k else f"OUT {rank or '-'}"
        print(f"{outcome:>6}  {case['id']}  {replayed['latency_seconds']:.2f}s", flush=True)
    representation = {"rerank_document": format_id}
    if format_id == RERANK_DOCUMENT_FORMAT:
        representation["body_max_chars"] = args.body_chars
    # save_depth is None when the full pool was stored; report the actual size
    # so the manifest describes the real replay input.
    pool_size = source.get("save_depth") or max(
        (len(case.get("results") or []) for case in cases_in), default=0
    )
    manifest = {
        "representation": representation,
        "replay": {
            "source": args.replay.name,
            "source_label": source.get("label"),
            "source_created_at": source.get("created_at"),
            "pool_size": pool_size,
        },
    }
    if args.body_from is not None:
        manifest["body_from"] = {"repo": str(args.body_from), "revision": source.get("revision")}
    return {
        "suite": source.get("suite"),
        "revision": source.get("revision"),
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label or f"replay-{format_id}",
        "priorart": priorart_revision(),
        "models": {"reranker": rerank_model},
        "manifest": manifest,
        "k": args.k,
        "save_depth": source.get("save_depth"),
        "summary": summarize(cases, args.k),
        "cases": cases,
    }


def _replay_source_cases(source: dict) -> list[dict]:
    if source.get("provenance", {}).get("redacted"):
        raise SystemExit("cannot replay a redacted (published) artifact")
    cases_in = source.get("cases") or []
    if not cases_in:
        raise SystemExit("replay source has no cases")
    for case in cases_in:
        if any(key not in case for key in ("id", "query", "expected")):
            raise SystemExit(f"replay source case lacks id/query/expected: {case.get('id')}")
        for candidate in case.get("results") or []:
            missing = {
                "path",
                "qualname",
                "kind",
                "signature",
                "full_signature",
                "docstring",
            } - candidate.keys()
            if missing:
                raise SystemExit(
                    f"source pool lacks {sorted(missing)} for case {case['id']}; "
                    "re-run the benchmark with the current runner first"
                )
    return cases_in


def _parse_snapshot_file(
    repo: Path, rel: str
) -> tuple[list[str], dict[tuple[str, int], int], dict[str, set[int]]]:
    """Parse one snapshot file into (lines, (qualname, line) -> end_line, qualname -> lines)."""
    path = PurePosixPath(rel)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"cannot extract body: refusing path outside the snapshot: {rel}")
    captured = capture_file(repo, rel)
    if captured is None:
        raise SystemExit(f"cannot extract body: unreadable or changed while reading: {rel}")
    data, _stat = captured
    lang = LANGS.get(Path(rel).suffix)
    if lang is None:
        raise SystemExit(f"cannot extract body: unsupported file {rel}")
    result = parse_source(data, lang, rel)
    lines = data.decode("utf-8", "replace").split("\n")
    spans: dict[tuple[str, int], int] = {}
    symbol_lines: dict[str, set[int]] = {}
    for symbol in result.symbols:
        spans.setdefault((symbol.qualname, symbol.line), symbol.end_line)
        symbol_lines.setdefault(symbol.qualname, set()).add(symbol.line)
    return lines, spans, symbol_lines


def _attach_bodies(source: dict, repo: Path) -> None:
    """Attach each pool candidate's source span as `body`, parsed from the snapshot."""
    revision = source.get("revision")
    if not revision:
        raise SystemExit("cannot attach bodies: replay source has no revision")
    _verify_snapshot(repo, revision)
    parsed: dict[str, tuple[list[str], dict[tuple[str, int], int], dict[str, set[int]]]] = {}
    for case in source["cases"]:
        for candidate in case.get("results") or []:
            if candidate.get("body"):
                continue
            line = candidate.get("line")
            if line is None:
                raise SystemExit(
                    f"cannot extract body: {candidate['path']}::{candidate['qualname']} has no "
                    "line; re-run the benchmark with the current runner first"
                )
            entry = parsed.get(candidate["path"])
            if entry is None:
                entry = _parse_snapshot_file(repo, candidate["path"])
                parsed[candidate["path"]] = entry
            lines, spans, symbol_lines = entry
            end = spans.get((candidate["qualname"], line))
            if end is None:
                found = symbol_lines.get(candidate["qualname"])
                if found:
                    locations = ", ".join(str(at) for at in sorted(found))
                    raise SystemExit(
                        f"snapshot drift: {candidate['path']}::{candidate['qualname']} is not at "
                        f"line {line} (found at {locations})"
                    )
                raise SystemExit(
                    f"cannot extract body: {candidate['path']} has no symbol "
                    f"{candidate['qualname']}"
                )
            candidate["body"] = "\n".join(lines[line - 1 : end])


def _require_bodies(cases_in: list[dict]) -> None:
    missing = sum(
        1
        for case in cases_in
        for candidate in case.get("results") or []
        if not candidate.get("body")
    )
    if missing:
        raise SystemExit(
            f"body format needs --body-from or a pool with bodies ({missing} candidates lack body)"
        )


def _replay_case(case: dict, rerank_fn, build_document, k: int):
    pool = case.get("results") or []
    warnings: list[str] = []
    order = None
    started = time.perf_counter()
    if pool:
        order, warning = rerank_fn(case["query"], [build_document(candidate) for candidate in pool])
        if warning:
            warnings.append(warning)
    latency = time.perf_counter() - started
    # Without a valid rerank order the reported rank keeps the source run's
    # order — the source reranker's, not the hybrid one. Flag it so A/B
    # consumers can see that the new reranker was not actually applied.
    replay_fallback = bool(pool) and rerank_positions(order, len(pool)) is None
    ordered = _reordered(pool, order, warnings)
    rank = expected_rank(ordered, case["expected"])
    return {
        "id": case["id"],
        "query": case["query"],
        "expected": case["expected"],
        "source_rank": case.get("rank"),
        "rank": rank,
        "path_rank": path_rank(ordered, case["expected"]),
        "loss_stage": _replay_loss_stage(rank, k),
        "replay_fallback": replay_fallback,
        "latency_seconds": latency,
        "warnings": warnings,
        "results": ordered,
    }


def _reordered(pool: list[dict], order, warnings: list[str]) -> list[dict]:
    ordered = rerank_positions(order, len(pool))
    if ordered is not None:
        positions = dict(order)
        return [{**pool[idx], "score": positions[idx]} for idx in ordered]
    if order:
        warnings.append(
            f"rerank order invalid or incomplete ({len(order)}/{len(pool)} "
            "candidates); kept source order"
        )
    return pool


def _replay_loss_stage(rank: int | None, k: int) -> str | None:
    if rank is None:
        return "pool_absent"
    if rank > k:
        return "ranked_deep"
    return None


def _coverage(runtime: _Runner) -> dict:
    repo = str(runtime.repo)
    return {
        "parse": dict(
            runtime.conn.execute(
                "SELECT status, COUNT(*) FROM parse_state WHERE repo = ? GROUP BY status",
                (repo,),
            ).fetchall()
        ),
        "symbols": runtime.symbol_count(),
        "vectorized": runtime.conn.execute(
            "SELECT COUNT(*) FROM symbols_vec WHERE repo = ?", (repo,)
        ).fetchone()[0],
    }


def _publish(result: dict, output: Path, results_dir: Path, replacements_path: Path | None = None):
    obfuscate = _obfuscation()
    obfuscator = obfuscate.Obfuscator(
        obfuscate.load_replacements(replacements_path or obfuscate.DEFAULT_REPLACEMENTS_PATH)
    )
    published_result = obfuscator.value(result)
    obfuscate.assert_locator_uniqueness(result, published_result)
    revision = result.get("revision")
    if isinstance(revision, str) and revision and published_result.get("revision") == revision:
        raise ValueError(
            "suite revision was not replaced by the obfuscation vocabulary; "
            "add the corpus revision to the private replacements file"
        )
    published_result["provenance"] = {"redacted": True, "revision": "opaque-alias"}
    published = results_dir / obfuscator.text(output.name)
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps(published_result, ensure_ascii=False, indent=2) + "\n")
    return published


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a priorart golden benchmark.")
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--save-depth",
        type=int,
        default=None,
        help="Candidates persisted per case (default: the full rerank pool)",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay rerank over the pool saved in a benchmark artifact (no indexing)",
    )
    parser.add_argument(
        "--rerank-format",
        choices=DOCUMENT_FORMAT_IDS,
        help="Rerank document format for --replay",
    )
    parser.add_argument(
        "--body-from",
        type=Path,
        help="Repository snapshot to extract candidate bodies from (replay only)",
    )
    parser.add_argument(
        "--body-chars",
        type=int,
        default=BODY_MAX_CHARS,
        help=f"Max body characters per rerank document (default: {BODY_MAX_CHARS})",
    )
    parser.add_argument(
        "--no-pool-expansion",
        action="store_true",
        help="Disable file-to-owner pool expansion (control runs)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also save an obfuscated copy to benchmarks/results/ for the tracked history",
    )
    return parser.parse_args()


def _verify_snapshot(repo: Path, revision: str) -> None:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as err:
        detail = (err.stderr or err.stdout or "").strip()
        raise SystemExit(f"git failed on {repo}: {detail or err}") from err
    if head != revision:
        raise SystemExit(f"snapshot HEAD is {head}, expected {revision}")
    if dirty:
        raise SystemExit("snapshot has uncommitted changes")


def _verify_expected_symbols(runtime: _Runner, cases: list[dict]) -> dict[str, int]:
    counts = Counter(case["id"] for case in cases)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    if duplicates:
        raise SystemExit(f"duplicate case ids in suite: {', '.join(duplicates)}")
    missing = []
    gold_ids: dict[str, int] = {}
    for case in cases:
        expected = case["expected"]
        row = runtime.conn.execute(
            "SELECT id FROM symbols WHERE repo = ? AND path = ? AND qualname = ?",
            (str(runtime.repo), expected["path"], expected["qualname"]),
        ).fetchone()
        if row is None:
            missing.append(case["id"])
        else:
            gold_ids[case["id"]] = row[0]
    if missing:
        raise SystemExit(f"golden symbols missing from index: {', '.join(missing)}")
    return gold_ids


def _percentile(values: list[float], fraction: float) -> float:
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def _default_output(suite: str | None, label: str, created_at: str) -> Path:
    safe_suite = re.sub(r"[^a-zA-Z0-9_.-]+", "-", suite or "replay")
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)
    timestamp = created_at.replace(":", "").replace("+", "-")
    return ROOT / ".bench" / "results" / f"{safe_suite}__{safe_label}__{timestamp}.json"


if __name__ == "__main__":
    main()
