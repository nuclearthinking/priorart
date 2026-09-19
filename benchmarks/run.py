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
from pathlib import Path

from priorart.config import Config
from priorart.expand import EXPAND_PROMPT
from priorart.indexer import preflight_parsers, repo_languages
from priorart.rerank import make_reranker
from priorart.runtime import Runtime
from priorart.search import CANDIDATE_LIMIT, RERANK_DOCUMENT_FORMAT, RRF_K, _valid_rerank

ROOT = Path(__file__).resolve().parents[1]

LEGACY_DOCUMENT_FORMAT = "signature-docstring-v1"


def _document_signature_docstring(candidate: dict) -> str:
    return f"{candidate['signature']}\n{candidate['docstring']}"


def _document_locator_signature_docstring(candidate: dict) -> str:
    header = f"{candidate['path']} :: {candidate['qualname']} ({candidate['kind']})"
    return f"{header}\n{candidate['full_signature']}\n{candidate['docstring']}"


RERANK_DOCUMENT_BUILDERS = {
    RERANK_DOCUMENT_FORMAT: _document_locator_signature_docstring,
    LEGACY_DOCUMENT_FORMAT: _document_signature_docstring,
}


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
    retrieved = any(gold_id in ranking for ranking in (*trace.fts_rankings, *trace.vec_rankings))
    if not retrieved:
        return "not_retrieved"
    if gold_id not in {symbol_id for symbol_id, _score in trace.fused}:
        return "pool_cutoff"
    if rank is None:
        return "not_fetched"
    return "ranked_deep"


def retrieved_by(gold_id: int, trace) -> list[str]:
    sources = []
    if any(gold_id in ranking for ranking in trace.fts_rankings):
        sources.append("fts")
    if any(gold_id in ranking for ranking in trace.vec_rankings):
        sources.append("dense")
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


def _run_benchmark(args) -> dict:
    _require_benchmark_args(args)
    save_depth = args.save_depth if args.save_depth is not None else args.candidate_depth
    suite = json.loads(args.suite.read_text())
    repo = args.repo.resolve()
    _verify_snapshot(repo, suite["revision"])
    failures = preflight_parsers(repo_languages(repo))
    if failures:
        details = "; ".join(f"{lang}: {reason}" for lang, reason in failures.items())
        raise SystemExit(f"parser preflight failed: {details}")

    embed_model = os.environ.get("PRIORART_EMBED_MODEL") or "lexical"
    embed_dim = os.environ.get("PRIORART_EMBED_DIM", "1024")
    if "PRIORART_DB" not in os.environ:
        model_key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", embed_model)
        os.environ["PRIORART_DB"] = str(ROOT / ".bench" / f"index-{model_key}-{embed_dim}.db")

    runtime = Runtime(repo)
    index_started = time.perf_counter()
    index_stats = runtime.reindex(rebuild=args.rebuild)
    index_seconds = time.perf_counter() - index_started
    gold_ids = _verify_expected_symbols(runtime, suite["cases"])
    cases = []
    for case in suite["cases"]:
        started = time.perf_counter()
        report = runtime.search(
            case["query"],
            k=max(args.k, args.candidate_depth),
        )
        latency = time.perf_counter() - started
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
            }
            for candidate in report.candidates
        ]
        rank = expected_rank(results, case["expected"])
        gold_id = gold_ids[case["id"]]
        trace = report.trace
        cases.append(
            {
                **case,
                "rank": rank,
                "path_rank": path_rank(results, case["expected"]),
                "loss_stage": loss_stage(gold_id, trace, rank, args.k),
                "retrieved_by": retrieved_by(gold_id, trace),
                "latency_seconds": latency,
                "warnings": report.warnings,
                "results": results[:save_depth],
                "trace": asdict(trace),
            }
        )
        outcome = f"HIT {rank}" if rank is not None and rank <= args.k else f"OUT {rank or '-'}"
        print(f"{outcome:>6}  {case['id']}  {latency:.2f}s", flush=True)

    created_at = datetime.now(UTC).isoformat()
    label = args.label or runtime.config.rerank_model
    return {
        "suite": suite["name"],
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
                "candidate_limit": CANDIDATE_LIMIT,
                "rrf_k": RRF_K,
            },
            "expansion_prompt_sha256": hashlib.sha256(EXPAND_PROMPT.encode()).hexdigest(),
        },
        "k": args.k,
        "candidate_depth": args.candidate_depth,
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
    if format_id not in RERANK_DOCUMENT_BUILDERS:
        raise SystemExit(f"unknown rerank format: {format_id}")
    build_document = RERANK_DOCUMENT_BUILDERS[format_id]
    cases = []
    for case in cases_in:
        replayed = _replay_case(case, rerank_fn, build_document, args.k)
        cases.append(replayed)
        rank = replayed["rank"]
        outcome = f"HIT {rank}" if rank is not None and rank <= args.k else f"OUT {rank or '-'}"
        print(f"{outcome:>6}  {case['id']}  {replayed['latency_seconds']:.2f}s", flush=True)
    return {
        "suite": source.get("suite"),
        "revision": source.get("revision"),
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label or f"replay-{format_id}",
        "priorart": priorart_revision(),
        "models": {"reranker": rerank_model},
        "manifest": {
            "representation": {"rerank_document": format_id},
            "replay": {
                "source": args.replay.name,
                "source_label": source.get("label"),
                "source_created_at": source.get("created_at"),
                "pool_size": source.get("save_depth"),
            },
        },
        "k": args.k,
        "candidate_depth": source.get("candidate_depth"),
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
        "latency_seconds": latency,
        "warnings": warnings,
        "results": ordered,
    }


def _reordered(pool: list[dict], order, warnings: list[str]) -> list[dict]:
    if _valid_rerank(order, len(pool)):
        positions = dict(order)
        ordered = sorted(positions, key=lambda idx: positions[idx], reverse=True)
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


def _coverage(runtime: Runtime) -> dict:
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
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument(
        "--save-depth",
        type=int,
        default=None,
        help="Candidates persisted per case (default: candidate depth)",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay rerank over the pool saved in a benchmark artifact (no indexing)",
    )
    parser.add_argument(
        "--rerank-format",
        choices=sorted(RERANK_DOCUMENT_BUILDERS),
        help="Rerank document format for --replay",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also save an obfuscated copy to benchmarks/results/ for the tracked history",
    )
    return parser.parse_args()


def _verify_snapshot(repo: Path, revision: str) -> None:
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
    if head != revision:
        raise SystemExit(f"snapshot HEAD is {head}, expected {revision}")
    if dirty:
        raise SystemExit("snapshot has uncommitted changes")


def _verify_expected_symbols(runtime: Runtime, cases: list[dict]) -> dict[str, int]:
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


def _default_output(suite: str, label: str, created_at: str) -> Path:
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)
    timestamp = created_at.replace(":", "").replace("+", "-")
    return ROOT / ".bench" / "results" / f"{suite}__{safe_label}__{timestamp}.json"


if __name__ == "__main__":
    main()
