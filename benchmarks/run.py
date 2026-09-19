from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from priorart.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


def expected_rank(results: list[dict], expected: dict) -> int | None:
    for rank, result in enumerate(results, 1):
        if (result["path"], result["qualname"]) == (
            expected["path"],
            expected["qualname"],
        ):
            return rank
    return None


def summarize(cases: list[dict], k: int) -> dict:
    ranks = [case["rank"] for case in cases if case["rank"] is not None and case["rank"] <= k]
    latencies = sorted(case["latency_seconds"] for case in cases)
    return {
        f"recall_at_{k}": len(ranks) / len(cases),
        f"mrr_at_{k}": sum(1 / rank for rank in ranks) / len(cases),
        "query_total_seconds": sum(latencies),
        "latency_p50_seconds": _percentile(latencies, 0.50),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "queries_with_warnings": sum(bool(case["warnings"]) for case in cases),
    }


def main() -> None:
    args = _parse_args()
    suite = json.loads(args.suite.read_text())
    repo = args.repo.resolve()
    _verify_snapshot(repo, suite["revision"])

    embed_model = os.environ.get("PRIORART_EMBED_MODEL") or "lexical"
    embed_dim = os.environ.get("PRIORART_EMBED_DIM", "1024")
    if "PRIORART_DB" not in os.environ:
        model_key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", embed_model)
        os.environ["PRIORART_DB"] = str(ROOT / ".bench" / f"index-{model_key}-{embed_dim}.db")

    runtime = Runtime(repo)
    index_started = time.perf_counter()
    index_stats = runtime.reindex(rebuild=args.rebuild)
    index_seconds = time.perf_counter() - index_started
    _verify_expected_symbols(runtime, suite["cases"])
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
            }
            for candidate in report.candidates
        ]
        rank = expected_rank(results, case["expected"])
        cases.append(
            {
                **case,
                "rank": rank,
                "latency_seconds": latency,
                "warnings": report.warnings,
                "results": results,
            }
        )
        outcome = f"HIT {rank}" if rank is not None and rank <= args.k else f"OUT {rank or '-'}"
        print(f"{outcome:>6}  {case['id']}  {latency:.2f}s", flush=True)

    created_at = datetime.now(UTC).isoformat()
    label = args.label or runtime.config.rerank_model
    result = {
        "suite": suite["name"],
        "revision": suite["revision"],
        "created_at": created_at,
        "label": label,
        "models": {
            "embedding": runtime.config.embed_model,
            "embedding_dimension": runtime.config.embed_dim,
            "reranker": runtime.config.rerank_model,
            "query_expansion": runtime.config.llm_model,
        },
        "k": args.k,
        "candidate_depth": args.candidate_depth,
        "index": index_stats,
        "index_seconds": index_seconds,
        "summary": summarize(cases, args.k),
        "cases": cases,
    }
    output = args.output or _default_output(suite["name"], label, created_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved: {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a priorart golden benchmark.")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--rebuild", action="store_true")
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


def _verify_expected_symbols(runtime: Runtime, cases: list[dict]) -> None:
    missing = []
    for case in cases:
        expected = case["expected"]
        row = runtime.conn.execute(
            "SELECT 1 FROM symbols WHERE repo = ? AND path = ? AND qualname = ?",
            (str(runtime.repo), expected["path"], expected["qualname"]),
        ).fetchone()
        if row is None:
            missing.append(case["id"])
    if missing:
        raise SystemExit(f"golden symbols missing from index: {', '.join(missing)}")


def _percentile(values: list[float], fraction: float) -> float:
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def _default_output(suite: str, label: str, created_at: str) -> Path:
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)
    timestamp = created_at.replace(":", "").replace("+", "-")
    return ROOT / ".bench" / "results" / f"{suite}__{safe_label}__{timestamp}.json"


if __name__ == "__main__":
    main()
