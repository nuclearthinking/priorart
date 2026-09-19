import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _benchmark_module():
    path = Path(__file__).parents[1] / "benchmarks" / "run.py"
    spec = importlib.util.spec_from_file_location("priorart_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_scores_exact_symbol_identity():
    benchmark = _benchmark_module()
    expected = {"path": "src/example.py", "qualname": "Example.run"}
    results = [
        {"path": "src/example.py", "qualname": "Other.run"},
        expected,
    ]
    assert benchmark.expected_rank(results, expected) == 2
    assert benchmark.path_rank(results, expected) == 1
    cases = [
        {
            "rank": 2,
            "path_rank": 1,
            "loss_stage": None,
            "latency_seconds": 1.0,
            "warnings": [],
        },
        {
            "rank": 12,
            "path_rank": None,
            "loss_stage": "not_retrieved",
            "latency_seconds": 3.0,
            "warnings": ["fallback"],
        },
    ]

    assert benchmark.summarize(cases, k=10) == {
        "recall_at_10": 0.5,
        "mrr_at_10": 0.25,
        "file_presence_at_10": 0.5,
        "loss_stages": {"not_retrieved": 1},
        "query_total_seconds": 4.0,
        "latency_p50_seconds": 1.0,
        "latency_p95_seconds": 3.0,
        "queries_with_warnings": 1,
    }


def _trace(*, fts=(99,), vec=(99,), fused=(99,)):
    from priorart.search import SearchTrace

    return SearchTrace(
        queries=["q"],
        stage_seconds={"expand": 0.0, "embed": 0.0, "retrieve": 0.0, "rerank": 0.0},
        fts_rankings=[list(fts)],
        vec_rankings=[list(vec)],
        fused=[(symbol_id, 0.01) for symbol_id in fused],
        rerank_order=None,
    )


def test_loss_stage_classifies_misses():
    benchmark = _benchmark_module()
    gold = 42

    in_pool = _trace(fts=(gold,), vec=(gold,), fused=(gold,))
    assert benchmark.loss_stage(gold, in_pool, rank=3, k=10) is None
    assert benchmark.loss_stage(gold, in_pool, rank=12, k=10) == "ranked_deep"
    assert benchmark.retrieved_by(gold, in_pool) == ["fts", "dense"]

    lexical_only = _trace(fts=(gold,), vec=(), fused=(gold,))
    assert benchmark.retrieved_by(gold, lexical_only) == ["fts"]

    cut_at_fusion = _trace(fts=(gold,), vec=(gold,), fused=(7,))
    assert benchmark.loss_stage(gold, cut_at_fusion, rank=12, k=10) == "pool_cutoff"

    never_retrieved = _trace(fts=(7,), vec=(7,), fused=(7,))
    assert benchmark.loss_stage(gold, never_retrieved, rank=None, k=10) == "not_retrieved"
    assert benchmark.retrieved_by(gold, never_retrieved) == []

    not_fetched = _trace(fts=(gold,), vec=(), fused=(gold,))
    assert benchmark.loss_stage(gold, not_fetched, rank=None, k=10) == "not_fetched"


def test_priorart_revision_records_sha_and_dirty_flag():
    benchmark = _benchmark_module()

    revision = benchmark.priorart_revision()

    assert len(revision["sha"]) == 40
    assert isinstance(revision["dirty"], bool)


def test_priorart_revision_fails_loudly_without_git(monkeypatch):
    benchmark = _benchmark_module()

    def failing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(benchmark.subprocess, "run", failing)
    with pytest.raises(SystemExit, match="cannot determine priorart revision"):
        benchmark.priorart_revision()


def _replay_args(**overrides):
    defaults = {
        "replay": Path("source.json"),
        "rerank_format": "signature-docstring-v1",
        "repo": None,
        "k": 10,
        "label": None,
        "suite": None,
        "rebuild": False,
        "save_depth": None,
        "body_from": None,
        "body_chars": 1200,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _source_artifact():
    return {
        "suite": "acme-golden",
        "revision": "abc123",
        "label": "remote-8b",
        "created_at": "2026-09-19T10:00:00+00:00",
        "save_depth": 2,
        "candidate_depth": 2,
        "cases": [
            {
                "id": "acme-1",
                "query": "find the runner",
                "expected": {"path": "a.py", "qualname": "run"},
                "rank": 2,
                "results": [
                    {
                        "path": "a.py",
                        "qualname": "walk",
                        "kind": "function",
                        "score": 0.9,
                        "signature": "def walk():",
                        "full_signature": "def walk():",
                        "docstring": "",
                    },
                    {
                        "path": "a.py",
                        "qualname": "run",
                        "kind": "function",
                        "score": 0.1,
                        "signature": "def run():",
                        "full_signature": "def run(\n    force: bool,\n):",
                        "docstring": "Run.",
                    },
                ],
            },
            {
                "id": "acme-2",
                "query": "find the missing",
                "expected": {"path": "z.py", "qualname": "gone"},
                "rank": None,
                "results": [
                    {
                        "path": "a.py",
                        "qualname": "walk",
                        "kind": "function",
                        "score": 0.9,
                        "signature": "def walk():",
                        "full_signature": "def walk():",
                        "docstring": "",
                    },
                ],
            },
        ],
    }


def _reverse_rerank(query, documents):
    return [(index, float(index)) for index in range(len(documents))], None


def test_run_replay_reranks_saved_pool_without_source():
    benchmark = _benchmark_module()

    result = benchmark._run_replay(
        _source_artifact(), _reverse_rerank, "reranker-x", _replay_args()
    )

    assert result["manifest"]["replay"]["source_label"] == "remote-8b"
    assert result["manifest"]["replay"]["pool_size"] == 2
    assert result["label"] == "replay-signature-docstring-v1"
    assert result["models"] == {"reranker": "reranker-x"}
    first, second = result["cases"]
    assert first["source_rank"] == 2
    assert first["rank"] == 1
    assert first["path_rank"] == 1
    assert first["loss_stage"] is None
    assert first["results"][0]["qualname"] == "run"
    assert first["results"][0]["score"] == 1.0
    assert first["warnings"] == []
    assert second["source_rank"] is None
    assert second["rank"] is None
    assert second["loss_stage"] == "pool_absent"
    assert result["summary"]["recall_at_10"] == 0.5
    assert result["summary"]["loss_stages"] == {"pool_absent": 1}


def test_replay_documents_match_format_specs():
    benchmark = _benchmark_module()
    candidate = {
        "path": "src/example.py",
        "qualname": "Example.run",
        "kind": "method",
        "signature": "def run(self):",
        "full_signature": "def run(self, force: bool):",
        "docstring": "Run the example.",
        "body": "def run(self, force: bool):\n    return force",
    }
    legacy = benchmark._document_builder("signature-docstring-v1", 1200)
    assert legacy(candidate) == "def run(self):\nRun the example."

    locator = benchmark._document_builder("path-qualname-kind-signature-docstring-v1", 1200)
    assert locator(candidate) == (
        "src/example.py :: Example.run (method)\ndef run(self, force: bool):\nRun the example."
    )

    body = benchmark._document_builder("path-qualname-kind-signature-docstring-body-v1", 1200)
    assert body(candidate) == (
        "src/example.py :: Example.run (method)\n"
        "def run(self, force: bool):\n"
        "Run the example.\n"
        "def run(self, force: bool):\n"
        "    return force"
    )


def test_bounded_body_truncates_on_line_boundary():
    benchmark = _benchmark_module()
    body = "\n".join(f"line {index}" for index in range(100))

    short = benchmark.bounded_body(body, 10_000)
    assert short == body

    truncated = benchmark.bounded_body(body, 60)
    kept, marker = truncated.rsplit("\n", 1)
    assert len(truncated) < 80
    assert marker.startswith("… (+")
    assert marker.endswith(" lines)")
    assert kept.splitlines()[-1] == "line 7"

    single_line = benchmark.bounded_body("x" * 500, 60)
    assert single_line == "x" * 60 + "\n…"

    assert benchmark.bounded_body("def f():\n    return 1", 0) == ""
    assert benchmark.bounded_body("def f():\n    return 1", -5) == ""


def test_replay_body_format_matches_search_rerank_document():
    benchmark = _benchmark_module()
    from priorart.search import Candidate, _rerank_document

    candidate = Candidate(
        path="src/example.py",
        name="run",
        qualname="Example.run",
        kind="method",
        lang="python",
        line=3,
        end_line=9,
        signature="def run(self):",
        full_signature="def run(self, force: bool):",
        docstring="Run the example.",
        body="def run(self, force: bool):\n    return force",
        score=0.1,
    )
    saved = {
        "path": "src/example.py",
        "qualname": "Example.run",
        "kind": "method",
        "full_signature": "def run(self, force: bool):",
        "docstring": "Run the example.",
        "body": "def run(self, force: bool):\n    return force",
    }

    from priorart.search import BODY_MAX_CHARS

    builder = benchmark._document_builder(
        "path-qualname-kind-signature-docstring-body-v1", BODY_MAX_CHARS
    )
    assert builder(saved) == _rerank_document(candidate)


def test_replay_reorders_pool_and_keeps_source_order_on_invalid():
    benchmark = _benchmark_module()
    pool = [
        {"path": "a.py", "qualname": "first", "score": 0.9},
        {"path": "b.py", "qualname": "second", "score": 0.1},
    ]

    warnings: list[str] = []
    ordered = benchmark._reordered(pool, [(1, 0.2), (0, 0.1)], warnings)
    assert [candidate["qualname"] for candidate in ordered] == ["second", "first"]
    assert [candidate["score"] for candidate in ordered] == [0.2, 0.1]
    assert warnings == []

    fallback = benchmark._reordered(pool, [(0, 0.5)], warnings)
    assert fallback == pool
    assert "kept source order" in warnings[-1]

    quiet = benchmark._reordered(pool, None, [])
    assert quiet == pool


def test_replay_loss_stage_classifies_pool_misses():
    benchmark = _benchmark_module()
    assert benchmark._replay_loss_stage(None, k=10) == "pool_absent"
    assert benchmark._replay_loss_stage(11, k=10) == "ranked_deep"
    assert benchmark._replay_loss_stage(3, k=10) is None


def test_replay_case_builds_documents_from_saved_pool():
    benchmark = _benchmark_module()
    case = _source_artifact()["cases"][0]
    captured = {}

    def rerank(query, documents):
        captured["documents"] = documents
        return [(index, float(index)) for index in range(len(documents))], None

    replayed = benchmark._replay_case(
        case, rerank, benchmark._document_locator_signature_docstring, 10
    )

    assert replayed["rank"] == 1
    assert (
        captured["documents"][1] == "a.py :: run (function)\ndef run(\n    force: bool,\n):\nRun."
    )


def test_run_replay_uses_saved_full_signatures_without_repo():
    benchmark = _benchmark_module()
    args = _replay_args(rerank_format="path-qualname-kind-signature-docstring-v1")

    result = benchmark._run_replay(_source_artifact(), _reverse_rerank, "reranker-x", args)

    first = result["cases"][0]
    assert first["rank"] == 1
    assert first["results"][0]["qualname"] == "run"


def test_run_replay_refuses_redacted_artifacts():
    benchmark = _benchmark_module()
    source = _source_artifact()
    source["provenance"] = {"redacted": True, "revision": "opaque-alias"}

    with pytest.raises(SystemExit, match="redacted"):
        benchmark._run_replay(source, _reverse_rerank, "reranker-x", _replay_args())


def test_run_replay_rejects_pools_without_saved_documents():
    benchmark = _benchmark_module()
    source = _source_artifact()
    source["cases"][0]["results"] = [
        {"path": "a.py", "qualname": "walk", "kind": "function", "score": 0.9}
    ]

    with pytest.raises(SystemExit, match="re-run the benchmark"):
        benchmark._run_replay(source, _reverse_rerank, "reranker-x", _replay_args())


def test_run_replay_body_format_requires_bodies():
    benchmark = _benchmark_module()
    args = _replay_args(rerank_format="path-qualname-kind-signature-docstring-body-v1")

    with pytest.raises(SystemExit, match="--body-from or a pool with bodies"):
        benchmark._run_replay(_source_artifact(), _reverse_rerank, "reranker-x", args)


def test_run_replay_body_format_ranks_from_pool_bodies():
    benchmark = _benchmark_module()
    source = _source_artifact()
    for case in source["cases"]:
        for candidate in case.get("results") or []:
            candidate["line"] = 1
            candidate["body"] = "def owner():\n    return 1"
    args = _replay_args(rerank_format="path-qualname-kind-signature-docstring-body-v1")

    result = benchmark._run_replay(source, _reverse_rerank, "reranker-x", args)

    assert result["manifest"]["representation"]["body_max_chars"] == 1200
    first = result["cases"][0]
    assert first["rank"] == 1
    assert first["results"][0]["body"] == "def owner():\n    return 1"


def test_attach_bodies_extracts_spans_from_snapshot(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def run(force):\n    return force\n\n\ndef walk():\n    return 1\n")
    source = _source_artifact()
    run_candidate = source["cases"][0]["results"][1]
    walk_candidate = source["cases"][0]["results"][0]
    run_candidate["line"] = 1
    walk_candidate["line"] = 5
    source["cases"][1]["results"][0]["line"] = 5

    benchmark._attach_bodies(source, repo)

    assert run_candidate["body"] == "def run(force):\n    return force"
    assert walk_candidate["body"] == "def walk():\n    return 1"
    assert source["cases"][1]["results"][0]["body"] == "def walk():\n    return 1"


def test_attach_bodies_fails_loudly_on_drift_and_missing_symbols(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def run(force):\n    return force\n\n\ndef walk():\n    return 1\n")

    drifted = _source_artifact()
    drifted["cases"][0]["results"][0]["line"] = 5
    drifted["cases"][0]["results"][1]["line"] = 9
    with pytest.raises(SystemExit, match="snapshot drift"):
        benchmark._attach_bodies(drifted, repo)

    absent = _source_artifact()
    absent["cases"][0]["results"][0]["line"] = 1
    absent["cases"][0]["results"][0]["qualname"] = "gone"
    with pytest.raises(SystemExit, match="has no symbol"):
        benchmark._attach_bodies(absent, repo)


def _body_source(path: str, qualname: str, line: int) -> dict:
    return {
        "suite": "acme-golden",
        "revision": "abc123",
        "save_depth": 1,
        "cases": [
            {
                "id": "acme-1",
                "query": "find the runner",
                "expected": {"path": path, "qualname": qualname},
                "results": [
                    {
                        "path": path,
                        "qualname": qualname,
                        "kind": "function",
                        "line": line,
                        "score": 0.5,
                        "signature": "def x():",
                        "full_signature": "def x():",
                        "docstring": "",
                    }
                ],
            }
        ],
    }


def test_attach_bodies_refuses_paths_outside_the_snapshot(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("def leak():\n    pass\n")

    for escape in ("../secret.py", str(secret), "/etc/passwd.py"):
        with pytest.raises(SystemExit, match="refusing path outside the snapshot"):
            benchmark._attach_bodies(_body_source(escape, "leak", 1), repo)


def test_attach_bodies_matches_duplicate_qualnames_by_line(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def run():\n    return 1\n\n\ndef run():\n    return 2\n")
    source = _body_source("a.py", "run", 5)

    benchmark._attach_bodies(source, repo)

    assert source["cases"][0]["results"][0]["body"] == "def run():\n    return 2"


def test_attach_bodies_splits_only_on_newlines(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_bytes(b"def run():\n    a = 1\x0c\n    b = 2\n    c = 3\n")
    source = _body_source("a.py", "run", 1)

    benchmark._attach_bodies(source, repo)

    assert source["cases"][0]["results"][0]["body"] == (
        "def run():\n    a = 1\x0c\n    b = 2\n    c = 3"
    )


def test_attach_bodies_requires_revision_and_candidate_lines(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(benchmark, "_verify_snapshot", lambda repo, revision: None)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def run():\n    return 1\n")

    no_revision = _body_source("a.py", "run", 1)
    no_revision["revision"] = None
    with pytest.raises(SystemExit, match="has no revision"):
        benchmark._attach_bodies(no_revision, repo)

    no_line = _body_source("a.py", "run", 1)
    del no_line["cases"][0]["results"][0]["line"]
    with pytest.raises(SystemExit, match="has no line"):
        benchmark._attach_bodies(no_line, repo)


def test_default_output_sanitizes_suite_and_label():
    benchmark = _benchmark_module()

    output = benchmark._default_output(
        "../../tmp/evil suite", "my experiment", "2026-09-20T10:00:00+00:00"
    )

    assert output.parent == benchmark.ROOT / ".bench" / "results"
    assert output.name.startswith("..-..-tmp-evil-suite__my-experiment__")
    fallback = benchmark._default_output(None, "x", "2026-09-20T10:00:00+00:00")
    assert fallback.name.startswith("replay__x__")


def test_run_benchmark_rejects_degenerate_suites(tmp_path):
    benchmark = _benchmark_module()
    base = argparse.Namespace(
        suite=None,
        repo=tmp_path,
        label=None,
        output=None,
        k=10,
        candidate_depth=50,
        save_depth=None,
        rebuild=False,
        rerank_format=None,
        body_from=None,
        body_chars=1200,
        publish=False,
    )

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"name": "s", "revision": "abc", "cases": []}))
    with pytest.raises(SystemExit, match="suite has no cases"):
        benchmark._run_benchmark(argparse.Namespace(**{**vars(base), "suite": empty}))

    no_revision = tmp_path / "norev.json"
    no_revision.write_text(json.dumps({"name": "s", "cases": [{"id": "x"}]}))
    with pytest.raises(SystemExit, match="suite has no revision"):
        benchmark._run_benchmark(argparse.Namespace(**{**vars(base), "suite": no_revision}))


def test_replay_artifact_wires_configured_reranker(tmp_path, monkeypatch):
    benchmark = _benchmark_module()
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_source_artifact()))
    captured = {}

    def rerank(query, documents):
        captured.setdefault("queries", []).append(query)
        return [(index, float(index)) for index in range(len(documents))], None

    class StubConfig:
        rerank_model = "reranker-x"

        def __init__(self):
            pass

    monkeypatch.setattr(benchmark, "Config", StubConfig)
    monkeypatch.setattr(benchmark, "make_reranker", lambda _config: rerank)

    result = benchmark._replay_artifact(_replay_args(replay=artifact))

    assert result["models"] == {"reranker": "reranker-x"}
    assert captured["queries"] == ["find the runner", "find the missing"]

    monkeypatch.setattr(benchmark, "make_reranker", lambda _config: None)
    with pytest.raises(SystemExit, match="replay needs a rerank endpoint"):
        benchmark._replay_artifact(_replay_args(replay=artifact))


def test_require_replay_args_rejects_conflicting_flags():
    benchmark = _benchmark_module()
    base = {"suite": None, "rebuild": False, "save_depth": None, "rerank_format": "f"}
    with pytest.raises(SystemExit, match="cannot be combined with --suite"):
        benchmark._require_replay_args(argparse.Namespace(**{**base, "suite": Path("s.json")}))
    with pytest.raises(SystemExit, match="cannot be combined with --rebuild"):
        benchmark._require_replay_args(argparse.Namespace(**{**base, "rebuild": True}))
    with pytest.raises(SystemExit, match="cannot be combined with --save-depth"):
        benchmark._require_replay_args(argparse.Namespace(**{**base, "save_depth": 5}))
    with pytest.raises(SystemExit, match="needs --rerank-format"):
        benchmark._require_replay_args(argparse.Namespace(**{**base, "rerank_format": None}))


def test_require_benchmark_args_rejects_missing_inputs():
    benchmark = _benchmark_module()
    base = {"suite": Path("s.json"), "repo": Path("r"), "rerank_format": None, "body_from": None}
    with pytest.raises(SystemExit, match="--suite is required"):
        benchmark._require_benchmark_args(argparse.Namespace(**{**base, "suite": None}))
    with pytest.raises(SystemExit, match="--repo is required"):
        benchmark._require_benchmark_args(argparse.Namespace(**{**base, "repo": None}))
    with pytest.raises(SystemExit, match="requires --replay"):
        benchmark._require_benchmark_args(argparse.Namespace(**{**base, "rerank_format": "f"}))
    with pytest.raises(SystemExit, match="--body-from requires --replay"):
        benchmark._require_benchmark_args(argparse.Namespace(**{**base, "body_from": Path("repo")}))


def test_parse_args_accepts_replay_mode(monkeypatch):
    benchmark = _benchmark_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--replay", "a.json", "--rerank-format", "signature-docstring-v1"],
    )

    args = benchmark._parse_args()

    assert args.replay == Path("a.json")
    assert args.rerank_format == "signature-docstring-v1"
    assert args.suite is None


def test_verify_expected_symbols_rejects_duplicate_case_ids():
    benchmark = _benchmark_module()
    cases = [
        {"id": "dup", "expected": {"path": "a.py", "qualname": "first"}},
        {"id": "dup", "expected": {"path": "a.py", "qualname": "second"}},
    ]

    with pytest.raises(SystemExit, match="duplicate case ids in suite: dup"):
        benchmark._verify_expected_symbols(object(), cases)


def test_assert_locator_uniqueness_detects_collisions(tmp_path):
    obfuscate, obfuscator = _obfuscator(tmp_path)
    original = {
        "cases": [
            {
                "id": "acme-1",
                "results": [
                    {"path": "src/acme/x.py", "qualname": "run"},
                    {"path": "src/zeta/x.py", "qualname": "run"},
                ],
            }
        ]
    }

    published = obfuscator.value(original)

    with pytest.raises(ValueError, match="collided locators"):
        obfuscate.assert_locator_uniqueness(original, published)


def test_assert_locator_uniqueness_passes_without_collisions(tmp_path):
    obfuscate, obfuscator = _obfuscator(tmp_path)
    original = {
        "cases": [
            {
                "id": "acme-1",
                "results": [
                    {"path": "src/acme/x.py", "qualname": "run"},
                    {"path": "src/acme/y.py", "qualname": "walk"},
                ],
            }
        ]
    }

    published = obfuscator.value(original)

    obfuscate.assert_locator_uniqueness(original, published)


def test_assert_locator_uniqueness_rejects_changed_case_count():
    obfuscate = _obfuscate_module()
    original = {"cases": [{"id": 1, "results": []}]}
    published = {"cases": []}

    with pytest.raises(ValueError, match="changed case count"):
        obfuscate.assert_locator_uniqueness(original, published)


def test_publish_marks_provenance_and_checks_locators(tmp_path):
    benchmark = _benchmark_module()
    result = {
        "cases": [
            {
                "id": "acme-1",
                "results": [{"path": "src/acme/x.py", "qualname": "run"}],
            }
        ]
    }
    replacements = tmp_path / "replacements.json"
    replacements.write_text(json.dumps({"acme": "zeta"}))
    output = tmp_path / "orig.json"
    out_dir = tmp_path / "published"

    published = benchmark._publish(result, output, out_dir, replacements)

    assert published == out_dir / "orig.json"
    data = json.loads(published.read_text())
    assert data["provenance"] == {"redacted": True, "revision": "opaque-alias"}
    assert data["cases"][0]["results"][0]["path"] == "src/zeta/x.py"


def _obfuscate_module():
    path = Path(__file__).parents[1] / "benchmarks" / "obfuscate.py"
    spec = importlib.util.spec_from_file_location("priorart_benchmark_obfuscate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _obfuscator(tmp_path):
    obfuscate = _obfuscate_module()
    replacements = tmp_path / "replacements.json"
    replacements.write_text(json.dumps({"acme": "zeta", "orbis": "nova"}))
    return obfuscate, obfuscate.Obfuscator(obfuscate.load_replacements(replacements))


def test_obfuscate_replaces_terms_with_case_preservation(tmp_path):
    _, obfuscator = _obfuscator(tmp_path)

    assert obfuscator.text("src/acme/daemon/worker.py") == "src/zeta/daemon/worker.py"
    assert obfuscator.text("acme-122-stop-repeated-repair") == "zeta-122-stop-repeated-repair"
    assert obfuscator.text("Ask Orbis Jira") == "Ask Nova Jira"
    assert obfuscator.text("ACME suite") == "ZETA suite"


def test_obfuscate_replaces_compound_identifiers(tmp_path):
    _, obfuscator = _obfuscator(tmp_path)

    assert obfuscator.text("ACME_BASE_URL is not set") == "ZETA_BASE_URL is not set"
    assert obfuscator.text("robot/acme_config.yaml") == "robot/zeta_config.yaml"


def test_obfuscate_keeps_similar_words_untouched(tmp_path):
    _, obfuscator = _obfuscator(tmp_path)
    text = "acmetry acmeology orbisync selectivity"

    assert obfuscator.text(text) == text


def test_obfuscate_walks_nested_structures_and_keys(tmp_path):
    _, obfuscator = _obfuscator(tmp_path)
    data = {"acme-master-v1": [{"path": "src/acme/x.py", "nested": {"query": "Orbis"}, "n": 1}]}

    assert obfuscator.value(data) == {
        "zeta-master-v1": [{"path": "src/zeta/x.py", "nested": {"query": "Nova"}, "n": 1}]
    }


def test_obfuscate_redacts_provenance_keys(tmp_path):
    obfuscate, obfuscator = _obfuscator(tmp_path)
    data = {"id": "acme-1", "issue_url": "https://git.example.org/ai/acme/issues/1"}

    result = obfuscator.value(data)

    assert result == {"id": "zeta-1"}
    assert "issue_url" in obfuscate.REDACTED_KEYS


def test_load_replacements_rejects_bad_vocabulary(tmp_path):
    obfuscate = _obfuscate_module()
    empty = tmp_path / "empty.json"
    empty.write_text("{}")

    with pytest.raises(ValueError, match="non-empty"):
        obfuscate.load_replacements(empty)
    with pytest.raises(FileNotFoundError):
        obfuscate.load_replacements(tmp_path / "missing.json")
