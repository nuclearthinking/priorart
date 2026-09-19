import importlib.util
from pathlib import Path


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
    cases = [
        {
            "rank": benchmark.expected_rank(
                [
                    {"path": "src/example.py", "qualname": "Other.run"},
                    expected,
                ],
                expected,
            ),
            "latency_seconds": 1.0,
            "warnings": [],
        },
        {"rank": 12, "latency_seconds": 3.0, "warnings": ["fallback"]},
    ]

    assert cases[0]["rank"] == 2
    assert benchmark.summarize(cases, 10) == {
        "recall_at_10": 0.5,
        "mrr_at_10": 0.25,
        "query_total_seconds": 4.0,
        "latency_p50_seconds": 1.0,
        "latency_p95_seconds": 3.0,
        "queries_with_warnings": 1,
    }
