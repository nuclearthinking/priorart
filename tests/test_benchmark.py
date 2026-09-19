import importlib.util
import json
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
    assert benchmark.summarize(cases, k=10) == {
        "recall_at_10": 0.5,
        "mrr_at_10": 0.25,
        "query_total_seconds": 4.0,
        "latency_p50_seconds": 1.0,
        "latency_p95_seconds": 3.0,
        "queries_with_warnings": 1,
    }


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
