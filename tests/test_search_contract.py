"""Request contracts for the discovery surface: one search vocabulary,
aggregate validation parity, and the map_symbols page guard.

The same ``INVALID_ARGUMENT`` with the same ordered ``violations`` must cross
every path an agent can take — direct validation, the embedded registry, the
daemon wire and the MCP boundary — so one bad call costs one repair round.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from priorart.core import (
    SEARCH_INTENTS,
    SEARCH_MODES,
    PriorartError,
    validate_map_request,
    validate_search_request,
)
from tests.helpers import DaemonFixture, git, init_repo, make_config

EXPECTED_VIOLATIONS = [
    {"field": "k", "input": 0, "accepted": ["integer >= 1"]},
    {"field": "mode", "input": "hybrid", "accepted": ["fast", "balanced", "deep"]},
    {"field": "intent", "input": "find both", "accepted": ["implementation", "tests", "any"]},
]

SAMPLE = "def contract_target():\n    pass\n"


def _repo_with(path: Path) -> Path:
    repo = init_repo(path)
    (repo / "app.py").write_text(SAMPLE)
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "init")
    return repo


# --- vocabulary and validation -------------------------------------------------


def test_vocabulary_is_the_single_source_of_truth():
    assert SEARCH_MODES == ("fast", "balanced", "deep")
    assert SEARCH_INTENTS == ("implementation", "tests", "any")


@pytest.mark.parametrize("mode", SEARCH_MODES)
@pytest.mark.parametrize("intent", SEARCH_INTENTS)
def test_every_vocabulary_value_is_valid(mode, intent):
    validate_search_request(1, mode, intent)


def test_validation_aggregates_every_violation_in_fixed_order():
    with pytest.raises(PriorartError) as err:
        validate_search_request(0, "hybrid", "find both")
    payload = err.value.payload()
    assert payload["code"] == "INVALID_ARGUMENT"
    assert payload["violations"] == EXPECTED_VIOLATIONS
    assert payload["next_action"] == "Correct every listed search argument and retry the call."


def test_validation_reports_each_single_field_alone():
    with pytest.raises(PriorartError) as err:
        validate_search_request(10, "balanced", "everything")
    assert err.value.details["violations"] == [
        {"field": "intent", "input": "everything", "accepted": list(SEARCH_INTENTS)}
    ]


def test_validation_rejects_non_integer_k():
    with pytest.raises(PriorartError) as err:
        validate_search_request("10", "balanced", "any")
    assert err.value.details["violations"] == [
        {"field": "k", "input": "10", "accepted": ["integer >= 1"]}
    ]


def test_validation_rejects_boolean_k():
    # a JSON `true` coerces to 1 through the schema layer; the domain
    # guard is where it must stop
    with pytest.raises(PriorartError) as err:
        validate_search_request(k=True, mode="balanced", intent="any")
    assert err.value.details["violations"] == [
        {"field": "k", "input": True, "accepted": ["integer >= 1"]}
    ]


# --- map_symbols request contract -----------------------------------------------


@pytest.mark.parametrize("limit", [0, -1, True, "10", 1.5])
def test_map_request_rejects_non_positive_limit(limit):
    with pytest.raises(PriorartError) as err:
        validate_map_request(limit)
    assert err.value.code == "INVALID_ARGUMENT"
    assert err.value.details["violations"] == [
        {"field": "limit", "input": limit, "accepted": ["integer >= 1"]}
    ]
    assert err.value.payload()["next_action"] == "Correct the listed argument and retry the call."


@pytest.mark.parametrize("limit", [1, 100])
def test_map_request_accepts_positive_integer_limits(limit):
    validate_map_request(limit)


def test_map_limit_guard_precedes_index_readiness(tmp_path):
    from priorart.registry import RuntimeRegistry

    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        # a zero-row page is indistinguishable from an exactly full one:
        # limit=0 would paginate forever, so it must fail before any read
        with pytest.raises(PriorartError) as err:
            handle.map_symbols("*", limit=0)
        assert err.value.code == "INVALID_ARGUMENT"
        assert err.value.details["violations"] == [
            {"field": "limit", "input": 0, "accepted": ["integer >= 1"]}
        ]
    finally:
        registry.close()


def test_mcp_map_limit_guard_is_structured(tmp_path):
    from priorart import server as server_mod

    repo = _repo_with(tmp_path / "repo")
    mcp = server_mod.build_server(repo, config=make_config(tmp_path), embedded=True)
    result = asyncio.run(mcp.call_tool("map_symbols", {"repo": str(repo), "limit": 0}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "INVALID_ARGUMENT"
    assert result.structured_content["error"]["violations"] == [
        {"field": "limit", "input": 0, "accepted": ["integer >= 1"]}
    ]


# --- embedded and daemon parity (A3) --------------------------------------------


def test_embedded_registry_aggregates_violations(tmp_path):
    from priorart.registry import RuntimeRegistry

    repo = _repo_with(tmp_path / "repo")
    registry = RuntimeRegistry(make_config(tmp_path))
    try:
        handle = registry.resolve(repo)
        with pytest.raises(PriorartError) as err:
            handle.search("contract_target", k=0, mode="hybrid", intent="find both")
        assert err.value.code == "INVALID_ARGUMENT"
        assert err.value.details["violations"] == EXPECTED_VIOLATIONS
    finally:
        registry.close()


def test_daemon_wire_carries_the_same_ordered_violations(tmp_path):
    repo = _repo_with(tmp_path / "repo")
    with DaemonFixture(tmp_path) as registry:
        handle = registry.resolve(repo)
        with pytest.raises(PriorartError) as err:
            handle.search("contract_target", k=0, mode="hybrid", intent="find both")
        assert err.value.code == "INVALID_ARGUMENT"
        assert err.value.details["violations"] == EXPECTED_VIOLATIONS


def test_mcp_boundary_keeps_domain_validation(tmp_path):
    from priorart import server as server_mod
    from priorart.core.jobs import FINAL_STATES

    repo = _repo_with(tmp_path / "repo")
    mcp = server_mod.build_server(repo, config=make_config(tmp_path), embedded=True)
    result = asyncio.run(
        mcp.call_tool(
            "search_codebase",
            {"query": "contract_target", "k": 0, "mode": "hybrid", "intent": "find both"},
        )
    )
    assert result.is_error is True
    error = result.structured_content["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert error["violations"] == EXPECTED_VIOLATIONS
    assert result.structured_content["repo"] == str(repo.resolve())

    submitted = asyncio.run(mcp.call_tool("refresh_index", {}))
    assert not submitted.is_error
    job_id = submitted.structured_content["data"]["job"]["job_id"]
    while True:
        current = asyncio.run(mcp.call_tool("get_index_job", {"job_id": job_id}))
        if current.structured_content["data"]["job"]["state"] in FINAL_STATES:
            break
    # one corrected retry is enough: no raw framework error crossed the boundary
    fixed = asyncio.run(
        mcp.call_tool(
            "search_codebase",
            {"query": "contract_target", "k": 3, "mode": "balanced", "intent": "any"},
        )
    )
    assert fixed.is_error is False
    assert fixed.structured_content["data"]["candidates"][0]["qualname"] == "contract_target"
