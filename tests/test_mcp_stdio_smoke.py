"""Black-box smoke for the documented agent-facing stdio setup.

Reads the README's marked MCP config block, starts the documented command
from ``cwd=/`` with a temporary HOME, index and socket, and drives one
session through the whole documented flow: tools/list schema, refresh, poll
lexical_ready, search (including two parallel searches on one session) and a
second worktree.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.helpers import repo_with_files

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _readme_server_definition() -> dict:
    """The marked MCP config block from the README (executable documentation)."""
    text = (_REPO_ROOT / "README.md").read_text()
    blocks = re.findall(r"```json mcp-config\n(.*?)```", text, re.DOTALL)
    assert len(blocks) == 1, "README must contain exactly one marked mcp-config block"
    definition = json.loads(blocks[0])["mcpServers"]["priorart"]
    assert definition["cwd"] == "/"
    assert definition["args"][-2:] == ["priorart", "serve"]
    return definition


def _readme_section(start: str, end: str) -> str:
    text = (_REPO_ROOT / "README.md").read_text()
    return text[text.index(start) : text.index(end)]


def _readme_search_modes() -> list[str]:
    section = _readme_section("Search modes:", "Exact symbol")
    return re.findall(r"^- `(\w+)`:", section, re.MULTILINE)


def _readme_search_intents() -> list[str]:
    section = _readme_section("Search intents are", "Search results are ranked")
    sentence = section[: section.index(".")]
    return re.findall(r"`(\w+)`", sentence)


def _readme_tool_names() -> set[str]:
    section = _readme_section("## MCP interface", "## Errors and recovery")
    return set(re.findall(r"^\| `(\w+)` \|", section, re.MULTILINE))


def test_documented_stdio_flow_from_foreign_cwd(tmp_path):
    asyncio.run(_stdio_flow(tmp_path))


async def _stdio_flow(tmp_path: Path) -> None:
    first = repo_with_files(
        tmp_path / "first",
        {
            "first.py": "def first_workspace_symbol():\n    return 1\n",
            "test_first.py": "def test_first_workspace_symbol():\n    assert True\n",
        },
    )
    second = repo_with_files(
        tmp_path / "second",
        {"second.py": "def second_workspace_symbol():\n    return 2\n"},
    )
    socket_dir = Path(tempfile.mkdtemp(prefix="pa-stdio-", dir="/tmp"))
    socket_path = socket_dir / "priorart.sock"
    config_path = tmp_path / "priorart.env"
    config_path.write_text(
        "\n".join(
            (
                f"PRIORART_INDEX_DIR={tmp_path / 'indexes'}",
                f"PRIORART_DAEMON_SOCKET={socket_path}",
                "PRIORART_WATCH_INTERVAL=0",
                "PRIORART_LLM_MODEL=",
                "PRIORART_EMBED_MODEL=",
                "PRIORART_RERANK_MODEL=",
            )
        )
        + "\n"
    )
    env = {**os.environ, "HOME": str(tmp_path / "home")}
    daemon = subprocess.Popen(  # noqa: ASYNC220, S603 - fixed interpreter/module argv
        [
            sys.executable,
            "-m",
            "priorart",
            "daemon",
            "start",
            "--socket",
            str(socket_path),
            "--config",
            str(config_path),
        ],
        cwd="/",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_socket(socket_path, daemon)
        # the README documents the command; this smoke runs the same
        # documented surface (priorart serve with an explicit config file)
        # from cwd=/ with a foreign HOME
        _readme_server_definition()
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "priorart", "serve", "--config", str(config_path)],
            cwd="/",
            env=env,
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            missing = await session.call_tool(
                "search_codebase", {"query": "first_workspace_symbol"}
            )
            assert missing.is_error
            assert missing.structured_content["error"]["code"] == "REPOSITORY_NOT_SELECTED"

            await _assert_search_tool_contract(session)

            first_canonical = await _refresh_and_wait(session, first)

            await _assert_schema_driven_first_search(session, first)

            await _assert_invalid_arguments_via_stdio(session, first, first_canonical)

            found_first, mapped = await asyncio.gather(
                session.call_tool(
                    "search_codebase",
                    {"query": "first_workspace_symbol", "repo": str(first), "mode": "fast"},
                ),
                session.call_tool("map_symbols", {"repo": str(first), "path_glob": "test_*"}),
            )
            assert not found_first.is_error
            assert found_first.structured_content["data"]["candidates"][0]["qualname"] == (
                "first_workspace_symbol"
            )
            assert not mapped.is_error

            await _refresh_and_wait(session, second)
            found_second, status = await asyncio.gather(
                session.call_tool(
                    "search_codebase",
                    {"query": "second_workspace_symbol", "repo": str(second), "mode": "fast"},
                ),
                session.call_tool("get_index_status", {"repo": str(second)}),
            )
            assert not found_second.is_error
            candidate = found_second.structured_content["data"]["candidates"][0]
            assert candidate["qualname"] == "second_workspace_symbol"
            assert found_second.structured_content["repo"] == str(second.resolve())
            assert not status.is_error
            assert status.structured_content["repo"] == str(second.resolve())

            # A6: two worktrees of the same session searched concurrently
            left, right = await asyncio.gather(
                session.call_tool(
                    "search_codebase",
                    {"query": "first_workspace_symbol", "repo": str(first), "mode": "fast"},
                ),
                session.call_tool(
                    "search_codebase",
                    {"query": "second_workspace_symbol", "repo": str(second), "mode": "fast"},
                ),
            )
            assert not left.is_error
            assert not right.is_error
            assert left.structured_content["repo"] == first_canonical
            assert right.structured_content["repo"] == str(second.resolve())
            assert left.structured_content["data"]["candidates"][0]["qualname"] == (
                "first_workspace_symbol"
            )
            assert right.structured_content["data"]["candidates"][0]["qualname"] == (
                "second_workspace_symbol"
            )
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=10)
        shutil.rmtree(socket_dir, ignore_errors=True)


async def _assert_search_tool_contract(session: ClientSession) -> None:
    """A1/A10: tools/list publishes the full contract; README agrees with it."""
    tools = {tool.name: tool for tool in (await session.list_tools()).tools}
    search = tools["search_codebase"]
    props = search.input_schema["properties"]
    assert props["mode"]["enum"] == ["fast", "balanced", "deep"]
    assert props["mode"]["default"] == "balanced"
    assert props["intent"]["enum"] == ["implementation", "tests", "any"]
    assert props["intent"]["default"] == "implementation"
    assert props["k"]["minimum"] == 1
    assert props["k"]["default"] == 10
    for name in ("query", "repo", "k", "mode", "intent"):
        assert props[name]["description"], name
    assert props["query"]["examples"]
    assert props["repo"]["examples"]
    assert props["k"]["examples"]
    assert tools["map_symbols"].input_schema["properties"]["limit"]["minimum"] == 1
    # the README describes the same surface, not a divergent copy of it
    assert _readme_search_modes() == props["mode"]["enum"]
    assert _readme_search_intents() == props["intent"]["enum"]
    assert _readme_tool_names() == set(tools)
    output = search.output_schema
    assert output["type"] == "object"
    # unwrapped success schema: no result envelope around the payload
    assert "result" not in output["properties"]
    assert output["properties"]["ok"]["const"] is True
    assert set(output["required"]) == {"ok", "repo", "index", "data", "warnings", "timings"}
    candidate_ref = output["$defs"]["_SearchCandidate"]["properties"]
    assert set(candidate_ref) == {
        "path",
        "name",
        "qualname",
        "kind",
        "lang",
        "line",
        "end_line",
        "signature",
        "full_signature",
        "docstring",
        "source_role",
        "score",
    }


async def _assert_schema_driven_first_search(session: ClientSession, repo: Path) -> None:
    """A2: an agent reading only the schema needs no repair rounds.

    Mirrors the real incident: one natural-language query covering both
    implementation and tests, arguments chosen from the published schema.
    """
    tools = {tool.name: tool for tool in (await session.list_tools()).tools}
    props = tools["search_codebase"].input_schema["properties"]
    mode = props["mode"]["default"]
    intent = "any" if "any" in props["intent"]["enum"] else props["intent"]["default"]
    k = props["k"]["default"]

    result = await session.call_tool(
        "search_codebase",
        {
            "query": "the first workspace symbol and its test",
            "repo": str(repo),
            "k": k,
            "mode": mode,
            "intent": intent,
        },
    )
    assert not result.is_error, result.structured_content
    # A9: machine payload carries no presentation copy of the rendered report
    assert "text" not in result.structured_content["data"]
    assert result.content
    assert result.content[0].text.strip()
    assert "Results are hints" in result.content[0].text
    assert json.dumps(result.structured_content).count("Results are hints") == 0
    qualnames = {
        candidate["qualname"] for candidate in result.structured_content["data"]["candidates"]
    }
    assert {"first_workspace_symbol", "test_first_workspace_symbol"} <= qualnames


async def _assert_invalid_arguments_via_stdio(
    session: ClientSession, repo: Path, canonical_repo: str
) -> None:
    """A3: the stdio path returns the same aggregated, ordered violations."""
    result = await session.call_tool(
        "search_codebase",
        {"query": "x", "repo": str(repo), "k": 0, "mode": "hybrid", "intent": "find both"},
    )
    assert result.is_error
    assert result.structured_content["repo"] == canonical_repo
    assert result.structured_content["error"]["code"] == "INVALID_ARGUMENT"
    assert result.structured_content["error"]["violations"] == [
        {"field": "k", "input": 0, "accepted": ["integer >= 1"]},
        {"field": "mode", "input": "hybrid", "accepted": ["fast", "balanced", "deep"]},
        {
            "field": "intent",
            "input": "find both",
            "accepted": ["implementation", "tests", "any"],
        },
    ]


async def _refresh_and_wait(session: ClientSession, repo: Path) -> str:
    submitted = await session.call_tool("refresh_index", {"repo": str(repo)})
    assert not submitted.is_error
    assert submitted.structured_content["data"]["submission"] == "started"
    job_id = submitted.structured_content["data"]["job"]["job_id"]
    canonical_repo = submitted.structured_content["repo"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        current = await session.call_tool(
            "get_index_job", {"repo": canonical_repo, "job_id": job_id}
        )
        assert not current.is_error
        job = current.structured_content["data"]["job"]
        if job["lexical_ready"]:
            return canonical_repo
        if job["state"] in {"failed", "cancelled", "interrupted"}:
            raise AssertionError(job)
        await asyncio.sleep(0.02)
    raise AssertionError("index did not become lexical-ready")


def _wait_for_socket(path: Path, process: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(f"daemon exited early: {stderr}")
        time.sleep(0.02)
    raise AssertionError("daemon socket did not appear")
