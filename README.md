# Priorart

Priorart is a local code-reuse search MCP for coding agents. It indexes symbols
from Git working trees and helps an agent find code to import, extend, or
refactor before creating a duplicate. Search results are advisory: always read
the referenced source before reuse.

## Use Priorart from Codex

### 1. Install and configure Priorart

```bash
git clone https://github.com/nuclearthinking/priorart.git
cd priorart
uv sync
mkdir -p ~/.priorart
cp .env.example ~/.priorart/priorart.env
```

Remote expansion, embedding, and reranking are optional. The default empty
model settings provide lexical FTS search without network calls.

### 2. Add the MCP server to Codex

Add this entry to `~/.codex/config.toml`, replacing both absolute paths:

```toml
[mcp_servers.priorart]
command = "/absolute/path/to/uv"
args = ["run", "--project", "/absolute/path/to/priorart", "priorart", "serve"]
cwd = "/"
```

Restart Codex after changing its configuration. The server deliberately works
from any process `cwd`; each repository-scoped call selects its target with an
absolute `repo` argument. See the official
[Codex MCP configuration](https://developers.openai.com/docs/extend/mcp) for
additional client settings.

### 3. Give the agent this repository instruction

```text
Before creating a function, class, or module, call Priorart search_codebase
with the absolute repository root. Prefer importing, extending, or refactoring
a relevant existing symbol. Read the referenced source before reuse. If the
index is absent, call refresh_index once, poll get_index_job until
lexical_ready is true, and then search. A new worktree has its own index.
```

### 4. First-run tool flow

An agent should execute this sequence:

1. Optionally call `list_workspaces` to discover configured or previously
   indexed paths. Discovery never changes the selected workspace.
2. Call `refresh_index` with `repo="/absolute/path/to/repository"`.
3. Save `data.job.job_id`. `data.submission` is `started` for a new job or
   `joined` when an equivalent refresh is already active.
4. Poll `get_index_job` with the same `repo` and `job_id` until
   `data.job.lexical_ready` is `true`. Search can start while embeddings are
   still running.
5. Call `search_codebase` with the same absolute `repo`.

Do not refresh before every search. The shared coordinator keeps jobs, file
watchers, and model caches alive when an MCP stdio client restarts. A separate
Git worktree is a separate workspace and needs one initial refresh.

## Use Priorart from another MCP client

Use the equivalent stdio definition for clients that accept JSON:

```json mcp-config
{
  "mcpServers": {
    "priorart": {
      "command": "/absolute/path/to/uv",
      "args": [
        "run", "--project", "/absolute/path/to/priorart",
        "priorart", "serve"
      ],
      "cwd": "/"
    }
  }
}
```

`priorart serve --repo /absolute/path` sets one startup default for clients
that cannot pass `repo`. Explicit `repo` always wins. Priorart never guesses
from process `cwd`, the last-used workspace, or the only existing index.
`priorart serve --embedded` hosts jobs inside the stdio process and is intended
only for development and tests.

## MCP interface

Every tool returns human-readable text in `content` plus a machine envelope
in `structuredContent`; the two never duplicate each other. All envelopes
contain `ok` and `repo`; successful calls add `index`, `data`, `warnings`,
and `timings`, while failed calls add `error` and set `is_error=true`.
Repository paths are canonicalized, so a subdirectory or symlink alias
resolves to its owning worktree. Another worktree cannot inspect or cancel
that workspace's job.

The primary contract is `tools/list`: `search_codebase` publishes its input
schema (enums, defaults, descriptions, examples, `k` minimum) and a
top-level success `outputSchema` describing the whole envelope. Error
results keep the separate `is_error=true` shape `{ok: false, repo, error}`.
Read the schema instead of guessing argument values.

| Tool | Important inputs | Result and use |
| --- | --- | --- |
| `search_codebase` | `query`, absolute `repo`, `k`, `mode`, `intent` | Ranked symbols with path, line, signature, role, stages, and degradation reasons |
| `map_symbols` | absolute `repo`, `path_glob`, `limit`, `cursor` | Paginated symbol inventory for a path or component |
| `refresh_index` | absolute `repo`, optional `rebuild`, optional `paths` | Non-blocking job plus `submission: started\|joined` |
| `get_index_job` | absolute `repo`, `job_id` | Job phase, counters, lexical/dense readiness, epoch, and structured failure |
| `cancel_index_job` | absolute `repo`, `job_id` | Requests cancellation of the shared repo-scoped job; a joined caller can cancel it for all participants |
| `get_index_status` | absolute `repo` | Live HEAD/file freshness plus lexical, dense, and parse coverage |
| `list_workspaces` | none | Discovery-only configured and known-indexed workspace paths |

Search modes:

- `fast`: exact and lexical retrieval without optional model stages;
- `balanced`: hybrid retrieval and configured reranking within one deadline;
- `deep`: expansion plus all configured retrieval stages.

Exact symbol or qualname matches are dispatched automatically in `fast` and
`balanced`; `exact` is not a separate mode value.

Search intents are `implementation`, `tests`, and `any`. `implementation`
prefers production symbols over comparable test helpers; `tests` returns test
symbols only. An explicitly requested exact test qualname remains findable.

Search results are ranked discovery hints, not a complete reference or call
graph: a missing candidate does not prove the code is absent. Verify
exhaustive callers and references by reading files and text or language
search. One MCP session may issue several tool calls concurrently; each
operation uses its own daemon connection, so parallel searches run in
parallel.

Stable `degradation_reasons` are `DENSE_INDEX_PARTIAL`,
`PARSE_COVERAGE_PARTIAL`, `QUERY_MODEL_UNAVAILABLE`, `RERANK_FALLBACK`, and
`DEADLINE_FALLBACK`. Intentional omissions in `fast` or a successful exact
dispatch are not degradation.

## Errors and recovery

Tool-call errors use stable `error.code` values and include `next_action` when
the caller can recover. `INVALID_ARGUMENT` carries an ordered `violations`
list (`field`, `input`, `accepted`) covering every invalid argument at once —
correct them all and retry once. Arguments are checked in two layers: the
published schema rejects values of the wrong JSON type (a string where an
integer is declared) with a framework validation message, while type-correct
but invalid values (an unknown `mode`, `k=0`, `limit=0`) return the
structured `INVALID_ARGUMENT` envelope. A failed or interrupted background
job is still a successful job lookup; inspect `data.job.failure` for `code`,
`message`, `retryable`, and `next_action`.

| Code | Agent action |
| --- | --- |
| `REPOSITORY_NOT_SELECTED` | Retry with an absolute `repo`, or configure one `--repo` startup default |
| `INDEX_NOT_READY` | Call `refresh_index`, then poll until `lexical_ready` |
| `INVALID_ARGUMENT` | Read each `violations[]` entry, correct every listed argument, and retry |
| `WRITER_BUSY` | Wait for the external writer to finish, then retry `refresh_index` |
| `REFRESH_FAILED` | Read the failure message, correct the cause, and submit another refresh |
| `JOB_INTERRUPTED` | The daemon restarted; keep a published lexical epoch if present and submit a refresh to finish |
| `DAEMON_PROFILE_MISMATCH` | Stop the existing Priorart daemon using that socket (`priorart daemon stop`), then reconnect; the MCP client starts one with the current config |
| `DAEMON_MISMATCH` | Run `priorart daemon restart`, then retry. If the daemon is already current, restart the MCP client session instead — its process may predate the last code change. If the message says the refresh outcome is unknown, the first refresh may have started: call `refresh_index` again to join or restart it |
| `DAEMON_STOP_FAILED` | `priorart daemon restart` could not stop the holder (no recorded pid, still draining, or another user's process); wait for it to exit and retry, or stop the process manually first |

Run diagnostics without printing secrets:

```bash
uv run --project /absolute/path/to/priorart priorart doctor \
  --repo /absolute/path/to/repository
```

It reports the effective runtime mode, daemon socket, service-profile
fingerprint, package versions, provider health, the daemon's reachability
and code identity, and the selected index state.

## Daemon restart hygiene

The shared daemon outlives MCP client sessions, and the handshake pins the
identity of the running code: after changing Priorart's own source (an
editable checkout changes identity on every edit; an installed
distribution only on upgrade), a daemon started from older code refuses
every operation with a structured `DAEMON_MISMATCH` instead of answering
with old payload shapes. Recovery is one command:

```bash
priorart daemon restart   # stop (graceful drain) + start a fresh detached daemon
```

- `priorart daemon start` runs the daemon in the foreground (what
  autostart launches, detached).
- `priorart daemon stop` is idempotent and never signals a process that
  does not hold the daemon claim: the flock'd sidecar is the single
  instance truth, a stale pid file alone is ignored.
- `priorart daemon restart` never spawns beside a daemon it could not
  stop: if the holder recorded no pid, is still draining, or belongs to
  another user, it fails with a structured `DAEMON_STOP_FAILED` instead
  of raising a replacement that would die on the singleton claim.
- An in-flight refresh interrupted by a restart is reconciled honestly as
  `JOB_INTERRUPTED` (journal-kept); a published lexical epoch keeps
  serving searches.
- `priorart doctor` reports whether the daemon on the configured socket is
  reachable and current — it distinguishes "restart the daemon" from
  "restart your MCP client session" (a client process predating the last
  code change mismatches a fresh daemon the same way).

## Configuration

Priorart reads `PRIORART_*` environment variables, then
`~/.priorart/priorart.env`, or a file passed with `priorart serve --config`.
Real environment variables have highest priority. There is no project-local
`.env` lookup, so configuration does not change with `cwd` or selected repo.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PRIORART_INDEX_DIR` | `~/.priorart/indexes` | Root containing one derived SQLite index per canonical worktree and profile |
| `PRIORART_DAEMON_SOCKET` | `~/.priorart/daemon.sock` | Shared coordinator Unix socket |
| `PRIORART_WATCH_INTERVAL` | `2.0` | Source polling interval; `0` disables auto-refresh |
| `PRIORART_SEARCH_DEADLINE_SECONDS` | `15.0` | End-to-end optional-stage search budget |
| `PRIORART_CANDIDATE_LIMIT` | `50` | Fused candidates entering reranking |
| `PRIORART_BASE_URL` / `PRIORART_API_KEY` | empty | Shared provider endpoint and bearer token |
| `PRIORART_LLM_BASE_URL` / `PRIORART_LLM_API_KEY` / `PRIORART_LLM_MODEL` | shared / empty | Query expansion provider |
| `PRIORART_EMBED_BASE_URL` / `PRIORART_EMBED_API_KEY` / `PRIORART_EMBED_MODEL` | shared / empty | Embedding provider |
| `PRIORART_EMBED_DIM` | `1024` | Stored vector dimension; must match provider output |
| `PRIORART_EMBED_INPUT_FORMAT` | `instruct-text` | `instruct-text` or `qwen3` input contract |
| `PRIORART_RERANK_BASE_URL` / `PRIORART_RERANK_API_KEY` / `PRIORART_RERANK_MODEL` | shared / empty | Reranking provider |
| `PRIORART_RERANK_PROTOCOL` | `openai` | `openai` or `llama-completion` wire protocol |
| `PRIORART_RERANK_QUERY_FORMAT` | `instruct` | `instruct` or `raw` query contract |

Service-specific endpoints and keys override the shared provider values.
Changing the embedding model, dimension, or input format creates a new derived
profile. Index databases are disposable artifacts; source files remain the
only source of truth.

## Index and search behavior

- Git repositories index `git ls-files`; dirty tracked content is read from the
  working tree. Non-Git directories use a filtered filesystem walk.
- Symlinks are skipped during capture so indexing cannot read source outside
  the repository.
- The lexical epoch is committed atomically and becomes searchable before
  optional embeddings finish.
- Source freshness (HEAD/file drift) is independent from dense and parse
  coverage. Partial vectors do not make unchanged source stale.
- The index supports Python, TypeScript/JavaScript, Go, Rust, Java, Ruby, PHP,
  C, C++, and C# through tree-sitter symbol extraction.
- SQLite FTS5 provides lexical search; sqlite-vec, query expansion, and
  reranking enhance it when configured.

## CLI

The CLI is useful for diagnostics and manual indexing:

```bash
uv run priorart index /absolute/path/to/repository
uv run priorart search "retry failed provider calls" --repo /absolute/path/to/repository
uv run priorart status --repo /absolute/path/to/repository
uv run priorart workspaces
```

## Quality evaluation and development

The generic runner in [`benchmarks/`](benchmarks/) evaluates a frozen repo
against verified `path + qualname` answers:

```bash
uv run python benchmarks/run.py \
  --suite /path/to/golden.json \
  --repo /path/to/frozen/repository \
  --label local-model
```

Keep proprietary inputs and results outside the public tree or in ignored
directories such as `.bench/` and `scratch/`.

```bash
uv run pytest -q
uv run ruff check .
```

`tree-sitter` is pinned below 0.26 because newer releases have caused parser
crashes on large repositories in this environment. The recommended runtime is
Python 3.14 with SQLite FTS5 support.
