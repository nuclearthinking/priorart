# priorart

Priorart is a local code-reuse search service for coding agents. Before an
agent creates a function, class, or module, it can ask whether the repository
already contains code that should be imported, extended, or refactored.

- tree-sitter symbol extraction for Python, TypeScript/JavaScript, Go, Rust,
  Java, Ruby, PHP, C, C++, and C#;
- hybrid SQLite FTS5 and sqlite-vec retrieval with reciprocal-rank fusion;
- optional LLM query expansion, embeddings, and reranking through configurable
  HTTP endpoints;
- MCP tools and a command-line interface;
- incremental indexing of Git-tracked working-tree files.

Search results are advisory. Read the referenced source before reusing it.

## Install

```bash
uv sync
cp .env.example .env
```

All remote model stages are optional. With no endpoints configured, Priorart
uses lexical FTS search and reports which stages were skipped.

## Configuration

Priorart is provider agnostic. It expects OpenAI-compatible chat and embedding
endpoints plus a rerank endpoint accepting `query`, `documents`, and `top_n`.
Base URLs should include the provider API prefix, commonly `/v1`.

One provider can serve every stage:

| Variable | Purpose |
|---|---|
| `PRIORART_BASE_URL` | Shared base URL for expansion, embeddings, and reranking |
| `PRIORART_API_KEY` | Optional shared bearer token |

Each stage can instead use a separate provider:

| Variable | Default | Purpose |
|---|---|---|
| `PRIORART_LLM_BASE_URL` | shared base URL | Query-expansion endpoint |
| `PRIORART_LLM_API_KEY` | shared API key | Query-expansion bearer token |
| `PRIORART_LLM_MODEL` | disabled | Query-expansion model |
| `PRIORART_EMBED_BASE_URL` | shared base URL | Embedding endpoint |
| `PRIORART_EMBED_API_KEY` | shared API key | Embedding bearer token |
| `PRIORART_EMBED_MODEL` | disabled | Embedding model |
| `PRIORART_EMBED_DIM` | `1024` | Stored vector dimension |
| `PRIORART_EMBED_INPUT_FORMAT` | `instruct-text` | Embedding input contract: `instruct-text` wraps queries and documents in `Instruct:`/`Text:` tags; `qwen3` follows the official Qwen3-Embedding usage (instruction prefix on queries only, raw documents, L2-normalized vectors) |
| `PRIORART_RERANK_BASE_URL` | shared base URL | Rerank endpoint |
| `PRIORART_RERANK_API_KEY` | shared API key | Reranking bearer token |
| `PRIORART_RERANK_MODEL` | disabled | Reranking model |
| `PRIORART_RERANK_PROTOCOL` | `openai` | Rerank client protocol: `openai` posts to the `/v1/rerank` endpoint; `llama-completion` scores each document through a llama-server `/completion` call and reads P(yes) from the first generated token's logprobs (Qwen3-Reranker judge template) |
| `PRIORART_RERANK_QUERY_FORMAT` | `instruct` | Rerank query contract for the `openai` protocol: `instruct` wraps the query in manual `<Instruct>`/`<Query>` tags; `raw` passes the query unchanged to endpoints whose template formats it (llama-server `/v1/rerank`) |
| `PRIORART_CANDIDATE_LIMIT` | `50` | Fused (FTS + dense, RRF) symbols entering the rerank pool |
| `PRIORART_DB` | `~/.priorart/index.db` | SQLite index path |

Service-specific settings override the shared provider. Keep real values in an
ignored `.env` file; `.env.example` contains only placeholders.

## CLI

```bash
set -a; source .env; set +a
uv run priorart index /path/to/repository [--rebuild]
uv run priorart search "retry failed provider calls" --repo /path/to/repository
uv run priorart status --repo /path/to/repository
uv run priorart serve [--repo /path/to/repository]
```

`serve` without `--repo` resolves the Git root from its process working
directory. Use `uv run --project /path/to/priorart`; `uv --directory` changes
the process working directory and breaks repository auto-detection.

## MCP configuration

Example local MCP entry:

```json
{
  "mcp": {
    "priorart": {
      "type": "local",
      "command": [
        "/bin/sh", "-c",
        "set -a; . /path/to/priorart/.env; exec uv run --project /path/to/priorart priorart serve"
      ],
      "enabled": true
    }
  }
}
```

Recommended repository instruction:

```text
Before creating a function, class, or module, call priorart search_codebase.
Prefer importing, extending, or refactoring a relevant existing symbol. Read
the referenced file before reuse. If no candidate fits, explain why.
```

## Index behavior

- Files come from `git ls-files`; non-Git directories use a filtered filesystem
  walk.
- Symlinks are skipped, so indexing cannot read source outside the repository.
- Dirty tracked files are indexed from the current working tree.
- Each checkout or worktree path has separate index metadata. Refresh a new
  worktree once before searching it.
- `status` compares indexed and current HEADs, checks file drift and dirty state,
  and reports vector coverage.
- Embedding failures leave a file pending so a later refresh retries it.

The embedding profile — model, vector dimension and input format — is recorded in
the index metadata. Changing any of it (`PRIORART_EMBED_MODEL`,
`PRIORART_EMBED_DIM`, `PRIORART_EMBED_INPUT_FORMAT`) resets the index and the
next refresh re-embeds everything; keeping stale vectors would silently mix
two incompatible vector spaces. The index database is a throwaway artifact
rebuildable by re-indexing.

## Language coverage

Symbol node types are maintained per tree-sitter grammar. Python docstrings are
extracted from the first body statement. Other languages currently index names,
signatures, paths, and source ranges; language-specific documentation comments
and some declaration forms may require additional extractors.

## Quality evaluation

The generic runner in [`benchmarks/`](benchmarks/) evaluates a frozen repository
against intent queries with verified `path + qualname` answers. It reports
recall@k, MRR@k, latency, warnings, and the deeper reranked candidate list.

```bash
uv run python benchmarks/run.py \
  --suite /path/to/golden.json \
  --repo /path/to/frozen/repository \
  --label local-model
```

Keep proprietary repositories, issue text, and benchmark results outside the
public tree or under an ignored directory such as `.bench/`.

## Development

```bash
uv run pytest -q
uv run ruff check .
```

`tree-sitter` is pinned below 0.26 because newer releases have caused parser
crashes on large repositories in this environment. The project uses Python
3.14 because the selected runtime build includes SQLite FTS5.
