from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

LANGS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
}

KIND_BY_TYPE = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "method": "method",
    "singleton_method": "method",
    "class_definition": "class",
    "class_declaration": "class",
    "class_specifier": "class",
    "class": "class",
    "module": "module",
    "struct_item": "struct",
    "struct_specifier": "struct",
    "struct_declaration": "struct",
    "trait_item": "trait",
    "impl_item": "impl",
    "interface_declaration": "interface",
    "type_alias_declaration": "type_alias",
    "enum_declaration": "enum",
    "enum_item": "enum",
}

DEF_TYPES = frozenset(KIND_BY_TYPE)

SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "target",
        ".idea",
    }
)

MAX_FILE_BYTES = 1_000_000

_NAME_FALLBACKS = (
    "identifier",
    "field_identifier",
    "type_identifier",
    "property_identifier",
    "qualified_identifier",
)

_DECLARATOR_CHAIN = frozenset(
    {
        "function_declarator",
        "pointer_declarator",
        "array_declarator",
        "parenthesized_declarator",
        "init_declarator",
        "sized_pointer_declarator",
        "attributed_declarator",
    }
)


@dataclass
class Symbol:
    path: str
    name: str
    qualname: str
    kind: str
    lang: str
    line: int
    end_line: int
    signature: str
    full_signature: str
    docstring: str
    body: str
    search_text: str
    embed_text: str


@dataclass
class ParseResult:
    symbols: list[Symbol]
    # ok: parsed cleanly; partial: syntax errors but symbols extracted;
    # empty: parsed cleanly, no symbols; unsupported: no parser;
    # error: syntax errors and no symbols, or parser raised.
    status: str
    detail: str | None = None


_parsers: dict[str, object] = {}
_parser_errors: dict[str, str] = {}


def _parser(lang: str):
    if lang not in _parsers:
        try:
            _parsers[lang] = get_parser(lang)
        except Exception as err:  # noqa: BLE001 - language packs raise varied errors
            _parsers[lang] = None
            _parser_errors[lang] = f"{type(err).__name__}: {err}"
    return _parsers[lang]


def preflight_parsers(langs) -> dict[str, str]:
    """Load parsers up front; return per-language failure reasons."""
    failures: dict[str, str] = {}
    for lang in sorted(set(langs)):
        if _parser(lang) is None:
            failures[lang] = _parser_errors.get(lang, "unknown error")
    return failures


def repo_languages(root: Path) -> list[str]:
    root = Path(root)
    return sorted({LANGS[Path(rel).suffix] for rel in _list_files(root)})


def parse_source(data: bytes, lang: str, rel_path: str) -> ParseResult:
    parser = _parser(lang)
    if parser is None:
        detail = f"no parser available for language {lang!r}"
        reason = _parser_errors.get(lang)
        if reason:
            detail = f"{detail} ({reason})"
        return ParseResult([], "unsupported", detail)
    try:
        tree = parser.parse(data)
    except Exception as err:  # noqa: BLE001 - tree-sitter raises varied errors
        return ParseResult([], "error", f"parser raised {type(err).__name__}: {err}")
    out: list[Symbol] = []
    lines = data.decode("utf-8", "replace").split("\n")
    _walk(tree.root_node, data, lines, [], out, lang, rel_path)
    if tree.root_node.has_error:
        if out:
            return ParseResult(out, "partial", "source contains syntax errors")
        return ParseResult([], "error", "source contains syntax errors")
    if not out:
        return ParseResult([], "empty", None)
    return ParseResult(out, "ok", None)


def _node_text(node, data: bytes) -> str:
    return data[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _kind(node) -> str | None:
    kind = KIND_BY_TYPE.get(node.type)
    if kind is not None:
        return kind
    if node.type == "type_spec":
        for child in node.named_children:
            if child.type == "struct_type":
                return "struct"
            if child.type == "interface_type":
                return "interface"
        return "type_alias"
    return None


def _signature_block(node, data: bytes) -> str:
    """Full definition header up to (excluding) the body; empty for bodyless nodes."""
    body = node.child_by_field_name("body")
    if body is None:
        return ""
    return data[node.start_byte : body.start_byte].decode("utf-8", "replace").strip()


def _walk(  # noqa: PLR0913, PLR0917 - recursive tree walk context
    node, data: bytes, lines: list[str], parents, out: list[Symbol], lang: str, rel_path: str
) -> None:
    next_parents = parents
    kind = _kind(node)
    if kind is not None:
        name = _node_name(node, data)
        if name:
            text = _node_text(node, data)
            qualname = ".".join([*parents, name])
            docstring = _docstring(node, data, lang)
            signature = text.split("\n", 1)[0][:200]
            full_signature = _signature_block(node, data) or signature
            body = "\n".join(lines[node.start_point.row : node.end_point.row + 1])
            search_text = "\n".join(
                part for part in (qualname, rel_path, signature, docstring) if part
            )
            embed_text = "\n".join(
                part
                for part in (f"{kind} {qualname}", f"file: {rel_path}", signature, docstring)
                if part
            )[:1500]
            out.append(
                Symbol(
                    path=rel_path,
                    name=name,
                    qualname=qualname,
                    kind=kind,
                    lang=lang,
                    line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    signature=signature,
                    full_signature=full_signature,
                    docstring=docstring,
                    body=body,
                    search_text=search_text,
                    embed_text=embed_text,
                )
            )
            next_parents = [*parents, name]
    for child in node.named_children:
        _walk(child, data, lines, next_parents, out, lang, rel_path)


def _node_name(node, data: bytes) -> str | None:
    child = node.child_by_field_name("name")
    if child is not None:
        return _node_text(child, data)
    if node.type in ("function_definition", "declaration", "field_declaration"):
        name = _declarator_name(node, data)
        if name is not None:
            return name
    for child in node.named_children:
        if child.type in _NAME_FALLBACKS:
            return _node_text(child, data)
    return None


def _declarator_name(node, data: bytes) -> str | None:
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type in _NAME_FALLBACKS:
            return _node_text(decl, data)
        nxt = decl.child_by_field_name("declarator")
        if nxt is None:
            break
        decl = nxt
    return None


def _docstring(node, data: bytes, lang: str) -> str:
    if lang != "python":
        return ""
    body = node.child_by_field_name("body")
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type == "expression_statement":
        if not first.named_children:
            return ""
        first = first.named_children[0]
    if first.type != "string":
        return ""
    for child in first.named_children:
        if child.type == "string_content":
            return _node_text(child, data).strip()[:500]
    return ""


def _git(root: Path, *args: str) -> str | None:
    proc = subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "-C", str(root), *args],  # noqa: S607 - partial path is fine
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _list_files(root: Path) -> list[str]:
    proc = subprocess.run(  # noqa: S603 - fixed git argv
        ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607 - partial path is fine
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        rels = proc.stdout.split("\0")
    else:
        rels = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                rels.append((Path(dirpath) / filename).relative_to(root).as_posix())
    out = []
    for rel in rels:
        if not rel or Path(rel).suffix not in LANGS:
            continue
        full = root / rel
        try:
            if full.is_symlink() or full.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        out.append(rel)
    return out


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _open_beneath(root: Path, rel: str) -> int:
    """Open rel strictly beneath root, refusing symlinked path components."""
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise FileNotFoundError(rel)
    dir_fd = os.open(root, os.O_RDONLY)
    opened: int | None = None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | _NOFOLLOW, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        opened = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    return opened


def _capture_file(root: Path, rel: str) -> tuple[bytes, os.stat_result] | None:
    """Read file bytes and its stat together, from the same open file description."""
    try:
        fd = _open_beneath(root, rel)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as fh:
            before = os.fstat(fh.fileno())
            data = fh.read()
            after = os.fstat(fh.fileno())
    except OSError:
        return None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return None
    if len(data) != before.st_size:
        return None
    return data, before


def index_repo(conn, root: Path, embed_fn=None, *, rebuild: bool = False) -> dict:
    root = Path(root).resolve()
    repo = str(root)
    head = _git(root, "rev-parse", "HEAD")
    rels = _list_files(root)
    current = set(rels)
    known = {
        path: (mtime_ns, size)
        for path, mtime_ns, size in conn.execute(
            "SELECT path, mtime_ns, size FROM files WHERE repo = ?", (repo,)
        )
    }
    removed = 0
    warnings: list[str] = []
    symbols = 0
    files = 0
    try:
        for rel in known:
            if rel not in current:
                _drop_file(conn, repo, rel)
                removed += 1
        for rel in rels:
            full = root / rel
            try:
                st = full.stat()
            except OSError:
                continue
            if not rebuild and known.get(rel) == (st.st_mtime_ns, st.st_size):
                continue
            count, warning = _reindex_file(conn, repo, root, rel, embed_fn)
            if warning is not None:
                warnings.append(warning)
                if count == 0:
                    continue
            symbols += count
            files += 1
        conn.execute(
            "INSERT INTO repos (repo, head, indexed_at) VALUES (?, ?, ?) "
            "ON CONFLICT(repo) DO UPDATE SET head = excluded.head, "
            "indexed_at = excluded.indexed_at",
            (repo, head, time.time()),
        )
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
    return {"files": files, "symbols": symbols, "removed": removed, "warnings": warnings}


def _reindex_file(conn, repo: str, root: Path, rel: str, embed_fn) -> tuple[int, str | None]:
    captured = _capture_file(root, rel)
    if captured is None:
        _record_parse_state(conn, repo, rel, "unreadable", None)
        return 0, f"{rel}: unreadable or changed while reading; not indexed, will retry"
    data, st = captured
    lang = LANGS[Path(rel).suffix]
    result = parse_source(data, lang, rel)
    if result.status in ("unsupported", "error"):
        _record_parse_state(conn, repo, rel, result.status, result.detail)
        return 0, f"{rel}: parse {result.status} ({result.detail}); kept previous symbols"
    symbols = result.symbols
    vectors = None
    if symbols and embed_fn is not None:
        vectors, warning = embed_fn([symbol.embed_text for symbol in symbols])
        if vectors is None:
            detail = warning or "embedding failed without a warning"
            _record_parse_state(conn, repo, rel, "embed_failed", f"{detail}; kept previous symbols")
            return 0, detail
    _record_parse_state(conn, repo, rel, result.status, result.detail)
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, rel))
    ]
    _drop_symbols(conn, ids)
    count = 0
    for symbol in symbols:
        cur = conn.execute(
            "INSERT INTO symbols (repo, path, name, qualname, kind, lang, line, end_line, "
            "signature, full_signature, docstring, body, search_text, embed_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                repo,
                symbol.path,
                symbol.name,
                symbol.qualname,
                symbol.kind,
                symbol.lang,
                symbol.line,
                symbol.end_line,
                symbol.signature,
                symbol.full_signature,
                symbol.docstring,
                symbol.body,
                symbol.search_text,
                symbol.embed_text,
            ),
        )
        if vectors is not None:
            conn.execute(
                "INSERT INTO symbols_vec (symbol_id, repo, embedding) VALUES (?, ?, ?)",
                (cur.lastrowid, repo, vectors[count]),
            )
        count += 1
    conn.execute(
        "INSERT INTO files (repo, path, mtime_ns, size) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(repo, path) DO UPDATE SET mtime_ns = excluded.mtime_ns, "
        "size = excluded.size",
        (repo, rel, st.st_mtime_ns, st.st_size),
    )
    conn.commit()
    warning = None
    if result.status == "partial":
        warning = f"{rel}: partial parse, {len(symbols)} symbols extracted from source with errors"
    try:
        now = (root / rel).stat()
    except OSError:
        return count, warning
    if (now.st_mtime_ns, now.st_size) != (st.st_mtime_ns, st.st_size):
        changed = f"{rel}: changed during indexing; result may be stale until next refresh"
        warning = f"{warning}; {changed}" if warning else changed
    return count, warning


def _record_parse_state(conn, repo: str, rel: str, status: str, detail: str | None) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO parse_state (repo, path, status, detail, attempted_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (repo, rel, status, detail, time.time()),
    )


def _drop_file(conn, repo: str, rel: str) -> None:
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, rel))
    ]
    _drop_symbols(conn, ids)
    conn.execute("DELETE FROM files WHERE repo = ? AND path = ?", (repo, rel))
    conn.execute("DELETE FROM parse_state WHERE repo = ? AND path = ?", (repo, rel))
    conn.commit()


def _drop_symbols(conn, ids: list[int]) -> None:
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM symbols_vec WHERE symbol_id IN ({marks})", ids)  # noqa: S608 - marks are placeholders only
    conn.execute(f"DELETE FROM symbols WHERE id IN ({marks})", ids)  # noqa: S608 - marks are placeholders only
