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
    docstring: str
    search_text: str
    embed_text: str


_parsers: dict[str, object] = {}


def _parser(lang: str):
    if lang not in _parsers:
        try:
            _parsers[lang] = get_parser(lang)
        except Exception:  # noqa: BLE001 - language packs raise varied errors
            _parsers[lang] = None
    return _parsers[lang]


def parse_source(data: bytes, lang: str, rel_path: str) -> list[Symbol]:
    parser = _parser(lang)
    if parser is None:
        return []
    tree = parser.parse(data)
    out: list[Symbol] = []
    _walk(tree.root_node, data, [], out, lang, rel_path)
    return out


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


def _walk(node, data: bytes, parents, out: list[Symbol], lang: str, rel_path: str) -> None:
    next_parents = parents
    kind = _kind(node)
    if kind is not None:
        name = _node_name(node, data)
        if name:
            text = _node_text(node, data)
            qualname = ".".join([*parents, name])
            docstring = _docstring(node, data, lang)
            signature = text.split("\n", 1)[0][:200]
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
                    docstring=docstring,
                    search_text=search_text,
                    embed_text=embed_text,
                )
            )
            next_parents = [*parents, name]
    for child in node.named_children:
        _walk(child, data, next_parents, out, lang, rel_path)


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
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _list_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
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


def index_repo(conn, root: Path, embed_fn=None, rebuild: bool = False) -> dict:
    root = Path(root).resolve()
    repo = str(root)
    head = _git(root, "rev-parse", "HEAD")
    rels = _list_files(root)
    current = set(rels)
    known = {
        path: (mtime, size)
        for path, mtime, size in conn.execute(
            "SELECT path, mtime, size FROM files WHERE repo = ?", (repo,)
        )
    }
    removed = 0
    for rel in known:
        if rel not in current:
            _drop_file(conn, repo, rel)
            removed += 1
    warnings: list[str] = []
    symbols = 0
    files = 0
    for rel in rels:
        full = root / rel
        try:
            st = full.stat()
        except OSError:
            continue
        if not rebuild and known.get(rel) == (st.st_mtime, st.st_size):
            continue
        count, warning = _reindex_file(conn, repo, root, rel, embed_fn)
        if warning is not None:
            warnings.append(warning)
            continue
        symbols += count
        files += 1
    conn.execute(
        "INSERT INTO repos (repo, head, indexed_at) VALUES (?, ?, ?) "
        "ON CONFLICT(repo) DO UPDATE SET head = excluded.head, indexed_at = excluded.indexed_at",
        (repo, head, time.time()),
    )
    conn.commit()
    return {"files": files, "symbols": symbols, "removed": removed, "warnings": warnings}


def _reindex_file(conn, repo: str, root: Path, rel: str, embed_fn) -> tuple[int, str | None]:
    full = root / rel
    lang = LANGS[full.suffix]
    try:
        data = full.read_bytes()
    except OSError:
        _drop_file(conn, repo, rel)
        return 0, None
    symbols = parse_source(data, lang, rel)
    vectors = None
    if symbols and embed_fn is not None:
        vectors, warning = embed_fn([symbol.embed_text for symbol in symbols])
        if vectors is None:
            return 0, warning
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, rel))
    ]
    _drop_symbols(conn, ids)
    count = 0
    for symbol in symbols:
        cur = conn.execute(
            "INSERT INTO symbols (repo, path, name, qualname, kind, lang, line, end_line, "
            "signature, docstring, search_text, embed_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                symbol.docstring,
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
    try:
        st = full.stat()
    except OSError:
        conn.commit()
        return count, None
    conn.execute(
        "INSERT INTO files (repo, path, mtime, size) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(repo, path) DO UPDATE SET mtime = excluded.mtime, size = excluded.size",
        (repo, rel, st.st_mtime, st.st_size),
    )
    conn.commit()
    return count, None


def _drop_file(conn, repo: str, rel: str) -> None:
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM symbols WHERE repo = ? AND path = ?", (repo, rel))
    ]
    _drop_symbols(conn, ids)
    conn.execute("DELETE FROM files WHERE repo = ? AND path = ?", (repo, rel))
    conn.commit()


def _drop_symbols(conn, ids: list[int]) -> None:
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM symbols_vec WHERE symbol_id IN ({marks})", ids)
    conn.execute(f"DELETE FROM symbols WHERE id IN ({marks})", ids)
