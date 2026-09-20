"""Tree-sitter symbol extraction.

The parser is a pure function from bytes to symbols; it never touches git,
SQLite or the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from tree_sitter_language_pack import get_parser

from priorart.core import bounded_body

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

MAX_FILE_BYTES = 1_000_000

# Representation v2: symbol texts carry a bounded body excerpt so that dense
# and lexical retrieval see behavior, not only names and docstrings.
BODY_EXCERPT_CHARS = 800
EMBED_TEXT_MAX_CHARS = 2200
EMBED_TEXT_FORMAT = "kind-qualname-file-signature-docstring-body800-v2"
SEARCH_TEXT_FORMAT = "qualname-path-signature-docstring-body800-v2"


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
            excerpt = bounded_body(body, BODY_EXCERPT_CHARS)
            search_text = "\n".join(
                part for part in (qualname, rel_path, signature, docstring, excerpt) if part
            )
            embed_text = "\n".join(
                part
                for part in (
                    f"{kind} {qualname}",
                    f"file: {rel_path}",
                    signature,
                    docstring,
                    excerpt,
                )
                if part
            )[:EMBED_TEXT_MAX_CHARS]
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


def language_of(rel: str) -> str | None:
    """Language for a relative path, or ``None`` when unsupported."""
    return LANGS.get(PurePosixPath(rel).suffix)
