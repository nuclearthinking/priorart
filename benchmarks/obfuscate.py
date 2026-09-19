"""Obfuscate benchmark artifacts for publication.

Replaces sensitive terms with abstract placeholders while preserving the JSON
structure and cross-file consistency: the same term maps to the same
placeholder in queries, rationales, paths, case ids, suite names, and file
names. Originals are never modified — each source is read and an obfuscated
copy is written to the output directory.

The replacement vocabulary is private: it lives in an ignored JSON file
(``{term: placeholder}``) so the mapping itself never enters the repository.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLACEMENTS_PATH = ROOT / ".bench" / "obfuscation.json"

# Keys that carry provenance data (internal issue trackers, internal repos)
# and must be dropped entirely from published copies.
REDACTED_KEYS = {"issue_url"}


def load_replacements(path: Path) -> dict[str, str]:
    replacements = json.loads(path.read_text())
    if not isinstance(replacements, dict) or not replacements:
        raise ValueError(f"replacements file must be a non-empty JSON object: {path}")
    invalid = [
        term
        for term, placeholder in replacements.items()
        if not isinstance(term, str) or not isinstance(placeholder, str)
    ]
    if invalid:
        raise ValueError(f"replacements must map strings to strings, got: {invalid}")
    return replacements


def _smart_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


class Obfuscator:
    """Applies a private term-to-placeholder vocabulary to strings and JSON."""

    def __init__(self, replacements: dict[str, str]) -> None:
        # Adjacent alphanumerics block a match, so words merely containing a
        # term stay untouched while hyphenated ids, paths, and UPPER_SNAKE
        # identifiers are replaced.
        self._patterns = {
            re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", re.IGNORECASE): placeholder
            for term, placeholder in replacements.items()
        }

    def text(self, value: str) -> str:
        for pattern, placeholder in self._patterns.items():
            value = pattern.sub(
                lambda match, repl=placeholder: _smart_case(repl, match.group(0)), value
            )
        return value

    def value(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            return {
                self.text(key): self.value(item)
                for key, item in value.items()
                if key not in REDACTED_KEYS
            }
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Obfuscate benchmark result files.")
    parser.add_argument("sources", nargs="+", type=Path, help="Result JSON files to obfuscate")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for obfuscated copies (default: benchmarks/results)",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        default=DEFAULT_REPLACEMENTS_PATH,
        help=f"Private vocabulary file, JSON object (default: {DEFAULT_REPLACEMENTS_PATH})",
    )
    args = parser.parse_args()

    obfuscator = Obfuscator(load_replacements(args.replacements))
    for source in args.sources:
        data = json.loads(source.read_text())
        output = args.out_dir / obfuscator.text(source.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(obfuscator.value(data), ensure_ascii=False, indent=2) + "\n")
        print(f"obfuscated: {source} -> {output}")


if __name__ == "__main__":
    main()
