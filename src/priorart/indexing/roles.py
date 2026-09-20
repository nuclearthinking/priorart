"""Source roles: what a file is, computed from path conventions.

Roles drive intent filtering (implementation vs tests) and the production
preference in ranking. Conventions only — configurable globs are a later
extension point; the first cut stays deterministic and language-agnostic
where possible.
"""

from __future__ import annotations

import re

ROLE_PRODUCTION = "production"
ROLE_TEST = "test"
ROLE_FIXTURE = "fixture"
ROLE_EXAMPLE = "example"
ROLE_GENERATED = "generated"
ROLE_UNKNOWN = "unknown"

_TEST_DIRS = re.compile(r"(^|/)(tests?|__tests__|spec)(/|$)")
_FIXTURE_DIRS = re.compile(r"(^|/)(fixtures?|testdata|__mocks__)(/|$)")
_EXAMPLE_DIRS = re.compile(r"(^|/)(examples?|samples?)(/|$)")
_GENERATED_DIRS = re.compile(r"(^|/)(generated|_generated|gen)(/|$)")

_TEST_NAMES = re.compile(
    r"^(test_[^/]+|[^/]+_test)\.(py|go|rs|ts|tsx|js|jsx|mjs|cjs)$"
    r"|^(conftest)\.py$"
    r"|[^/]+\.(spec|test)\.(ts|tsx|js|jsx|mjs|cjs)$"
    r"|^test\.py$"
)
_FIXTURE_NAMES = re.compile(r"(^|/)(fixture|fixtures|conftest|mocks?)($|[._])", re.IGNORECASE)
_EXAMPLE_NAMES = re.compile(r"(^|/)(example|sample|demo)s?($|[._])", re.IGNORECASE)
_GENERATED_SUFFIXES = (".g.dart", ".pb.go", "_pb2.py", ".generated.ts", ".gen.ts")


def source_role(rel: str) -> str:
    """Classify one repository-relative path into a source role."""
    path = rel.strip("/")
    if not path:
        return ROLE_UNKNOWN
    name = path.rsplit("/", 1)[-1]
    if _FIXTURE_DIRS.search(path) or _FIXTURE_NAMES.search(name):
        return ROLE_FIXTURE
    if _GENERATED_DIRS.search(path) or name.endswith(_GENERATED_SUFFIXES):
        return ROLE_GENERATED
    if _TEST_DIRS.search(path) or _TEST_NAMES.match(name):
        return ROLE_TEST
    if _EXAMPLE_DIRS.search(path) or _EXAMPLE_NAMES.search(name):
        return ROLE_EXAMPLE
    return ROLE_PRODUCTION


__all__ = [
    "ROLE_EXAMPLE",
    "ROLE_FIXTURE",
    "ROLE_GENERATED",
    "ROLE_PRODUCTION",
    "ROLE_TEST",
    "ROLE_UNKNOWN",
    "source_role",
]
