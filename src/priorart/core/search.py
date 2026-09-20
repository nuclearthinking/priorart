"""Search request vocabulary shared by every layer.

``mode`` and ``intent`` are one domain vocabulary: adapters publish it as
schema metadata, this module validates it once, and retrieval consumes the
values. Validation aggregates every violation in a fixed order so one bad
call needs one repair round instead of one per field.
"""

from __future__ import annotations

from typing import Literal

from .errors import INVALID_ARGUMENT, PriorartError

SearchMode = Literal["fast", "balanced", "deep"]
SearchIntent = Literal["implementation", "tests", "any"]

SEARCH_MODES: tuple[str, ...] = ("fast", "balanced", "deep")
SEARCH_INTENTS: tuple[str, ...] = ("implementation", "tests", "any")


def _is_count(value: object) -> bool:
    # bool is an int subclass: a JSON `true` coerces to 1 silently otherwise
    return isinstance(value, int) and not isinstance(value, bool)


def _invalid(message: str, violations: list[dict], *, retry_all: bool = True) -> PriorartError:
    return PriorartError(
        INVALID_ARGUMENT,
        message,
        violations=violations,
        next_action=(
            "Correct every listed search argument and retry the call."
            if retry_all
            else "Correct the listed argument and retry the call."
        ),
    )


def validate_search_request(k: int, mode: str, intent: str) -> None:
    """Validate search arguments in one pass, reporting every violation.

    Raises ``INVALID_ARGUMENT`` with an ordered ``violations`` list covering
    ``k``, ``mode`` and ``intent`` together.
    """
    violations: list[dict] = []
    if not _is_count(k) or k < 1:
        violations.append({"field": "k", "input": k, "accepted": ["integer >= 1"]})
    if mode not in SEARCH_MODES:
        violations.append({"field": "mode", "input": mode, "accepted": list(SEARCH_MODES)})
    if intent not in SEARCH_INTENTS:
        violations.append({"field": "intent", "input": intent, "accepted": list(SEARCH_INTENTS)})
    if violations:
        raise _invalid("invalid search arguments", violations)


def validate_map_request(limit: int) -> None:
    """Validate a ``map_symbols`` page request.

    ``limit < 1`` would page forever on an empty result (a zero-row page is
    indistinguishable from an exactly full one) and a negative limit lifts
    SQLite's row cap entirely.
    """
    if not _is_count(limit) or limit < 1:
        raise _invalid(
            "invalid map_symbols arguments",
            [{"field": "limit", "input": limit, "accepted": ["integer >= 1"]}],
            retry_all=False,
        )
