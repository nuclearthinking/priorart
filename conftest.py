"""Project pytest configuration."""

from __future__ import annotations

import sys


def _disable_gremlins_lightweight_runner() -> None:
    # pytest-gremlins 1.9.0 ships a "lightweight" per-mutant test runner that
    # imports test modules and calls test functions directly, without pytest.
    # Fixture-based tests (tmp_path, monkeypatch) then raise TypeError on call
    # and every mutant is misreported as zapped, including mutants in code no
    # test executes. Force the real pytest subprocess instead.
    replacement = lambda command, env: None  # noqa: E731
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith("pytest_gremlins") and hasattr(module, "build_lightweight_command"):
            module.build_lightweight_command = replacement


_disable_gremlins_lightweight_runner()
