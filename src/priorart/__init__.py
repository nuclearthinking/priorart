"""Local agentic code search: find existing symbols before writing new code.

Layered layout: ``core`` (domain), ``storage`` (per-worktree stores),
``indexing`` (parsing and refresh pipeline), ``retrieval`` (search),
``models`` (embed/expand/rerank clients); ``registry`` composes them, and
``server``/``cli`` are thin adapters. Layers import each other's public
modules (``priorart.<layer>.<module>``), never each other's internals.
"""
