"""Local agentic code search: find existing symbols before writing new code.

Layered layout: ``core`` (domain), ``storage`` (per-worktree stores),
``indexing`` (parsing and refresh pipeline), ``retrieval`` (search),
``models`` (embed/expand/rerank clients); ``registry`` composes them. The
``coordinator`` owns long-lived registries, jobs, watchers and caches;
``server``/``cli`` are thin lifecycle/rendering adapters. Layers import each
other's public modules (``priorart.<layer>.<module>``), never internals.
"""
