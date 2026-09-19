"""Standalone verification for the CATALOG ``catalog_cards`` ingestion phase.

Covers the three phase-machine hooks added to ``cat_alog.py`` plus the
phase-machine guard on the two ``before_rabbithole_*`` hooks:

- ``ingestion_phase_specs`` appends
  ``PhaseSpec("catalog_cards", None, ("parsing_chunking", "embedding"))``
  preserving prior accumulator entries, and is a no-op when EffING is absent;
- ``ingestion_phase_settings_marker`` returns a SHA-256 fingerprint that
  differs when the card settings change, and ``None`` for unknown phases;
- ``ingestion_phase_run`` returns ``not_ready`` when the source has no stored
  points, ``done`` (with a card point stored: ``is_catalogue_card: True`` +
  ``full_card`` metadata) when it has, ``None`` (fail-hard) for unknown
  phases, and propagates exceptions;
- ``before_rabbithole_splits_documents`` / ``before_rabbithole_stores_documents``
  are no-ops when the phase machine is importable, and keep the legacy
  behavior (TTL cache + card append) when it is not.

Runnable:  python test_catalog_phase.py
Pure-stdlib + unittest.mock, plain asserts, no pytest, no model loading. All
external packages (cat, cat.services.memory.models, langchain_core, pydantic,
cat.plugins.cat_efficient_ingestion.phases) are stubbed in sys.modules
*BEFORE* cat_alog is imported (same pattern as MyGRAPH's
test_ingestion_phase_machine.py). The ENTIRE body lives under
``if __name__ == "__main__":`` so importing this file executes zero code and
requires zero prerequisites (host import-safety for plugin test files).
"""

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import types
    from types import SimpleNamespace
    from unittest.mock import Mock

    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # Stubs (installed BEFORE importing cat_alog)
    # ------------------------------------------------------------------

    class _StubLog:
        def info(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class _StubField:
        def __init__(self, default=None, **kwargs):
            self.default = default

    class _StubBaseModel:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    class _StubPhaseSpec:
        """Mirror of the real PhaseSpec dataclass (positional args)."""

        def __init__(self, id, settings_category=None, depends_on=()):
            self.id = id
            self.settings_category = settings_category
            self.depends_on = depends_on

        def __eq__(self, other):
            return (
                isinstance(other, _StubPhaseSpec)
                and self.id == other.id
                and self.settings_category == other.settings_category
                and self.depends_on == other.depends_on
            )

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = _StubBaseModel
    pydantic_mod.Field = _StubField
    sys.modules["pydantic"] = pydantic_mod

    lc = types.ModuleType("langchain_core")
    lc.__path__ = []
    sys.modules["langchain_core"] = lc
    lc_docs = types.ModuleType("langchain_core.documents")
    lc_docs.__path__ = []
    sys.modules["langchain_core.documents"] = lc_docs
    lc_base = types.ModuleType("langchain_core.documents.base")
    lc_base.Document = _StubBaseModel
    sys.modules["langchain_core.documents.base"] = lc_base

    cat_mod = types.ModuleType("cat")
    cat_mod.__path__ = []
    cat_mod.hook = lambda *a, **k: (lambda f: f)
    cat_mod.plugin = lambda *a, **k: (lambda f: f)
    cat_mod.tool = lambda *a, **k: (lambda f: f)
    cat_mod.log = _StubLog()
    cat_mod.AgenticWorkflowTask = _StubBaseModel
    sys.modules["cat"] = cat_mod

    cat_log = types.ModuleType("cat.log")
    cat_log.log = _StubLog()
    sys.modules["cat.log"] = cat_log

    cat_services = types.ModuleType("cat.services")
    cat_services.__path__ = []
    sys.modules["cat.services"] = cat_services

    cat_memory = types.ModuleType("cat.services.memory")
    cat_memory.__path__ = []
    sys.modules["cat.services.memory"] = cat_memory

    cat_models = types.ModuleType("cat.services.memory.models")
    cat_models.VectorMemoryType = type(
        "VectorMemoryType", (), {"DECLARATIVE": "declarative", "EPISODIC": "episodic"}
    )
    cat_models.PointStruct = _StubBaseModel
    sys.modules["cat.services.memory.models"] = cat_models

    # cat.plugins.cat_efficient_ingestion.phases.PhaseSpec (EffING present)
    cat_plugins = types.ModuleType("cat.plugins")
    cat_plugins.__path__ = []
    sys.modules["cat.plugins"] = cat_plugins
    effing = types.ModuleType("cat.plugins.cat_efficient_ingestion")
    effing.__path__ = []
    sys.modules["cat.plugins.cat_efficient_ingestion"] = effing
    effing_phases = types.ModuleType("cat.plugins.cat_efficient_ingestion.phases")
    effing_phases.PhaseSpec = _StubPhaseSpec
    sys.modules["cat.plugins.cat_efficient_ingestion.phases"] = effing_phases

    sys.path.insert(0, REPO_ROOT)
    import cat_alog  # noqa: E402

    # ------------------------------------------------------------------
    # Fakes
    # ------------------------------------------------------------------

    class _FakePlugin:
        def __init__(self, settings):
            self._settings = settings

        async def load_settings(self):
            return dict(self._settings)

    class _FakeMadHatter:
        def __init__(self, settings):
            self._plugin = _FakePlugin(settings)

        def get_plugin(self):
            return self._plugin

    class _FakeAgenticWorkflow:
        def __init__(self, output="fake summary"):
            self._output = output

        async def run(self, task=None, llm=None):
            return Mock(output=self._output)

    class _FakeEmbedder:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    class _FakeLizard:
        async def embedder(self):
            return _FakeEmbedder()

    class _FakeHandler:
        def __init__(self, records):
            self._records = records
            self.stored = []

        async def get_all_tenant_points(
            self, collection_name, metadata=None, limit=None, offset=None, with_vectors=True
        ):
            return self._records, None

        async def add_points_to_tenant(self, collection_name, points):
            self.stored.extend(points)

    def make_cat(settings, records, summary="fake summary"):
        cat = Mock()
        cat.agent_key = "agent"
        cat.id = "chat"
        cat.mad_hatter = _FakeMadHatter(settings)
        cat.agentic_workflow = _FakeAgenticWorkflow(summary)
        cat.large_language_model = Mock()
        cat.lizard = _FakeLizard()
        handler = _FakeHandler(records)
        cat.vector_memory_handler = handler
        return cat, handler

    def make_record(content, source="src", is_card=False):
        meta = {"source": source, "when": 1.0}
        if is_card:
            meta["is_catalogue_card"] = True
            meta["full_card"] = "# CATALOGUE CARD\n\n## Source file or URL: src"
        return SimpleNamespace(payload={"page_content": content, "metadata": meta})

    SETTINGS = {
        "max_document_chars": 8000,
        "max_summary_words": 200,
        "cache_ttl_seconds": 3600,
        "cache_max_size": 100,
    }

    # ------------------------------------------------------------------
    # 1. ingestion_phase_specs
    # ------------------------------------------------------------------
    cat, _ = make_cat(SETTINGS, [])
    specs = cat_alog.ingestion_phase_specs([], cat)
    assert len(specs) == 1, specs
    spec = specs[0]
    assert spec.id == "catalog_cards", spec
    assert spec.settings_category is None, spec
    assert spec.depends_on == ("parsing_chunking", "embedding"), spec

    # accumulator convention: prior entries preserved
    specs = cat_alog.ingestion_phase_specs(["prior"], cat)
    assert specs == ["prior", spec], specs

    # None-safe accumulator
    specs = cat_alog.ingestion_phase_specs(None, cat)
    assert specs == [spec], specs

    # ------------------------------------------------------------------
    # 2. ingestion_phase_settings_marker
    # ------------------------------------------------------------------
    m1 = asyncio.run(cat_alog.ingestion_phase_settings_marker("catalog_cards", cat))
    assert isinstance(m1, str) and len(m1) == 64, m1

    cat2, _ = make_cat({**SETTINGS, "max_summary_words": 300}, [])
    m2 = asyncio.run(cat_alog.ingestion_phase_settings_marker("catalog_cards", cat2))
    assert m1 != m2, "marker must change when the card settings change"

    cat3, _ = make_cat({**SETTINGS, "cache_max_size": 50}, [])
    m3 = asyncio.run(cat_alog.ingestion_phase_settings_marker("catalog_cards", cat3))
    assert m1 != m3, "marker must change when cache_max_size changes"

    # unknown phase -> None (conservative stale)
    assert asyncio.run(cat_alog.ingestion_phase_settings_marker("graph_building", cat)) is None

    # ------------------------------------------------------------------
    # 3. ingestion_phase_run
    # ------------------------------------------------------------------
    # 3a. no stored points -> not_ready
    cat_empty, _ = make_cat(SETTINGS, [])
    result = asyncio.run(cat_alog.ingestion_phase_run("catalog_cards", "src", [], cat_empty))
    assert result == {"status": "not_ready", "retry_after": 5}, result

    # 3b. stored points -> done + card point stored
    records = [
        make_record("first chunk of the document"),
        make_record("second chunk of the document"),
    ]
    cat_ok, handler = make_cat(SETTINGS, records)
    result = asyncio.run(cat_alog.ingestion_phase_run("catalog_cards", "src", [], cat_ok))
    assert result == {"status": "done"}, result
    assert len(handler.stored) == 1, handler.stored
    point = handler.stored[0]
    assert point.payload["metadata"]["is_catalogue_card"] is True
    assert "full_card" in point.payload["metadata"]
    assert "CATALOGUE CARD" in point.payload["page_content"]
    assert "## Source file or URL: src" in point.payload["page_content"]
    assert "first chunk of the document" in point.payload["page_content"]
    assert "fake summary" in point.payload["page_content"]
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.id == point.id  # deterministic id present

    # 3c. a previously stored card is excluded from the summary input
    records_with_old_card = [
        make_record("first chunk"),
        make_record("old card", is_card=True),
    ]
    cat_ok2, handler2 = make_cat(SETTINGS, records_with_old_card)
    result = asyncio.run(cat_alog.ingestion_phase_run("catalog_cards", "src", [], cat_ok2))
    assert result == {"status": "done"}, result
    assert "old card" not in handler2.stored[0].payload["page_content"]

    # 3d. unknown phase -> None (fail-hard, never silent success)
    assert asyncio.run(cat_alog.ingestion_phase_run("graph_building", "src", [], cat_ok)) is None

    # 3e. exceptions propagate (fail-hard -> ERROR)
    class _Boom:
        async def get_all_tenant_points(self, *a, **k):
            raise RuntimeError("boom")

    cat_bad = make_cat(SETTINGS, [])[0]
    cat_bad.vector_memory_handler = _Boom()
    try:
        asyncio.run(cat_alog.ingestion_phase_run("catalog_cards", "src", [], cat_bad))
    except RuntimeError:
        pass
    else:
        raise AssertionError("ingestion_phase_run must propagate exceptions")

    # ------------------------------------------------------------------
    # 4. before_rabbithole_* hooks: no-ops while the phase machine is active
    # ------------------------------------------------------------------
    docs = [SimpleNamespace(page_content="hello world", metadata={"source": "src"})]
    out = asyncio.run(cat_alog.before_rabbithole_splits_documents(docs, cat_ok))
    assert out is docs, "splits hook must be a no-op when the phase machine is active"
    out2 = cat_alog.before_rabbithole_stores_documents(docs, cat_ok)
    assert out2 is docs, "stores hook must be a no-op when the phase machine is active"

    # ------------------------------------------------------------------
    # 5. EffING absent: specs no-op, hooks keep the legacy behavior
    # ------------------------------------------------------------------
    del sys.modules["cat.plugins.cat_efficient_ingestion.phases"]
    del sys.modules["cat.plugins.cat_efficient_ingestion"]
    del sys.modules["cat.plugins"]

    assert cat_alog._phase_machine_active(cat_ok) is False

    # specs accumulator is a no-op without EffING
    specs = cat_alog.ingestion_phase_specs([], cat_ok)
    assert specs == [], specs

    # legacy splits hook: summary generated and cached
    out = asyncio.run(cat_alog.before_rabbithole_splits_documents(docs, cat_ok))
    assert out is docs
    assert cat_alog.CATALOGUES.get("agent", "chat", "src") == "fake summary"

    # legacy stores hook: card appended
    out2 = cat_alog.before_rabbithole_stores_documents(docs, cat_ok)
    assert len(out2) == 2, len(out2)
    assert out2[1].metadata["is_catalogue_card"] is True
    assert "fake summary" in out2[1].page_content
    assert cat_alog.CATALOGUES.get("agent", "chat", "src") is None  # cache consumed

    print("test_catalog_phase: ALL ASSERTIONS PASSED")