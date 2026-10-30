"""Standalone regression suite for cat_alog (branch feat/prompt-files-list).

Covers the refactored shared collector/formatter and the agent_prompt_suffix
hook: tool byte-regression (incl. empty case), formatter modes (previews /
compact / empty), hook append semantics (separator preserved, no strip),
empty no-op and collector-failure no-op.

Import-safe per plugin rules: the host imports every .py under the plugin
folder, so the ENTIRE body lives inside `if __name__ == "__main__":` and
importing this file executes zero code. Pure stdlib only.
"""
if __name__ == "__main__":
    import asyncio
    import enum
    import sys
    import types
    from datetime import datetime
    from types import SimpleNamespace

    # ── sys.modules stubs for the `cat` package surface ──────────────────
    # Installed BEFORE `import cat_alog` so the module import has zero
    # external prerequisites.

    pydantic_mod = types.ModuleType("pydantic")


    class BaseModel:
        pass


    def Field(*args, **kwargs):
        return None


    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field

    langchain_base = types.ModuleType("langchain_core.documents.base")


    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


    langchain_base.Document = Document

    cat_mod = types.ModuleType("cat")

    log_mod = types.ModuleType("cat.log")
    log_mod.info = lambda *a, **k: None
    log_mod.debug = lambda *a, **k: None
    log_mod.warning = lambda *a, **k: None
    log_mod.error = lambda *a, **k: None
    cat_mod.log = log_mod


    def _deco(*args, **kwargs):
        def wrap(f):
            return f
        return wrap


    cat_mod.hook = _deco
    cat_mod.plugin = _deco
    cat_mod.tool = _deco
    cat_mod.AgenticWorkflowTask = object

    cat_services = types.ModuleType("cat.services")
    cat_memory = types.ModuleType("cat.services.memory")
    cat_memory_models = types.ModuleType("cat.services.memory.models")


    class VectorMemoryType(enum.Enum):
        DECLARATIVE = "declarative"
        EPISODIC = "episodic"

        def __str__(self):
            return self.value


    class PointStruct:
        def __init__(self, id=None, payload=None, vector=None):
            self.id = id
            self.payload = payload or {}
            self.vector = vector


    cat_memory_models.VectorMemoryType = VectorMemoryType
    cat_memory_models.PointStruct = PointStruct
    cat_memory.models = cat_memory_models
    cat_services.memory = cat_memory

    # `cat.services.factory.ingestion.resolved_config_name` (SYSTEM-level
    # ingestion engine resolver used by the phase gate). Mutable so the three
    # truth-table rows can be exercised without re-importing cat_alog.
    RESOLVED_INGESTION = {"name": "EfficientIngestionConfiguration"}

    cat_factory = types.ModuleType("cat.services.factory")
    cat_factory.__path__ = []
    ingestion_factory = types.ModuleType("cat.services.factory.ingestion")

    async def _fake_resolved_config_name(lizard):
        return RESOLVED_INGESTION["name"]

    ingestion_factory.resolved_config_name = _fake_resolved_config_name
    cat_services.factory = cat_factory

    cat_plugins = types.ModuleType("cat.plugins")

    # Minimal `cat.db` surface used by the phase-path ingestion status reader.
    # Tests swap the row maps below; defaults are inert.
    import types as _types

    INGESTION_ROWS = {}
    INGESTION_KEYS = []
    LLM_SETTINGS = {}


    async def _fake_get_ingestion_status_keys(agent_id):
        return list(INGESTION_KEYS)


    async def _fake_crud_read(key):
        return INGESTION_ROWS.get(key)


    async def _fake_get_settings_by_category(key_id, category):
        if category == "llm":
            return LLM_SETTINGS.get(key_id)
        return None


    cat_db = _types.ModuleType("cat.db")
    cat_db.crud = _types.ModuleType("cat.db.crud")
    cat_db.crud.read = _fake_crud_read
    cat_db_cruds = _types.ModuleType("cat.db.cruds")
    cat_db_settings = _types.ModuleType("cat.db.cruds.settings")
    cat_db_settings.get_ingestion_status_keys = _fake_get_ingestion_status_keys
    cat_db_settings.get_settings_by_category = _fake_get_settings_by_category
    cat_db_cruds.settings = cat_db_settings

    sys.modules["pydantic"] = pydantic_mod
    sys.modules["langchain_core"] = types.ModuleType("langchain_core")
    sys.modules["langchain_core.documents"] = types.ModuleType("langchain_core.documents")
    sys.modules["langchain_core.documents.base"] = langchain_base
    sys.modules["cat"] = cat_mod
    sys.modules["cat.log"] = log_mod
    sys.modules["cat.services"] = cat_services
    sys.modules["cat.services.memory"] = cat_memory
    sys.modules["cat.services.memory.models"] = cat_memory_models
    sys.modules["cat.services.factory"] = cat_factory
    sys.modules["cat.services.factory.ingestion"] = ingestion_factory
    sys.modules["cat.plugins"] = cat_plugins
    sys.modules["cat.db"] = cat_db
    sys.modules["cat.db.crud"] = cat_db.crud
    sys.modules["cat.db.cruds"] = cat_db_cruds
    sys.modules["cat.db.cruds.settings"] = cat_db_settings

    sys.path.insert(0, "/root/git/plugins/cat_alog")
    import cat_alog  # noqa: E402

    # ── Fake cat surface ──────────────────────────────────────────────────
    CARD_TEXT = (
        "# CATALOGUE CARD\n\n"
        "## Source file or URL: report.pdf\n\n"
        "## Initial page\n\n"
        "abstract\n\n"
        "## Summary\n\n"
        "fake summary\n"
    )


    class FakeVectorMemoryHandler:
        """Returns the configured records for every get_all_tenant_points call."""

        def __init__(self, records):
            self.records = records
            self.calls = 0

        async def get_all_tenant_points(self, collection_name, metadata=None,
                                        limit=None, offset=None, with_vectors=True):
            self.calls += 1
            return list(self.records), None


    class FakeFileManager:
        def __init__(self, agent_files, files=None):
            self.agent_files = agent_files
            self.files = files or {}

        def list_files(self, path):
            if path == "agent":
                return self.agent_files
            return []

        def read_file(self, remote_filename, remote_root_dir=None):
            if (remote_root_dir, remote_filename) in self.files:
                return self.files[(remote_root_dir, remote_filename)]
            return self.files.get(remote_filename)


    class FakePlugin:
        def __init__(self, settings):
            self._settings = settings

        async def load_settings(self):
            return dict(self._settings)


    class FakePluginManager:
        def __init__(self, settings):
            self._settings = settings

        def get_plugin(self):
            return FakePlugin(self._settings)


    class FakePhaseHandler:
        """Collection-aware fake for the phase: separate chunks per collection."""

        def __init__(self, by_collection):
            self.by_collection = by_collection
            self.added = []

        async def get_all_tenant_points(self, collection_name, metadata=None,
                                        limit=None, offset=None, with_vectors=True):
            return list(self.by_collection.get(collection_name, [])), None

        async def add_points_to_tenant(self, collection_name, points):
            self.added.append((collection_name, list(points)))


    class FakeCat:
        def __init__(self, file_manager, handler, settings=None):
            self.agent_key = "agent"
            self.id = "chat"
            self.file_manager = file_manager
            self.vector_memory_handler = handler
            self.plugin_manager = FakePluginManager(settings or {})


    def build_cat(with_file=True, settings=None):
        agent_files = (
            [SimpleNamespace(name="report.pdf", last_modified=1234567890.0)]
            if with_file
            else []
        )
        card_payload = {
            "page_content": CARD_TEXT,
            "metadata": {
                "source": "report.pdf",
                "is_catalogue_card": True,
                "full_card": CARD_TEXT,
            },
        }
        records = [SimpleNamespace(payload=card_payload)] if with_file else []
        return FakeCat(
            FakeFileManager(agent_files), FakeVectorMemoryHandler(records), settings
        )


    cat = build_cat(with_file=True)
    empty_cat = build_cat(with_file=False)

    # ── (a) Tool regression: full equality, not substring ─────────────────
    expected_tool = (
        "# Loaded Files\n"
        + "\n"
        + "## FILE NAME: report.pdf\n"
        + "\n"
        + "- **Type**: Agent"
        + "\n"
        + "- **Uploaded**: 1234567890.0\n"
        + "\n"
        + "**Content preview:**\n"
        + "\n"
        + "```\n"
        + "fake summary...\n"
        + "```\n"
    )
    assert cat_alog.list_loaded_files(cat) == expected_tool, "(a) tool regression mismatch"

    # ── (b) Formatter empty cases ─────────────────────────────────────────
    assert cat_alog._format_loaded_files({}, with_previews=True) == (
        "# Loaded Files\n\nNo files loaded in this conversation or agent."
    ), "(b) empty previews mismatch"
    assert cat_alog._format_loaded_files({}, with_previews=False) == (
        "# Loaded files"
    ), "(b) empty compact mismatch"

    # ── (c) Compact formatter: date computed, never a literal ─────────────
    compact = cat_alog._format_loaded_files(
        {
            "report.pdf": {
                "source": "report.pdf",
                "when": 1234567890.0,
                "content": None,
                "type": "agent",
            }
        },
        with_previews=False,
    )
    expected_compact = (
        "# Loaded files"
        + "\n"
        + f"- **report.pdf** (Agent, uploaded {datetime.fromtimestamp(1234567890).strftime('%Y-%m-%d %H:%M')})"
    )
    assert compact == expected_compact, "(c) compact formatter mismatch"
    assert "**Content preview:**" not in compact, "(c) compact must not contain previews"

    # ── (d) Hook append: leading "\n\n" preserved, no strip ───────────────
    out_d = asyncio.run(cat_alog.agent_prompt_suffix("", cat))
    assert out_d == "\n\n" + compact, "(d) hook append mismatch"

    # ── (e) Hook with existing suffix ─────────────────────────────────────
    out_e = asyncio.run(cat_alog.agent_prompt_suffix("EXISTING SUFFIX", cat))
    assert out_e == "EXISTING SUFFIX\n\n" + compact, "(e) hook existing-suffix mismatch"

    # ── (f) Empty no-op ───────────────────────────────────────────────────
    out_f = asyncio.run(cat_alog.agent_prompt_suffix("x", empty_cat))
    assert out_f == "x", "(f) empty no-op mismatch"

    # ── (g) Raising collector: hook swallows and returns suffix unchanged ─
    original_collect = cat_alog._collect_loaded_files
    try:
        async def _boom(cat):
            raise RuntimeError("boom")

        cat_alog._collect_loaded_files = _boom
        out_g = asyncio.run(cat_alog.agent_prompt_suffix("x", cat))
        assert out_g == "x", "(g) raising-collector no-op mismatch"
    finally:
        cat_alog._collect_loaded_files = original_collect

    # ── Tool empty output via the tool itself ─────────────────────────────
    assert cat_alog.list_loaded_files(empty_cat) == (
        "# Loaded Files\n\nNo files loaded in this conversation or agent."
    ), "tool empty output mismatch"

    # ── (h) Hook escapes braces in file names (core does system_prompt.format) ─
    brace_cat = build_cat(with_file=True)
    brace_cat.file_manager.agent_files = [
        SimpleNamespace(name="report{2024}.pdf", last_modified=1234567890.0)
    ]
    out_h = asyncio.run(cat_alog.agent_prompt_suffix("", brace_cat))
    assert "report{{2024}}.pdf" in out_h, "(h) braces not escaped in hook output"
    _ = "{}".format(out_h)  # must not raise on the escaped listing

    # ── (i) Compact formatter: falsy when (0.0) -> "unknown" ──────────────
    compact_i = cat_alog._format_loaded_files(
        {
            "zero.pdf": {
                "source": "zero.pdf",
                "when": 0.0,
                "content": None,
                "type": "agent",
            }
        },
        with_previews=False,
    )
    assert "uploaded unknown" in compact_i, "(i) falsy when not normalized to unknown"

    # ── (j) Setting OFF: prompt untouched and NO collection / DB / vector call ─
    off_cat = build_cat(
        with_file=True, settings={"append_file_list_to_prompt": False}
    )
    original_collect_j = cat_alog._collect_loaded_files

    async def _must_not_collect(cat):
        raise AssertionError("collector must not run when the setting is off")

    try:
        cat_alog._collect_loaded_files = _must_not_collect
        out_j = asyncio.run(cat_alog.agent_prompt_suffix("KEEP", off_cat))
    finally:
        cat_alog._collect_loaded_files = original_collect_j
    assert out_j == "KEEP", "(j) setting off must leave the prompt untouched"
    assert off_cat.vector_memory_handler.calls == 0, "(j) no vector query when off"

    # ── (k) Setting ON + phase path: list comes from ingestion status rows ──
    INGESTION_KEYS[:] = [
        "agents:agent:ingestion:agent:aaa",
        "agents:agent:ingestion:agent:bbb",
        "agents:agent:ingestion:agent:aaa-dup",
        "agents:agent:ingestion:delete",
        "agents:agent:ingestion:chat:zzz",
    ]
    INGESTION_ROWS.clear()
    INGESTION_ROWS.update({
        "agents:agent:ingestion:agent:aaa": {
            "source": "lesson1.pdf", "scope": "agent", "type": "file",
            "status": "completed", "updated_at": 1234567890.0,
        },
        "agents:agent:ingestion:agent:bbb": {
            "source": "https://example.test/slides", "scope": "agent", "type": "url",
            "status": "completed", "updated_at": 1234567890.0,
        },
        # Same source again (e.g. re-ingested row): must NOT duplicate.
        "agents:agent:ingestion:agent:aaa-dup": {
            "source": "lesson1.pdf", "scope": "agent", "type": "file",
            "status": "error", "updated_at": 1234567890.0,
        },
        "agents:agent:ingestion:delete": {"delete_marker": True},
        "agents:agent:ingestion:chat:zzz": {
            "source": "chatfile.pdf", "scope": "chat", "type": "file",
            "status": "completed", "updated_at": 1234567890.0,
        },
    })
    phase_cat = build_cat(with_file=True)
    original_phase_k = cat_alog._phase_machine_active

    async def _phase_on(cat=None):
        return True

    try:
        cat_alog._phase_machine_active = _phase_on
        out_k = asyncio.run(cat_alog.agent_prompt_suffix("x", phase_cat))
    finally:
        cat_alog._phase_machine_active = original_phase_k
    assert "lesson1.pdf" in out_k, "(k) status source missing from list"
    assert out_k.count("lesson1.pdf") == 1, "(k) duplicate source listed twice"
    assert "https://example.test/slides" in out_k, "(k) URL source missing"
    assert "chatfile.pdf" not in out_k, "(k) non-AGENT-scope row leaked"
    assert "delete" not in out_k.lower(), "(k) deletion marker listed as a file"
    assert phase_cat.vector_memory_handler.calls == 0, (
        "(k) phase path must not run the classic vector collector"
    )

    # ── (l) Extracted image never listed; whole-image source stays ──────────
    image_records = [
        SimpleNamespace(payload={"metadata": {
            "source": "original.pdf", "image": True,
            "image_file": "original_img_0_abcd.jpg",
        }}),
        SimpleNamespace(payload={"metadata": {
            "source": "photo.jpg", "image": True,
            "image_file": "photo.jpg",
        }}),
    ]
    img_cat = FakeCat(
        FakeFileManager([
            SimpleNamespace(name="original.pdf", last_modified=1.0),
            SimpleNamespace(name="original_img_0_abcd.jpg", last_modified=1.0),
            SimpleNamespace(name="photo.jpg", last_modified=1.0),
        ]),
        FakeVectorMemoryHandler(image_records),
        {},
    )
    original_phase_l = cat_alog._phase_machine_active

    async def _phase_off(cat=None):
        return False

    try:
        cat_alog._phase_machine_active = _phase_off
        out_l = asyncio.run(cat_alog.agent_prompt_suffix("x", img_cat))
    finally:
        cat_alog._phase_machine_active = original_phase_l
    assert "original_img_0_abcd.jpg" not in out_l, (
        "(l) image extracted from a file was listed"
    )
    assert "original.pdf" in out_l, "(l) the source file itself disappeared"
    assert "photo.jpg" in out_l, "(l) an image that IS the source must stay"

    # ── (m) Deterministic card id shared by the phase and the classic hook ──
    classic_cat = build_cat(with_file=True)
    cat_alog.CATALOGUES.set("agent", classic_cat.id, "report.pdf", "S")
    out_m = asyncio.run(cat_alog.before_rabbithole_stores_documents(
        [Document(page_content="chunk text", metadata={"source": "report.pdf"})],
        classic_cat,
    ))
    cards_m = [d for d in out_m if d.metadata.get("is_catalogue_card") is True]
    assert len(cards_m) == 1, "(m) classic card missing"
    assert cards_m[0].id == cat_alog._catalogue_card_point_id("agent", classic_cat.id, "report.pdf"), (
        "(m) classic card does not use the shared deterministic id"
    )

    # Load the REAL token-budget splitter under the canonical core-plugin path
    # so (n) exercises the same segmentation the classic store path uses.
    import importlib.util

    cat_core = types.ModuleType("cat.core_plugins")
    cat_core.__path__ = []
    effing_pkg = types.ModuleType("cat.core_plugins.cat_efficient_ingestion")
    effing_pkg.__path__ = []
    sys.modules["cat.core_plugins"] = cat_core
    sys.modules["cat.core_plugins.cat_efficient_ingestion"] = effing_pkg
    _split_spec = importlib.util.spec_from_file_location(
        "cat.core_plugins.cat_efficient_ingestion.split",
        "/root/git/plugins/cat-efficient-ingestion/split.py",
    )
    sys.modules["langchain_core.documents"].Document = Document
    log_mod.log = log_mod
    _split_mod = importlib.util.module_from_spec(_split_spec)
    sys.modules[_split_spec.name] = _split_mod
    _split_spec.loader.exec_module(_split_mod)

    DEC = str(VectorMemoryType.DECLARATIVE)
    EPS = str(VectorMemoryType.EPISODIC)

    class FakeEmbedder:
        def __init__(self, max_tokens=20):
            self.max_input_tokens = max_tokens

        def _estimate_tokens(self, text):
            return max(1, len(text.split()))

        def embed_documents(self, texts):
            return [[float(len(text))] for text in texts]

    async def _fake_summary_agent(cat, prompt):
        return " ".join(f"word{i}" for i in range(300))

    async def _fake_resolve_embedder(cat):
        return FakeEmbedder(20)

    original_summary = cat_alog._run_summary_agent
    original_embedder = cat_alog._resolve_embedder
    cat_alog._run_summary_agent = _fake_summary_agent
    cat_alog._resolve_embedder = _fake_resolve_embedder
    try:
        # ── (n) EPISODIC scope, start_index ordering, segmented card, ids ──
        chunk_late = SimpleNamespace(payload={
            "page_content": "LATE CHUNK",
            "metadata": {"source": "chat.pdf", "chat_id": "chat1", "start_index": 50},
        })
        chunk_first = SimpleNamespace(payload={
            "page_content": "FIRST CHUNK",
            "metadata": {"source": "chat.pdf", "chat_id": "chat1", "start_index": 0},
        })
        phase_handler = FakePhaseHandler({EPS: [chunk_late, chunk_first]})
        phase_cat = FakeCat(FakeFileManager([]), phase_handler, {})

        result_n = asyncio.run(
            cat_alog.ingestion_phase_run("catalog_cards", "chat.pdf", {}, phase_cat)
        )
        assert result_n == {"status": "done"}, "(n) phase did not finish"
        assert len(phase_handler.added) == 1, "(n) expected one upsert batch"
        collection_n, points_n = phase_handler.added[0]
        assert collection_n == EPS, "(n) chat-scoped card must land in EPISODIC"
        assert len(points_n) > 1, "(n) oversized card was not segmented"
        assert all(
            p.payload["metadata"].get("is_catalogue_card") is True for p in points_n
        ), "(n) every segment must carry the card marker"
        assert all(p.payload["metadata"].get("full_card") for p in points_n), (
            "(n) every segment must carry full_card"
        )
        assert all(p.payload["metadata"].get("chat_id") == "chat1" for p in points_n), (
            "(n) EPISODIC chat_id lost on the card"
        )
        assert "FIRST CHUNK" in points_n[0].payload["metadata"]["full_card"], (
            "(n) initial page not taken from the smallest start_index chunk"
        )
        assert points_n[0].id == cat_alog._catalogue_card_point_id("agent", "chat1", "chat.pdf"), (
            "(n) first segment must use the shared deterministic id"
        )
        assert all(
            p.payload.get("id") == p.id for p in points_n
        ), "(n) every card segment payload must carry its id (re-embed identity)"
        ids_n = [p.id for p in points_n]
        assert len(set(ids_n)) == len(ids_n), "(n) duplicate segment ids"
        assert "uuid4" not in repr(ids_n)

        phase_handler.added.clear()
        asyncio.run(
            cat_alog.ingestion_phase_run("catalog_cards", "chat.pdf", {}, phase_cat)
        )
        assert [p.id for p in phase_handler.added[0][1]] == ids_n, (
            "(n) re-run must upsert the same ids"
        )

        # ── (o) FILE source: initial page from the saved bytes (DECLARATIVE) ──
        file_manager_o = FakeFileManager(
            [], files={("agent", "lesson.txt"): b"PAGE ONE CONTENT " * 20}
        )
        handler_o = FakePhaseHandler({DEC: [SimpleNamespace(payload={
            "page_content": "CHUNK BODY",
            "metadata": {"source": "lesson.txt", "start_index": 0},
        })]})
        cat_o = FakeCat(file_manager_o, handler_o, {})

        async def _short_summary(cat, prompt):
            return "SHORT SUMMARY"

        cat_alog._run_summary_agent = _short_summary

        async def _no_limit_embedder(cat):
            return FakeEmbedder(None)

        cat_alog._resolve_embedder = _no_limit_embedder
        result_o = asyncio.run(
            cat_alog.ingestion_phase_run("catalog_cards", "lesson.txt", {}, cat_o)
        )
        assert result_o == {"status": "done"}, "(o) phase did not finish"
        collection_o, points_o = handler_o.added[0]
        assert collection_o == DEC, "(o) agent-scoped card must land in DECLARATIVE"
        full_card_o = points_o[0].payload["metadata"]["full_card"]
        assert "PAGE ONE CONTENT" in full_card_o, (
            "(o) initial page not read from the source file bytes"
        )
        assert points_o[0].id == cat_alog._catalogue_card_point_id("agent", None, "lesson.txt")
        assert points_o[0].payload.get("id") == points_o[0].id, (
            "(o) agent-scoped card payload must carry its id"
        )

        # ── (p) Marker: LLM + sizes + prompt; cache tuning is NOT material ──
        LLM_SETTINGS["agent"] = {"name": "LLMConfig", "value": {"model": "gpt-x"}}
        marker_cat = build_cat(settings={})
        m1 = asyncio.run(
            cat_alog.ingestion_phase_settings_marker("catalog_cards", marker_cat)
        )
        cache_changed = build_cat(settings={
            "cache_ttl_seconds": 1, "cache_max_size": 1,
        })
        m2 = asyncio.run(
            cat_alog.ingestion_phase_settings_marker("catalog_cards", cache_changed)
        )
        assert m1 == m2, "(p) cache tuning must not change the marker"
        words_changed = build_cat(settings={"max_summary_words": 201})
        m3 = asyncio.run(
            cat_alog.ingestion_phase_settings_marker("catalog_cards", words_changed)
        )
        assert m3 != m1, "(p) max_summary_words must change the marker"
        LLM_SETTINGS["agent"] = {"name": "LLMConfig", "value": {"model": "gpt-y"}}
        m4 = asyncio.run(
            cat_alog.ingestion_phase_settings_marker("catalog_cards", marker_cat)
        )
        assert m4 != m1, "(p) agent LLM change must change the marker"
        assert asyncio.run(
            cat_alog.ingestion_phase_settings_marker("other_phase", marker_cat)
        ) is None, "(p) unknown phases must yield a conservative None marker"
    finally:
        cat_alog._run_summary_agent = original_summary
        cat_alog._resolve_embedder = original_embedder

    # ── (q) Truth table: EffING importable × SYSTEM engine -> gate ─────────
    # Restored after 0a8633f deleted them. `_phase_machine_active` imports
    # `cat.core_plugins.cat_efficient_ingestion.phases` lazily and resolves the
    # SYSTEM ingestion engine; both are stubbed, so the three rows are pure.
    phases_mod = types.ModuleType("cat.core_plugins.cat_efficient_ingestion.phases")
    cat_core_phases_parent = sys.modules["cat.core_plugins.cat_efficient_ingestion"]
    gate_cat = SimpleNamespace()

    # (q1) EffING absent -> classic, regardless of the engine setting.
    RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"
    sys.modules.pop("cat.core_plugins.cat_efficient_ingestion.phases", None)
    assert asyncio.run(cat_alog._phase_machine_active(gate_cat)) is False, (
        "(q1) absent EffING must keep the classic path"
    )

    # (q2) EffING present + CORE engine -> classic.
    sys.modules["cat.core_plugins.cat_efficient_ingestion.phases"] = phases_mod
    cat_core_phases_parent.phases = phases_mod
    RESOLVED_INGESTION["name"] = "CoreIngestionConfiguration"
    assert asyncio.run(cat_alog._phase_machine_active(gate_cat)) is False, (
        "(q2) core engine must keep the classic path"
    )

    # (q3) EffING present + EFFICIENT engine -> phase path active.
    RESOLVED_INGESTION["name"] = "EfficientIngestionConfiguration"
    assert asyncio.run(cat_alog._phase_machine_active(gate_cat)) is True, (
        "(q3) efficient engine must activate the phase path"
    )
    # Back to the classic path for the id-scoping checks below.
    sys.modules.pop("cat.core_plugins.cat_efficient_ingestion.phases", None)

    # ── (r) Tenant-scoped id: same source in two agents -> two distinct cards ─
    def _classic_card(cat, source):
        cat_alog.CATALOGUES.set(cat.agent_key, cat.id, source, "S")
        out = asyncio.run(cat_alog.before_rabbithole_stores_documents(
            [Document(page_content="chunk", metadata={"source": source})], cat,
        ))
        return [d for d in out if d.metadata.get("is_catalogue_card") is True][0]

    agent_a = build_cat(with_file=True)
    agent_a.agent_key = "agentA"
    agent_b = build_cat(with_file=True)
    agent_b.agent_key = "agentB"
    card_a = _classic_card(agent_a, "same.pdf")
    card_b = _classic_card(agent_b, "same.pdf")
    assert card_a.id != card_b.id, "(r) same source in two agents must not collide"
    assert card_a.id == cat_alog._catalogue_card_point_id("agentA", agent_a.id, "same.pdf")
    assert card_b.id == cat_alog._catalogue_card_point_id("agentB", agent_b.id, "same.pdf")

    # ── (s) Chat-scoped id: same source in two chats of one agent -> two cards ─
    chat_1 = build_cat(with_file=True)
    chat_1.id = "chatA"
    chat_2 = build_cat(with_file=True)
    chat_2.id = "chatB"
    card_c1 = _classic_card(chat_1, "same.pdf")
    card_c2 = _classic_card(chat_2, "same.pdf")
    assert card_c1.id != card_c2.id, "(s) same source in two chats must not collide"

    print("test_catalog_phase: ALL ASSERTIONS PASSED")