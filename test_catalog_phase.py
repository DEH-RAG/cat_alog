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


    class PointStruct:
        pass


    cat_memory_models.VectorMemoryType = VectorMemoryType
    cat_memory_models.PointStruct = PointStruct
    cat_memory.models = cat_memory_models
    cat_services.memory = cat_memory

    cat_plugins = types.ModuleType("cat.plugins")

    # Minimal `cat.db` surface used by the phase-path ingestion status reader.
    # Tests swap the row maps below; defaults are inert.
    import types as _types

    INGESTION_ROWS = {}
    INGESTION_KEYS = []


    async def _fake_get_ingestion_status_keys(agent_id):
        return list(INGESTION_KEYS)


    async def _fake_crud_read(key):
        return INGESTION_ROWS.get(key)


    cat_db = _types.ModuleType("cat.db")
    cat_db.crud = _types.ModuleType("cat.db.crud")
    cat_db.crud.read = _fake_crud_read
    cat_db_cruds = _types.ModuleType("cat.db.cruds")
    cat_db_settings = _types.ModuleType("cat.db.cruds.settings")
    cat_db_settings.get_ingestion_status_keys = _fake_get_ingestion_status_keys
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
        def __init__(self, agent_files):
            self.agent_files = agent_files

        def list_files(self, path):
            if path == "agent":
                return self.agent_files
            return []


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

    print("test_catalog_phase: ALL ASSERTIONS PASSED")