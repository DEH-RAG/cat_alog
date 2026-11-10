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
    import importlib.util
    import os
    import sys
    import types
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace

    #: The plugin root: this file lives in ``tests/`` after the move.
    REPO_ROOT = Path(__file__).resolve().parent.parent

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


    cat_mod.__path__ = []
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
    sys.modules["cat.plugins"] = cat_plugins
    sys.modules["cat.db"] = cat_db
    sys.modules["cat.db.crud"] = cat_db.crud
    sys.modules["cat.db.cruds"] = cat_db_cruds
    sys.modules["cat.db.cruds.settings"] = cat_db_settings

    # The core ingestion protocol is the source of truth: load the REAL
    # `cat/rabbit_hole/{phases,split}.py` from the host checkout instead of
    # duplicating the contract here. Only the couple of host modules they
    # import are stubbed above.
    def _core_file(relative):
        candidates = []
        core_root = os.environ.get("CAT_CORE_ROOT")
        if core_root:
            candidates.append(Path(core_root))
        candidates.extend([Path.cwd(), Path("/root/git/grinning-cat-core")])
        for root in candidates:
            candidate = root / relative
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            f"cat_alog test: cannot find the core file {relative!r}; "
            "run from the grinning-cat-core checkout or set CAT_CORE_ROOT"
        )

    def _load_core_module(name, relative):
        spec = importlib.util.spec_from_file_location(name, _core_file(relative))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    cat_db_database = _types.ModuleType("cat.db.database")
    cat_db_database.DEFAULT_SYSTEM_KEY = "system:agent"
    sys.modules["cat.db.database"] = cat_db_database

    cat_utils = _types.ModuleType("cat.utils")
    cat_utils.SUFFIX_TO_CRYPT = ()
    sys.modules["cat.utils"] = cat_utils

    cat_env = _types.ModuleType("cat.env")
    cat_env.get_env_int = lambda name: None
    sys.modules["cat.env"] = cat_env

    sys.modules["langchain_core.documents"].Document = Document
    log_mod.log = log_mod

    rabbit_hole = types.ModuleType("cat.rabbit_hole")
    rabbit_hole.__path__ = []
    sys.modules["cat.rabbit_hole"] = rabbit_hole

    _load_core_module("cat.rabbit_hole.phases", "cat/rabbit_hole/phases.py")
    _load_core_module("cat.rabbit_hole.split", "cat/rabbit_hole/split.py")

    # `read_source_bytes` returns the bytes the host kept for a source (the
    # uploaded file or the RabbitHole copy); the fake file manager below
    # implements the same contract.
    chunks_mod = types.ModuleType("cat.rabbit_hole.chunks")

    def _fake_read_source_bytes(file_manager, agent_id, chat_id, source):
        folder = os.path.join(agent_id, str(chat_id)) if chat_id else agent_id
        if not str(source).startswith(("http://", "https://")):
            data = file_manager.read_file(source, folder)
            if data is not None:
                return data
        import hashlib as _hashlib

        return file_manager.read_file(
            _hashlib.sha256(str(source).encode()).hexdigest() + ".source", folder
        )

    chunks_mod.read_source_bytes = _fake_read_source_bytes
    sys.modules["cat.rabbit_hole.chunks"] = chunks_mod

    sys.path.insert(0, str(REPO_ROOT))
    try:
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
            """Collection-aware fake for the phase: separate chunks per collection.

            ``add_points_to_tenant`` stores what it receives (like the real upsert)
            so the replace-on-re-run behaviour (delete old card points, then write
            the new ones) is exercised against a realistic store.
            """

            def __init__(self, by_collection):
                self.by_collection = {name: list(records) for name, records in by_collection.items()}
                self.added = []
                self.deleted = []

            async def get_all_tenant_points(self, collection_name, metadata=None,
                                            limit=None, offset=None, with_vectors=True):
                records = self.by_collection.get(collection_name, [])
                if metadata:
                    records = [
                        r for r in records
                        if all(
                            ((r.payload or {}).get("metadata", {}) or {}).get(k) == v
                            for k, v in metadata.items()
                        )
                    ]
                return list(records), None

            async def add_points_to_tenant(self, collection_name, points):
                points = list(points)
                self.added.append((collection_name, points))
                store = self.by_collection.setdefault(collection_name, [])
                for point in points:
                    store.append(SimpleNamespace(id=point.id, payload=point.payload))

            async def delete_tenant_points_by_ids(self, collection_name, points_ids):
                ids = set(points_ids)
                self.deleted.append((collection_name, ids))
                store = self.by_collection.get(collection_name, [])
                self.by_collection[collection_name] = [
                    r for r in store if getattr(r, "id", None) not in ids
                ]


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
        INGESTION_KEYS[:] = ["agents:agent:ingestion:agent:report"]
        INGESTION_ROWS.clear()
        INGESTION_ROWS["agents:agent:ingestion:agent:report"] = {
            "source": "report.pdf", "scope": "agent", "type": "file",
            "status": "completed", "updated_at": 1234567890.0,
        }
        out_d = asyncio.run(cat_alog.agent_prompt_suffix("", cat))
        assert out_d == "\n\n" + compact, "(d) hook append mismatch"

        # ── (e) Hook with existing suffix ─────────────────────────────────────
        out_e = asyncio.run(cat_alog.agent_prompt_suffix("EXISTING SUFFIX", cat))
        assert out_e == "EXISTING SUFFIX\n\n" + compact, "(e) hook existing-suffix mismatch"

        # ── (f) Empty no-op ───────────────────────────────────────────────────
        INGESTION_KEYS[:] = []
        INGESTION_ROWS.clear()
        out_f = asyncio.run(cat_alog.agent_prompt_suffix("x", empty_cat))
        assert out_f == "x", "(f) empty no-op mismatch"

        # ── (g) Raising collector: hook swallows and returns suffix unchanged ─
        original_collect = cat_alog._collect_ingestion_sources
        try:
            async def _boom(cat):
                raise RuntimeError("boom")

            cat_alog._collect_ingestion_sources = _boom
            out_g = asyncio.run(cat_alog.agent_prompt_suffix("x", cat))
            assert out_g == "x", "(g) raising-collector no-op mismatch"
        finally:
            cat_alog._collect_ingestion_sources = original_collect

        # ── Tool empty output via the tool itself ─────────────────────────────
        assert cat_alog.list_loaded_files(empty_cat) == (
            "# Loaded Files\n\nNo files loaded in this conversation or agent."
        ), "tool empty output mismatch"

        # ── (h) Hook escapes braces in file names (core does system_prompt.format) ─
        INGESTION_KEYS[:] = ["agents:agent:ingestion:agent:brace"]
        INGESTION_ROWS.clear()
        INGESTION_ROWS["agents:agent:ingestion:agent:brace"] = {
            "source": "report{2024}.pdf", "scope": "agent", "type": "file",
            "status": "completed", "updated_at": 1234567890.0,
        }
        out_h = asyncio.run(cat_alog.agent_prompt_suffix("", cat))
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
        original_collect_j = cat_alog._collect_ingestion_sources

        async def _must_not_collect(cat):
            raise AssertionError("collector must not run when the setting is off")

        try:
            cat_alog._collect_ingestion_sources = _must_not_collect
            out_j = asyncio.run(cat_alog.agent_prompt_suffix("KEEP", off_cat))
        finally:
            cat_alog._collect_ingestion_sources = original_collect_j
        assert out_j == "KEEP", "(j) setting off must leave the prompt untouched"
        assert off_cat.vector_memory_handler.calls == 0, "(j) no vector query when off"

        # ── (k) Setting ON: the ingestion status registry is the source list ──
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
        out_k = asyncio.run(cat_alog.agent_prompt_suffix("x", phase_cat))
        assert "lesson1.pdf" in out_k, "(k) status source missing from list"
        assert out_k.count("lesson1.pdf") == 1, "(k) duplicate source listed twice"
        assert "https://example.test/slides" in out_k, "(k) URL source missing"
        assert "chatfile.pdf" not in out_k, "(k) non-AGENT-scope row leaked"
        assert "delete" not in out_k.lower(), "(k) deletion marker listed as a file"
        assert phase_cat.vector_memory_handler.calls == 0, (
            "(k) the status path must not run the vector collector"
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
        out_l = cat_alog.list_loaded_files(img_cat)
        assert "original_img_0_abcd.jpg" not in out_l, (
            "(l) image extracted from a file was listed"
        )
        assert "original.pdf" in out_l, "(l) the source file itself disappeared"
        assert "photo.jpg" in out_l, "(l) an image that IS the source must stay"

        # ── (m) Deterministic card id: stable, tenant- and chat-scoped ─────────
        id_m = cat_alog._catalogue_card_point_id("agent", "chat", "report.pdf")
        assert id_m == cat_alog._catalogue_card_point_id("agent", "chat", "report.pdf"), (
            "(m) the card id must be deterministic"
        )
        assert id_m != cat_alog._catalogue_card_point_id("agent", "other", "report.pdf"), (
            "(m) the same source in two chats must not collide"
        )
        assert id_m != cat_alog._catalogue_card_point_id("other", "chat", "report.pdf"), (
            "(m) the same source in two agents must not collide"
        )

        # The testkit already installed EffING's real token-budget splitter under
        # the canonical core-plugin path, so (n) exercises the same segmentation
        # the classic store path uses.

        DEC = str(VectorMemoryType.DECLARATIVE)
        EPS = str(VectorMemoryType.EPISODIC)

        def _source(name, cat, chat_id=None):
            return cat_alog.IngestionSource(
                agent_id=cat.agent_key, chat_id=chat_id, name=name,
            )

        def _run_phase(name, cat, chat_id=None):
            return asyncio.run(
                cat_alog.CatalogCardsPhase().process(_source(name, cat, chat_id), cat)
            )

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

            result_n = _run_phase("chat.pdf", phase_cat)
            assert result_n.status == "done", "(n) phase did not finish"
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
            _run_phase("chat.pdf", phase_cat)
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
            result_o = _run_phase("lesson.txt", cat_o)
            assert result_o.status == "done", "(o) phase did not finish"
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

            # ── (t) B1: same source in two chats -> two isolated cards ──────────
            chunk_alpha = SimpleNamespace(payload={
                "page_content": "ALPHA CHUNK",
                "metadata": {"source": "same.pdf", "chat_id": "chatA", "start_index": 0},
            })
            chunk_beta = SimpleNamespace(payload={
                "page_content": "BETA CHUNK",
                "metadata": {"source": "same.pdf", "chat_id": "chatB", "start_index": 0},
            })
            handler_t = FakePhaseHandler({EPS: [chunk_alpha, chunk_beta]})
            fm_t = FakeFileManager([], files={
                ("agent/chatA", "same.pdf"): b"ALPHA PAGE ONLY",
                ("agent/chatB", "same.pdf"): b"BETA PAGE ONLY",
            })
            cat_t = FakeCat(fm_t, handler_t, {})

            async def _echo_summary(cat, prompt):
                return prompt

            cat_alog._run_summary_agent = _echo_summary
            cat_alog._resolve_embedder = _fake_resolve_embedder
            result_t = _run_phase("same.pdf", cat_t)
            assert result_t.status == "done", "(t) phase did not finish"
            assert len(handler_t.added) == 2, (
                "(t) two chats of one source must produce two card writes"
            )
            cards_by_chat = {}
            for collection_t, points_t in handler_t.added:
                assert collection_t == EPS, "(t) chat-scoped card must land in EPISODIC"
                cards_by_chat[points_t[0].payload["metadata"]["chat_id"]] = points_t
            assert set(cards_by_chat) == {"chatA", "chatB"}, cards_by_chat.keys()
            full_alpha = cards_by_chat["chatA"][0].payload["metadata"]["full_card"]
            full_beta = cards_by_chat["chatB"][0].payload["metadata"]["full_card"]
            assert "ALPHA" in full_alpha and "BETA" not in full_alpha, (
                "(t) chat A card contains chat B text"
            )
            assert "BETA" in full_beta and "ALPHA" not in full_beta, (
                "(t) chat B card contains chat A text"
            )
            assert cards_by_chat["chatA"][0].id != cards_by_chat["chatB"][0].id, (
                "(t) the two chat cards must have distinct ids"
            )

            # ── (u) B2: oversized card segments are marked, id from start_index ──
            handler_u = FakePhaseHandler({DEC: [SimpleNamespace(payload={
                "page_content": "SMALL CHUNK",
                "metadata": {"source": "big.txt", "start_index": 0},
            })]})
            fm_u = FakeFileManager([], files={("agent", "big.txt"): b"PAGE " * 400})
            cat_u = FakeCat(fm_u, handler_u, {})
            cat_alog._resolve_embedder = _fake_resolve_embedder
            result_u = _run_phase("big.txt", cat_u)
            assert result_u.status == "done", "(u) phase did not finish"
            _, points_u = handler_u.added[0]
            assert len(points_u) > 1, "(u) oversized card was not segmented"
            for point_u in points_u:
                meta_u = point_u.payload["metadata"]
                assert meta_u.get("is_catalogue_card_segment") is True, (
                    "(u) card segment must be marked so a re-embed never re-splits it"
                )
                assert isinstance(meta_u.get("start_index"), int), (
                    "(u) card segment must carry a stable start_index"
                )
                assert point_u.id == cat_alog._catalogue_card_point_id(
                    "agent", None, "big.txt", start_index=meta_u["start_index"]
                ), "(u) card segment id must derive from start_index"
            ids_u = [p.id for p in points_u]
            assert len(set(ids_u)) == len(ids_u), "(u) duplicate segment ids"

            handler_u.added.clear()
            _run_phase("big.txt", cat_u)
            assert [p.id for p in handler_u.added[0][1]] == ids_u, (
                "(u) a re-run must reproduce the same segment ids"
            )

            # ── (v) F2: a changed embedder window REPLACES the old segments ──────
            # Same source, same chat: a lower/higher window yields a different
            # segment count, so the phase must delete the source's old card points
            # instead of leaving them next to the new ones. Other sources and other
            # chats must stay untouched.
            class FakeWindowEmbedder(FakeEmbedder):
                def __init__(self, max_tokens):
                    super().__init__(max_tokens)

            window_embedder = FakeWindowEmbedder(37)

            async def _window_embedder(cat):
                return window_embedder

            cat_alog._resolve_embedder = _window_embedder
            handler_v = FakePhaseHandler({
                DEC: [
                    SimpleNamespace(payload={
                        "page_content": "WIN CHUNK",
                        "metadata": {"source": "win.txt", "start_index": 0},
                    }),
                    SimpleNamespace(payload={
                        "page_content": "OTHER CHUNK",
                        "metadata": {"source": "other.txt", "start_index": 0},
                    }),
                ],
                EPS: [SimpleNamespace(payload={
                    "page_content": "CHAT CHUNK",
                    "metadata": {"source": "win.txt", "chat_id": "chatZ", "start_index": 0},
                })],
            })
            fm_v = FakeFileManager([], files={
                ("agent", "win.txt"): b"WIN PAGE " * 400,
                ("agent", "other.txt"): b"OTHER PAGE " * 20,
                ("agent/chatZ", "win.txt"): b"CHAT PAGE " * 200,
            })
            cat_v = FakeCat(fm_v, handler_v, {})
            cat_alog._run_summary_agent = _fake_summary_agent

            def _cards_v(collection):
                return [
                    r for r in handler_v.by_collection.get(collection, [])
                    if ((r.payload or {}).get("metadata", {}) or {}).get("is_catalogue_card") is True
                ]

            def _card_ids_v(collection, chat_id, source="win.txt"):
                return sorted(
                    r.id for r in _cards_v(collection)
                    if ((r.payload or {}).get("metadata", {}) or {}).get("chat_id") == chat_id
                    and ((r.payload or {}).get("metadata", {}) or {}).get("source") == source
                )

            # A third source's card, produced before the window changes.
            _run_phase("other.txt", cat_v)
            other_ids_before = _card_ids_v(DEC, None, source="other.txt")
            assert other_ids_before, "(v) other source card was not produced"

            def _run_win():
                handler_v.added.clear()
                handler_v.deleted.clear()
                result = _run_phase("win.txt", cat_v)
                assert result.status == "done", "(v) phase did not finish"
                return {collection: list(points) for collection, points in handler_v.added}

            first = _run_win()
            first_dec_ids = sorted(p.id for p in first[DEC])
            first_eps_ids = sorted(p.id for p in first[EPS])
            assert len(first_dec_ids) > 1, "(v) window 37 did not segment the card"

            # LOWER the window -> MORE segments, all ids new.
            window_embedder.max_input_tokens = 23
            second = _run_win()
            second_dec_ids = sorted(p.id for p in second[DEC])
            second_eps_ids = sorted(p.id for p in second[EPS])
            assert len(second_dec_ids) > len(first_dec_ids), (
                "(v) lowering the window must produce more segments",
                len(first_dec_ids), len(second_dec_ids),
            )
            # start_index 0 keeps the classic (suffix-less) id by design; every
            # other old segment id must be gone.
            base_dec_id = cat_alog._catalogue_card_point_id("agent", None, "win.txt")
            assert base_dec_id in second_dec_ids
            assert set(first_dec_ids) - set(second_dec_ids), (
                "(v) lower window kept the old segment ids"
            )
            assert _card_ids_v(DEC, None) == second_dec_ids, (
                "(v) old agent-scoped segments survived the lower-window re-run",
                _card_ids_v(DEC, None), second_dec_ids,
            )
            assert _card_ids_v(EPS, "chatZ") == second_eps_ids, (
                "(v) old chat-scoped segments survived the lower-window re-run"
            )
            assert _card_ids_v(DEC, None) and _card_ids_v(EPS, "chatZ"), (
                "(v) a re-run must not delete everything"
            )

            # RAISE the window -> FEWER segments (one whole card), old ones gone.
            window_embedder.max_input_tokens = 3000
            third = _run_win()
            third_dec_ids = sorted(p.id for p in third[DEC])
            third_eps_ids = sorted(p.id for p in third[EPS])
            assert len(third_dec_ids) < len(second_dec_ids), (
                "(v) raising the window must produce fewer segments"
            )
            assert _card_ids_v(DEC, None) == third_dec_ids, (
                "(v) old agent-scoped segments survived the higher-window re-run"
            )
            assert _card_ids_v(EPS, "chatZ") == third_eps_ids, (
                "(v) old chat-scoped segments survived the higher-window re-run"
            )
            assert base_dec_id in third_dec_ids, (
                "(v) the start_index-0 segment must keep the classic id"
            )
            assert set(second_dec_ids) - set(third_dec_ids), (
                "(v) higher window kept the old segment ids"
            )

            # Other sources are untouched by every re-run of win.txt.
            assert sorted(
                r.id for r in _cards_v(DEC)
                if ((r.payload or {}).get("metadata", {}) or {}).get("source") == "other.txt"
            ) == other_ids_before, "(v) a different source's card was deleted/replaced"

            # ── (p) settings(): LLM + sizes + prompt; unrelated tuning is not material ──
            LLM_SETTINGS["agent"] = {"name": "LLMConfig", "value": {"model": "gpt-x"}}
            marker_cat = build_cat(settings={})
            m1 = asyncio.run(cat_alog.CatalogCardsPhase().settings(marker_cat))
            unrelated = build_cat(settings={"append_file_list_to_prompt": False})
            m2 = asyncio.run(cat_alog.CatalogCardsPhase().settings(unrelated))
            assert m1 == m2, "(p) unrelated tuning must not change the settings"
            words_changed = build_cat(settings={"max_summary_words": 201})
            m3 = asyncio.run(cat_alog.CatalogCardsPhase().settings(words_changed))
            assert m3 != m1, "(p) max_summary_words must change the settings"
            LLM_SETTINGS["agent"] = {"name": "LLMConfig", "value": {"model": "gpt-y"}}
            m4 = asyncio.run(cat_alog.CatalogCardsPhase().settings(marker_cat))
            assert m4 != m1, "(p) agent LLM change must change the settings"
            assert m1["prompt"] == cat_alog._SUMMARY_PROMPT_TEMPLATE, (
                "(p) the summary prompt must be part of the material settings"
            )
        finally:
            cat_alog._run_summary_agent = original_summary
            cat_alog._resolve_embedder = original_embedder

        # ── (q) Registration: ONE hook, phase CLASS, no mutation, no instance ───
        gate_cat = build_cat(with_file=True)

        # (q1) the hook appends the class (the host instantiates it per run).
        registered = cat_alog.rabbithole_ingestion_phases([], gate_cat)
        assert registered == [cat_alog.CatalogCardsPhase], (
            "(q1) the hook must append the CatalogCardsPhase class"
        )

        # (q2) the contribution is a core-protocol phase subclass.
        assert isinstance(registered[0], type) and issubclass(
            registered[0], cat_alog.IngestionPhase
        ), "(q2) the registered entry must be an IngestionPhase subclass"

        # (q3) the hook returns a new list and never mutates the caller's one.
        base_phases = []
        out_q = cat_alog.rabbithole_ingestion_phases(base_phases, gate_cat)
        assert out_q is not base_phases and base_phases == [], (
            "(q3) the hook must not mutate the caller's phase list"
        )
        assert out_q == [cat_alog.CatalogCardsPhase], "(q3) exactly one phase is contributed"

        # ── (r) Tenant-scoped id: same source in two agents -> two distinct cards ─
        agent_a = build_cat(with_file=True)
        agent_a.agent_key = "agentA"
        agent_b = build_cat(with_file=True)
        agent_b.agent_key = "agentB"
        card_a_id = cat_alog._catalogue_card_point_id("agentA", agent_a.id, "same.pdf")
        card_b_id = cat_alog._catalogue_card_point_id("agentB", agent_b.id, "same.pdf")
        assert card_a_id != card_b_id, "(r) same source in two agents must not collide"
        assert card_a_id == cat_alog._catalogue_card_point_id("agentA", agent_a.id, "same.pdf")
        assert card_b_id == cat_alog._catalogue_card_point_id("agentB", agent_b.id, "same.pdf")

        # ── (s) Chat-scoped id: same source in two chats of one agent -> two cards ─
        chat_1 = build_cat(with_file=True)
        chat_1.id = "chatA"
        chat_2 = build_cat(with_file=True)
        chat_2.id = "chatB"
        card_c1_id = cat_alog._catalogue_card_point_id("agent", chat_1.id, "same.pdf")
        card_c2_id = cat_alog._catalogue_card_point_id("agent", chat_2.id, "same.pdf")
        assert card_c1_id != card_c2_id, "(s) same source in two chats must not collide"

        # ── (u) Phase failure -> status dict, never the phase id ─────────────
        class NotFoundError(Exception):
            pass

        async def _raise_not_found(cat, prompt):
            raise NotFoundError("model gpt-x not found (404)")

        handler_u = FakePhaseHandler({DEC: [SimpleNamespace(payload={
            "page_content": "CHUNK BODY",
            "metadata": {"source": "bad.pdf", "start_index": 0},
        })]})
        cat_u = FakeCat(FakeFileManager([]), handler_u, {})

        async def _embedder_u(cat):
            return FakeEmbedder(None)

        original_summary_u = cat_alog._run_summary_agent
        original_embedder_u = cat_alog._resolve_embedder
        original_log_error_u = cat_alog.log.error
        logged_u = []
        cat_alog._run_summary_agent = _raise_not_found
        cat_alog._resolve_embedder = _embedder_u
        cat_alog.log.error = lambda msg, *a, **k: logged_u.append(str(msg))
        try:
            result_u = _run_phase("bad.pdf", cat_u)
        finally:
            cat_alog._run_summary_agent = original_summary_u
            cat_alog._resolve_embedder = original_embedder_u
            cat_alog.log.error = original_log_error_u

        assert isinstance(result_u, cat_alog.PhaseResult), (
            "(u) failure must return a core PhaseResult"
        )
        assert result_u.status == "error", "(u) permanent failure must be error"
        assert result_u.reason, "(u) error status must carry a reason"
        assert "NotFoundError" in result_u.reason, "(u) reason must name the exception"
        assert result_u != "catalog_cards", "(u) phase id leaked as the result"
        assert logged_u, "(u) phase failure was not logged"
        assert "catalog_cards" in logged_u[0], "(u) log line must carry the phase id"
        assert "NotFoundError" in logged_u[0], "(u) log line must carry the exception class"
        assert len(logged_u) == 1, "(u) phase failure must be logged exactly once"
        assert not handler_u.added, "(u) a failed phase must not write a card"

        # ── (w) Empty-output policy: a phase that RAN with chunks but produced
        # ZERO cards is DONE + a warning, never an error. "Could not run" keeps
        # its not_ready/error path.
        class ZeroEmbedder(FakeEmbedder):
            def embed_documents(self, texts):
                return []

        async def _zero_embedder(cat):
            return ZeroEmbedder(None)

        def _chunk(source, text="SOURCE BODY", chat_id=None, start_index=0):
            metadata = {"source": source, "start_index": start_index}
            if chat_id is not None:
                metadata["chat_id"] = chat_id
            return SimpleNamespace(payload={"page_content": text, "metadata": metadata})

        original_summary_w = cat_alog._run_summary_agent
        original_embedder_w = cat_alog._resolve_embedder
        try:
            cat_alog._run_summary_agent = _fake_summary_agent
            cat_alog._resolve_embedder = _zero_embedder

            # (w-a) Ran with chunks, produced 0 cards -> done + warning, no error.
            handler_wa = FakePhaseHandler({DEC: [_chunk("empty.pdf")]})
            cat_wa = FakeCat(FakeFileManager([]), handler_wa, {})
            result_wa = _run_phase("empty.pdf", cat_wa)
            assert result_wa.status == "done", (
                "(w-a) an empty extraction must be done, not error/not_ready"
            )
            assert result_wa.reason is None, "(w-a) a warning result must not carry a reason"
            warnings_wa = result_wa.warnings
            assert isinstance(warnings_wa, list) and len(warnings_wa) == 1, (
                "(w-a) exactly one warning must be attached", warnings_wa
            )
            warning_wa = warnings_wa[0]
            assert warning_wa.get("code") == "empty-extraction", warning_wa
            assert "'empty.pdf'" in warning_wa.get("message", ""), warning_wa
            assert "1 chunk(s)" in warning_wa.get("message", ""), warning_wa
            assert result_wa.processed == 0, (
                "(w-a) processed must report the 0 cards produced"
            )

            # The phase object carries the same warning on its result.
            phase_wa = cat_alog.CatalogCardsPhase()
            result_wa_obj = asyncio.run(
                phase_wa.process(_source("empty.pdf", cat_wa), cat_wa)
            )
            assert result_wa_obj.status == "done", "(w-a) phase object must be done"
            assert result_wa_obj.warnings == warnings_wa, (
                "(w-a) the phase object must carry the warning"
            )
            assert result_wa_obj.reason is None, "(w-a) a warning is not a reason"

            # (w-b) Chunks missing -> still not_ready (could not run).
            handler_wb = FakePhaseHandler({})
            cat_wb = FakeCat(FakeFileManager([]), handler_wb, {})
            result_wb = asyncio.run(
                cat_alog._run_catalog_cards_phase("missing.pdf", cat_wb)
            )
            assert result_wb.status == "not_ready" and result_wb.retry_after == 5, (
                "(w-b) a source without chunks must stay not_ready", result_wb
            )
            phase_wb = cat_alog.CatalogCardsPhase()
            assert asyncio.run(
                phase_wb.process(_source("missing.pdf", cat_wb), cat_wb)
            ).status == "not_ready", (
                "(w-b) the phase object must stay not_ready without chunks"
            )

            # (w-c) LLM failure -> still the current error path, no warning.
            handler_wc = FakePhaseHandler({DEC: [_chunk("bad.pdf")]})
            cat_wc = FakeCat(FakeFileManager([]), handler_wc, {})
            cat_alog._run_summary_agent = _raise_not_found
            cat_alog._resolve_embedder = _embedder_u
            result_wc = _run_phase("bad.pdf", cat_wc)
            assert result_wc.status == "error", (
                "(w-c) an LLM failure must stay an error", result_wc
            )
            assert "NotFoundError" in (result_wc.reason or ""), result_wc
            assert not result_wc.warnings, "(w-c) an error must carry no warning"
            assert not handler_wc.added, "(w-c) a failed phase must not write a card"

            # (w-d) done + warnings is COMPLETE: no retry signal is produced, so
            # the machine does not retry and dependents are not blocked.
            cat_alog._run_summary_agent = _fake_summary_agent
            cat_alog._resolve_embedder = _zero_embedder
            result_wd = asyncio.run(
                phase_wa.process(_source("empty.pdf", cat_wa), cat_wa)
            )
            assert result_wd.status == "done", (
                "(w-d) done is the completion signal for a warned phase"
            )
            assert result_wd.warnings == warnings_wa, result_wd
            assert result_wd.retry_after is None, (
                "(w-d) a warned phase must not request a retry", result_wd
            )
            assert result_wd.reason is None, result_wd
        finally:
            cat_alog._run_summary_agent = original_summary_w
            cat_alog._resolve_embedder = original_embedder_w

        # ── Phase ABC migration tests ──────────────────────────────────────────
        assert cat_alog.CatalogCardsPhase is not None, (
            "CatalogCardsPhase must be defined"
        )
        assert issubclass(cat_alog.CatalogCardsPhase, cat_alog.IngestionPhase), (
            "CatalogCardsPhase must subclass the core IngestionPhase"
        )

        # (abc-a) applies_to is a CAPABILITY check only: the catalogue has no
        # per-agent enable flag, so the phase always applies — no engine gate is
        # re-derived inside the phase class.
        catalogue_off = build_cat(with_file=True)
        phase_a = cat_alog.CatalogCardsPhase()
        applies_off, reason_off = asyncio.run(
            phase_a.applies_to(_source("anything.pdf", catalogue_off), catalogue_off)
        )
        assert applies_off is True and reason_off is None, (
            "(abc-a) applies_to must be a capability check with no reason"
        )
        catalogue_on = build_cat(
            with_file=True, settings={"append_file_list_to_prompt": True}
        )
        applies_on, reason_on = asyncio.run(
            phase_a.applies_to(_source("anything.pdf", catalogue_on), catalogue_on)
        )
        assert applies_on is True and reason_on is None, (
            "(abc-a) the catalogue capability applies without a reason"
        )
        registered = cat_alog.rabbithole_ingestion_phases([], catalogue_off)
        assert registered == [cat_alog.CatalogCardsPhase], (
            "(abc-a) the phase must be registered through rabbithole_ingestion_phases"
        )

        # (abc-a2) The phase class must NOT own the ingestion protocol: no diary,
        # staleness, lock, retry or status-derivation helper may be referenced.
        import inspect as _inspect

        class_source = _inspect.getsource(cat_alog.CatalogCardsPhase)
        for forbidden in (
            "record_phase_",
            "should_abandon",
            "distributed_lock",
            "derive_source_status",
            "_phase_machine_active",
        ):
            assert forbidden not in class_source, (
                f"(abc-a2) phase class must not reference protocol helper {forbidden!r}"
            )

        # (abc-b) delete() removes ONLY the source's cards and is idempotent.
        del_dec_id = cat_alog._catalogue_card_point_id("agent", "chat", "del.pdf")
        keep_dec_id = cat_alog._catalogue_card_point_id("agent", "chat", "keep.pdf")
        del_chunk_id = "stale-card-on-chunk"
        del_eps_id = cat_alog._catalogue_card_point_id("agent", "chat", "del.pdf")
        keep_eps_id = cat_alog._catalogue_card_point_id("agent", "chat", "keep.pdf")
        handler_b = FakePhaseHandler({
            DEC: [
                SimpleNamespace(
                    id=del_dec_id,
                    payload={"metadata": {
                        "source": "del.pdf", "is_catalogue_card": True,
                        "chat_id": "chat",
                    }},
                ),
                SimpleNamespace(
                    id=keep_dec_id,
                    payload={"metadata": {
                        "source": "keep.pdf", "is_catalogue_card": True,
                        "chat_id": "chat",
                    }},
                ),
                SimpleNamespace(
                    id=del_chunk_id,
                    payload={"metadata": {
                        "source": "del.pdf", "chunk_index": 0,
                        "is_catalogue_card": True, "full_card": "a stale card body",
                    }},
                ),
            ],
            EPS: [
                SimpleNamespace(
                    id=del_eps_id,
                    payload={"metadata": {
                        "source": "del.pdf", "is_catalogue_card": True,
                        "chat_id": "chat",
                    }},
                ),
                SimpleNamespace(
                    id=keep_eps_id,
                    payload={"metadata": {
                        "source": "keep.pdf", "is_catalogue_card": True,
                        "chat_id": "chat",
                    }},
                ),
            ],
        })
        cat_b = FakeCat(FakeFileManager([]), handler_b, {})
        phase_b = cat_alog.CatalogCardsPhase()
        asyncio.run(phase_b.delete(_source("del.pdf", cat_b), cat_b))
        remaining_dec = [r.id for r in handler_b.by_collection.get(DEC, [])]
        remaining_eps = [r.id for r in handler_b.by_collection.get(EPS, [])]
        assert del_dec_id not in remaining_dec and del_eps_id not in remaining_eps, (
            "(abc-b) delete must remove the source's cards in both collections"
        )
        assert del_chunk_id in remaining_dec, (
            "(abc-b) delete must not remove a chunk that merely carries the card flags"
        )
        assert keep_dec_id in remaining_dec and keep_eps_id in remaining_eps, (
            "(abc-b) delete must not touch another source's cards"
        )
        deletes_after_first = len(handler_b.deleted)
        asyncio.run(phase_b.delete(_source("del.pdf", cat_b), cat_b))
        assert len(handler_b.deleted) == deletes_after_first, (
            "(abc-b) a second delete must be a no-op (idempotent)"
        )
        asyncio.run(phase_b.delete(_source("never-existed.pdf", cat_b), cat_b))
        assert len(handler_b.deleted) == deletes_after_first, (
            "(abc-b) delete of an absent source must be safe and write nothing"
        )

        # (abc-c) settings() is stable, changes with the config, ignores unrelated tuning.
        LLM_SETTINGS["agent"] = {"name": "LLMConfig", "value": {"model": "gpt-x"}}
        tag_base = build_cat(settings={"max_summary_words": 200})
        tag_phase = cat_alog.CatalogCardsPhase()
        t1 = asyncio.run(tag_phase.settings(tag_base))
        t2 = asyncio.run(tag_phase.settings(tag_base))
        assert t1 == t2, "(abc-c) an unchanged configuration must keep the same settings"
        tag_cache = cat_alog.CatalogCardsPhase()
        assert asyncio.run(tag_cache.settings(build_cat(settings={
            "max_summary_words": 200, "append_file_list_to_prompt": False,
        }))) == t1, "(abc-c) the prompt-list switch must not change the settings"
        tag_words = cat_alog.CatalogCardsPhase()
        assert asyncio.run(tag_words.settings(build_cat(settings={
            "max_summary_words": 201,
        }))) != t1, "(abc-c) a catalogue config change must change the settings"

        # (abc-d) process() delegates to the existing card-generation path.
        delegate_calls = []

        async def _fake_cards_path(source, cat):
            delegate_calls.append((source, cat))
            return cat_alog.PhaseResult(status="done")

        original_cards_path = cat_alog._run_catalog_cards_phase
        cat_alog._run_catalog_cards_phase = _fake_cards_path
        try:
            result_d = asyncio.run(
                phase_b.process(_source("delegate.pdf", cat_b), cat_b)
            )
        finally:
            cat_alog._run_catalog_cards_phase = original_cards_path
        assert delegate_calls == [("delegate.pdf", cat_b)], (
            "(abc-d) process must delegate to _run_catalog_cards_phase"
        )
        assert result_d.status == "done", "(abc-d) a done result must stay done"

        # (v) GraphRAG-shaped payload: a REAL source chunk that carries stale
        # catalogue-card metadata (is_catalogue_card/full_card/is_catalogue_card_
        # segment). The loader must NOT treat it as a card: the flag alone is not
        # authoritative, and dropping it made the phase return not_ready forever.
        full_card_v = (
            "# CATALOGUE CARD\n\n## Source file or URL: graph.pdf\n\n"
            "## Initial page\n\n%PDF-1.3 binary payload\n\n## Summary\n\nold summary\n"
        )
        source_text_v = "SOURCE BODY TEXT " * 40
        chunk_v = SimpleNamespace(id="graphrag-chunk", payload={
            "page_content": source_text_v,
            "metadata": {
                "source": "graph.pdf",
                "chat_id": None,
                "chunk_index": 0,
                "start_index": 39735,
                "source_chunks": [{"text": "SOURCE BODY", "metadata": {"source": "graph.pdf"}}],
                "is_catalogue_card": True,
                "is_catalogue_card_segment": True,
                "full_card": full_card_v,
            },
        })
        handler_v = FakePhaseHandler({DEC: [chunk_v], EPS: []})
        cat_v = FakeCat(
            FakeFileManager([], files={("agent", "graph.pdf"): b"%PDF-1.3 bytes"}),
            handler_v,
            {},
        )

        groups_v = asyncio.run(
            cat_alog._load_source_chunk_groups(handler_v, "graph.pdf", cat_v.agent_key)
        )
        assert [(g[0], g[1], len(g[2])) for g in groups_v] == [(DEC, None, 1)], (
            "(v) a GraphRAG chunk flagged with stale card metadata must still form a group"
        )
        assert "is_catalogue_card" not in chunk_v.payload["metadata"], (
            "(v) stale card keys must be stripped from the chunk before building the card"
        )

        # (v1) A plain chunk (chunk fields only) must yield exactly one group and
        # must be left untouched by the loader.
        clean_chunk = SimpleNamespace(id="clean-chunk", payload={
            "page_content": "CLEAN SOURCE TEXT " * 5,
            "metadata": {"source": "clean.pdf", "chunk_index": 0, "has_formula": False},
        })
        clean_meta = dict(clean_chunk.payload["metadata"])
        handler_clean = FakePhaseHandler({DEC: [clean_chunk], EPS: []})
        groups_clean = asyncio.run(
            cat_alog._load_source_chunk_groups(handler_clean, "clean.pdf", "agent")
        )
        assert [(g[0], g[1], len(g[2])) for g in groups_clean] == [(DEC, None, 1)], (
            "(v1) a plain chunk must form exactly one group"
        )
        assert clean_chunk.payload["metadata"] == clean_meta, (
            "(v1) the loader must not mutate a plain chunk's metadata"
        )

        # (v2) A GENUINE card (its content is a slice of its own full_card) must be
        # excluded: the flag is authoritative when the content really is the card.
        genuine_full = (
            "# CATALOGUE CARD\n\n## Source file or URL: card.pdf\n\n"
            "## Initial page\n\nabstract\n\n## Summary\n\nold card summary\n"
        )
        genuine_card = SimpleNamespace(
            id=cat_alog._catalogue_card_point_id("agent", None, "card.pdf"),
            payload={
                "page_content": "## Summary\n\nold card summary",
                "metadata": {
                    "source": "card.pdf",
                    "is_catalogue_card": True,
                    "full_card": genuine_full,
                },
            },
        )
        handler_card = FakePhaseHandler({DEC: [genuine_card], EPS: []})
        groups_card = asyncio.run(
            cat_alog._load_source_chunk_groups(handler_card, "card.pdf", "agent")
        )
        assert groups_card == [], "(v2) a genuine catalogue card must be excluded"
        assert genuine_card.payload["metadata"]["is_catalogue_card"] is True, (
            "(v2) excluding a card must not strip the card's own marker"
        )

        async def _short_summary_v(cat, prompt):
            return "SHORT SUMMARY"

        cat_alog._run_summary_agent = _short_summary_v
        cat_alog._resolve_embedder = _fake_resolve_embedder
        result_v = _run_phase("graph.pdf", cat_v)
        assert result_v.status == "done", (
            "(v) the phase must complete instead of returning not_ready"
        )
        surviving_ids_v = [r.id for r in handler_v.by_collection.get(DEC, [])]
        assert "graphrag-chunk" in surviving_ids_v, (
            "(v) the real source chunk must not be purged as if it were a card"
        )
        _, points_v = handler_v.added[-1]
        assert points_v[0].payload["metadata"].get("is_catalogue_card") is True, (
            "(v) the newly generated card must still be marked as a card"
        )
        assert points_v[0].payload["metadata"].get("full_card"), (
            "(v) the newly generated card must still carry full_card"
        )
        assert points_v[0].id == cat_alog._catalogue_card_point_id(
            "agent", None, "graph.pdf"
        ), "(v) the new card must use the deterministic id"

        # (v3) The card write must not mutate the chunk: the chunk keeps its own
        # fields and gains NO card keys, while the card is a DISTINCT point (its
        # deterministic id, not the chunk's id).
        chunk_meta_after = chunk_v.payload["metadata"]
        assert chunk_meta_after.get("chunk_index") == 0, (
            "(v3) the card write dropped the chunk's chunk_index"
        )
        assert chunk_meta_after.get("source_chunks"), (
            "(v3) the card write dropped the chunk's source_chunks"
        )
        assert not any(key in chunk_meta_after for key in (
            "is_catalogue_card", "is_catalogue_card_segment", "full_card",
        )), "(v3) the card write must not add card keys to the chunk"
        assert points_v[0].id != chunk_v.id, (
            "(v3) the card must not reuse the chunk's point id"
        )
        assert "graphrag-chunk" in surviving_ids_v and points_v[0].id in surviving_ids_v, (
            "(v3) the store must hold the chunk AND the card as distinct points"
        )

        # (w-a) LIVE REGRESSION: a single-chunk source whose chunk carries
        # card flags + full_card but whose id is NOT the deterministic card id.
        # Identity, not content, decides: it must be returned as a source chunk.
        full_card_w = (
            "# CATALOGUE CARD\n\n## Source file or URL: test_oop_verify.txt\n\n"
            "## Initial page\n\ntext\n\n## Summary\n\nsummary\n"
        )
        chunk_w = SimpleNamespace(id="chunk-not-card", payload={
            "page_content": full_card_w,
            "metadata": {
                "source": "test_oop_verify.txt",
                "chunk_index": 0,
                "is_catalogue_card": True,
                "is_catalogue_card_segment": True,
                "full_card": full_card_w,
            },
        })
        handler_w = FakePhaseHandler({DEC: [chunk_w], EPS: []})
        groups_w = asyncio.run(cat_alog._load_source_chunk_groups(
            handler_w, "test_oop_verify.txt", "agent",
        ))
        assert [(g[0], g[1], len(g[2])) for g in groups_w] == [(DEC, None, 1)], (
            "(w-a) a chunk carrying card flags at a non-card id must remain a chunk"
        )
        assert "is_catalogue_card" not in chunk_w.payload["metadata"], (
            "(w-a) the card-only keys must be stripped from the chunk"
        )

        # (w-b) A record whose id IS the deterministic card id is a card and is
        # excluded from the source chunks.
        card_id_wb = cat_alog._catalogue_card_point_id(
            "agent", None, "identity.pdf",
        )
        card_wb = SimpleNamespace(id=card_id_wb, payload={
            "page_content": "## Summary\n\nanything at all",
            "metadata": {
                "source": "identity.pdf",
                "chunk_index": 0,
                "is_catalogue_card": True,
            },
        })
        handler_wb = FakePhaseHandler({DEC: [card_wb], EPS: []})
        groups_wb = asyncio.run(cat_alog._load_source_chunk_groups(
            handler_wb, "identity.pdf", "agent",
        ))
        assert groups_wb == [], "(w-b) a point at the card id must be excluded"

        # (w-c) A segmented card: every segment id derives from the same source
        # with its own :s<start_index> suffix and must be excluded.
        seg_records = [
            SimpleNamespace(
                id=cat_alog._catalogue_card_point_id(
                    "agent", None, "seg.pdf", index, start_index=offset,
                ),
                payload={
                    "page_content": f"segment {offset}",
                    "metadata": {
                        "source": "seg.pdf",
                        "chunk_index": 0,
                        "is_catalogue_card": True,
                        "is_catalogue_card_segment": True,
                        "start_index": offset,
                    },
                },
            )
            for index, offset in enumerate((0, 1200, 2600))
        ]
        handler_wc = FakePhaseHandler({DEC: seg_records, EPS: []})
        groups_wc = asyncio.run(cat_alog._load_source_chunk_groups(
            handler_wc, "seg.pdf", "agent",
        ))
        assert groups_wc == [], "(w-c) every segment of a card must be excluded"

        # (w-d) The purge paths delete ONLY records at a card id, never a chunk
        # that merely carries the flags.
        base_purge_id = cat_alog._catalogue_card_point_id("agent", "chat", "purge.pdf")
        seg_purge_id = cat_alog._catalogue_card_point_id(
            "agent", "chat", "purge.pdf", 1, start_index=900,
        )
        flagged_chunk_id = "flagged-chunk-not-card"
        purge_records = [
            SimpleNamespace(id=base_purge_id, payload={"metadata": {
                "source": "purge.pdf", "chat_id": "chat", "is_catalogue_card": True,
            }}),
            SimpleNamespace(id=seg_purge_id, payload={"metadata": {
                "source": "purge.pdf", "chat_id": "chat", "chunk_index": 0,
                "start_index": 900, "is_catalogue_card": True,
                "is_catalogue_card_segment": True,
            }}),
            SimpleNamespace(id=flagged_chunk_id, payload={
                "page_content": "real source body",
                "metadata": {
                    "source": "purge.pdf", "chat_id": "chat", "chunk_index": 0,
                    "is_catalogue_card": True, "full_card": "stale card",
                },
            }),
        ]
        handler_wd = FakePhaseHandler({DEC: list(purge_records), EPS: []})
        cat_wd = FakeCat(FakeFileManager([]), handler_wd, {})
        removed_wd = asyncio.run(
            cat_alog._delete_source_catalogue_cards(cat_wd, "purge.pdf")
        )
        remaining_wd = {r.id for r in handler_wd.by_collection[DEC]}
        assert removed_wd == 2, "(w-d) delete must remove exactly the two card points"
        assert base_purge_id not in remaining_wd and seg_purge_id not in remaining_wd, (
            "(w-d) delete must remove every real card segment"
        )
        assert flagged_chunk_id in remaining_wd, (
            "(w-d) delete must never remove a chunk that merely carries the flags"
        )

        handler_wd2 = FakePhaseHandler({DEC: list(purge_records), EPS: []})
        purged_wd2 = asyncio.run(cat_alog._purge_existing_cards(
            handler_wd2, DEC, "purge.pdf", "chat", "test", "agent",
        ))
        remaining_wd2 = {r.id for r in handler_wd2.by_collection[DEC]}
        assert purged_wd2 == 2, "(w-d) replace must purge exactly the two card points"
        assert flagged_chunk_id in remaining_wd2, (
            "(w-d) replace must never purge a chunk that merely carries the flags"
        )

        # ── (x) Source classification + parsed-text initial page ─────────────
        assert cat_alog._is_textual_source("notes.md") is True
        assert cat_alog._is_textual_source("notes.txt") is True
        assert cat_alog._is_textual_source("page.html") is True
        assert cat_alog._is_textual_source("rows.csv") is True
        assert cat_alog._is_textual_source("report.pdf") is False
        assert cat_alog._is_textual_source("slides.pptx") is False
        assert cat_alog._is_textual_source("sheet.xlsx") is False
        assert cat_alog._is_textual_source("scan.png") is False
        assert cat_alog._is_textual_source("https://example.org/report.PDF?x=1") is False, (
            "(x) a PDF URL must be binary (query string ignored)"
        )
        assert cat_alog._is_textual_source("https://example.org/notes.txt") is True
        assert cat_alog._is_textual_source("unknown.weird") is True, (
            "(x) an unknown extension keeps the textual default"
        )

        original_summary_x = cat_alog._run_summary_agent
        original_embedder_x = cat_alog._resolve_embedder

        captured_prompts = []

        async def _summary_x(cat, prompt):
            captured_prompts.append(prompt)
            return "CARD SUMMARY"

        async def _embedder_x(cat):
            return FakeEmbedder(100000)

        cat_alog._run_summary_agent = _summary_x
        cat_alog._resolve_embedder = _embedder_x
        try:
            # (a) PDF source: the card's initial page is the PARSED text, never the
            # raw bytes (no ``%PDF`` header, no replacement-char garbage).
            parsed_pdf = "PARSED PROSE about the motor and the mission. " * 8
            pdf_bytes = (
                b"%PDF-1.5\n1 0 obj\n<< /Type /Catalog >>\nstream\n"
                b"\x00\xff\xfe binary %PDF garbage \xff\xfe\nendstream\n"
            )
            chunk_pdf = SimpleNamespace(payload={
                "page_content": parsed_pdf,
                "metadata": {"source": "doc.pdf", "chunk_index": 0},
            })
            handler_pdf = FakePhaseHandler({DEC: [chunk_pdf], EPS: []})
            cat_pdf = FakeCat(
                FakeFileManager([], files={("agent", "doc.pdf"): pdf_bytes}),
                handler_pdf,
                {},
            )
            assert asyncio.run(cat_alog._run_catalog_cards_phase("doc.pdf", cat_pdf)).status == "done", "(a) PDF card phase must complete"
            _, card_points_pdf = handler_pdf.added[-1]
            card_pdf = card_points_pdf[0].payload["page_content"]
            initial_pdf = card_pdf.split("## Initial page\n\n", 1)[1].split(
                "\n\n## Summary", 1
            )[0]
            assert initial_pdf.strip() == parsed_pdf.strip(), (
                "(a) a PDF's initial page must be the parsed text"
            )
            assert "%PDF" not in initial_pdf, (
                "(a) a PDF's initial page must not contain raw PDF bytes"
            )
            assert "\ufffd" not in initial_pdf, (
                "(a) a PDF's initial page must not contain replacement chars"
            )

            # (d) The card fed to the concept/relation prompt is readable prose:
            # its first 200 chars contain no binary marker and no replacement char.
            card_head = card_pdf[:200]
            assert "%PDF" not in card_head and "\ufffd" not in card_head, (
                "(d) the PDF card head must not be binary"
            )
            assert any(ch.isalnum() for ch in card_head), (
                "(d) the PDF card head must contain readable alphanumerics"
            )
            assert parsed_pdf[:120] in card_pdf, (
                "(d) the parsed prose must be present in the card"
            )

            # (b) Markdown source: unchanged behaviour — the raw bytes ARE the text,
            # so the initial page is the file content, not the parsed chunk.
            md_bytes = b"# Raw markdown title\n\nRAW MARKDOWN BODY FROM THE FILE\n"
            parsed_md = "PARSED CHUNK TEXT THAT MUST NOT BE THE PREVIEW"
            chunk_md = SimpleNamespace(payload={
                "page_content": parsed_md,
                "metadata": {"source": "notes.md", "chunk_index": 0},
            })
            handler_md = FakePhaseHandler({DEC: [chunk_md], EPS: []})
            cat_md = FakeCat(
                FakeFileManager([], files={("agent", "notes.md"): md_bytes}),
                handler_md,
                {},
            )
            assert asyncio.run(cat_alog._run_catalog_cards_phase("notes.md", cat_md)).status == "done", "(b) markdown card phase must complete"
            _, card_points_md = handler_md.added[-1]
            card_md = card_points_md[0].payload["page_content"]
            initial_md = card_md.split("## Initial page\n\n", 1)[1].split(
                "\n\n## Summary", 1
            )[0]
            assert "RAW MARKDOWN BODY FROM THE FILE" in initial_md, (
                "(b) markdown must keep the raw-bytes preview"
            )
            assert parsed_md not in initial_md, (
                "(b) markdown must not switch to the parsed chunk"
            )

            # (c) A prose chunk that was leaked a catalogue-card flag must be
            # cleaned in place: no card keys survive on the source chunk.
            leak_meta = {
                "source": "leak.pdf",
                "chat_id": None,
                "chunk_index": 0,
                "start_index": 39735,
                "source_chunks": [{"text": "piece"}],
                "is_catalogue_card": True,
                "is_catalogue_card_segment": True,
                "full_card": "OLD CARD BODY",
            }
            chunk_leak = SimpleNamespace(
                id="chunk-leak-not-card",
                payload={
                    "page_content": "REAL PROSE BODY " * 20,
                    "metadata": dict(leak_meta),
                },
            )
            handler_leak = FakePhaseHandler({DEC: [chunk_leak], EPS: []})
            cat_leak = FakeCat(FakeFileManager([]), handler_leak, {})
            assert asyncio.run(cat_alog._run_catalog_cards_phase("leak.pdf", cat_leak)).status == "done", "(c) leak card phase must complete"
            meta_after_leak = chunk_leak.payload["metadata"]
            for card_key in (
                "is_catalogue_card", "is_catalogue_card_segment", "full_card",
            ):
                assert card_key not in meta_after_leak, (
                    f"(c) the prose chunk still leaks {card_key}"
                )
            assert "start_index" not in meta_after_leak, (
                "(c) the card-derived start_index must be dropped from the chunk"
            )
            assert meta_after_leak.get("chunk_index") == 0, (
                "(c) the chunk must keep its own chunk_index"
            )
            _, card_points_leak = handler_leak.added[-1]
            assert card_points_leak[0].payload["metadata"].get(
                "is_catalogue_card"
            ) is True, "(c) the newly written card must still be flagged"
            assert card_points_leak[0].id != "chunk-leak-not-card", (
                "(c) the card must be a distinct point, not the chunk"
            )

            # A plain chunk (no stale flags) is never mutated by the phase.
            plain_meta = {"source": "plain.pdf", "chunk_index": 0, "start_index": 0}
            chunk_plain = SimpleNamespace(
                id="chunk-plain",
                payload={
                    "page_content": "PLAIN SOURCE PROSE",
                    "metadata": dict(plain_meta),
                },
            )
            handler_plain = FakePhaseHandler({DEC: [chunk_plain], EPS: []})
            cat_plain = FakeCat(FakeFileManager([]), handler_plain, {})
            assert asyncio.run(cat_alog._run_catalog_cards_phase("plain.pdf", cat_plain)).status == "done", "(c) plain card phase must complete"
            assert chunk_plain.payload["metadata"] == plain_meta, (
                "(c) a plain chunk must keep its metadata untouched"
            )

            # ── (y) PHASE: the summary prompt + initial page contain ONLY the
            # parsed TEXT chunk. An image point, a binary chunk and a real
            # catalogue card must never enter either.
            captured_prompts.clear()
            mixed_source = "mixed.pdf"
            text_body = "REAL TEXT BODY about the motor and the mission."
            text_chunk = SimpleNamespace(payload={
                "page_content": text_body,
                "metadata": {"source": mixed_source, "chunk_index": 1, "start_index": 16},
            })
            image_point = SimpleNamespace(
                id="img-point",
                payload={
                    "page_content": f"[Image] {mixed_source}",
                    "metadata": {
                        "source": mixed_source,
                        "image": True,
                        "image_file": "mixed_img.png",
                        "image_mime_type": "image/png",
                        "chunk_index": 2,
                    },
                },
            )
            binary_chunk = SimpleNamespace(payload={
                "page_content": "%PDF-1.5\n\x00\xff binary stream bytes",
                "metadata": {"source": mixed_source, "chunk_index": 3},
            })
            mixed_card_id = cat_alog._catalogue_card_point_id(
                "agent", None, mixed_source, 0
            )
            card_point = SimpleNamespace(
                id=mixed_card_id,
                payload={
                    "page_content": (
                        "# CATALOGUE CARD\n\n## Initial page\n\n"
                        "CARD INITIAL PAGE BODY\n\n## Summary\n\nCARD SUMMARY BODY\n"
                    ),
                    "metadata": {
                        "source": mixed_source,
                        "chunk_index": 0,
                        "is_catalogue_card": True,
                        "full_card": "CARD SUMMARY BODY",
                    },
                },
            )
            handler_mixed = FakePhaseHandler({
                DEC: [image_point, card_point, binary_chunk, text_chunk],
                EPS: [],
            })
            cat_mixed = FakeCat(FakeFileManager([]), handler_mixed, {})
            assert asyncio.run(cat_alog._run_catalog_cards_phase(mixed_source, cat_mixed)).status == "done", "(y) mixed phase must complete"
            assert len(captured_prompts) == 1, captured_prompts
            mixed_prompt = captured_prompts[0]
            assert text_body in mixed_prompt, "(y) the text chunk must be summarized"
            assert "[Image]" not in mixed_prompt, "(y) image point leaked into the prompt"
            assert "CARD INITIAL PAGE BODY" not in mixed_prompt, (
                "(y) card initial page leaked into the prompt"
            )
            assert "CARD SUMMARY BODY" not in mixed_prompt, (
                "(y) card summary leaked into the prompt"
            )
            assert "%PDF" not in mixed_prompt, "(y) binary bytes leaked into the prompt"
            mixed_card = handler_mixed.added[-1][1][0].payload["page_content"]
            mixed_initial = mixed_card.split("## Initial page\n\n", 1)[1].split(
                "\n\n## Summary", 1
            )[0]
            assert mixed_initial.strip() == text_body, (
                "(y) the initial page must be the parsed text chunk only"
            )

            # ── (z) LEGACY SURFACE: the two classic ingestion hooks, the EffING
            # phase hooks and their gate are GONE; only the core protocol hook
            # registers the phase. The module must not import EffING.
            for legacy_name in (
                "before_rabbithole_splits_documents",
                "before_rabbithole_stores_documents",
                "ingestion_phase_" + "objects",
                "ingestion_phase_" + "specs",
                "ingestion_phase_" + "settings_marker",
                "ingestion_phase_" + "run",
                "ingestion_phase_" + "purge",
                "_phase_machine_active",
            ):
                assert not hasattr(cat_alog, legacy_name), (
                    f"(z) the legacy ingestion surface {legacy_name!r} must be removed"
                )
            assert hasattr(cat_alog, "rabbithole_ingestion_phases"), (
                "(z) the core registration hook must exist"
            )
            module_source = _inspect.getsource(cat_alog)
            assert "cat_efficient_ingestion" not in module_source, (
                "(z) the plugin must not import the efficient-ingestion plugin"
            )
        finally:
            cat_alog._run_summary_agent = original_summary_x
            cat_alog._resolve_embedder = original_embedder_x

        # ── (y) Qdrant-canonical (dashed) UUID ids: the phase purge must still
        # recognise the source's stored cards and replace them EXACTLY. Qdrant
        # returns point ids as canonical dashed UUID strings while the product
        # mints ``uuid5(...).hex`` (undashed): identity must normalise both forms,
        # otherwise the purge never matches and a changed embedder window leaves
        # the old segments as orphans.
        import uuid as _uuid_y

        def _dashed_y(point_id):
            return str(_uuid_y.UUID(str(point_id)))

        class QdrantCanonicalHandler(FakePhaseHandler):
            """Fake store returning dashed ids and upserting by id, like Qdrant."""

            async def add_points_to_tenant(self, collection_name, points):
                points = list(points)
                self.added.append((collection_name, points))
                store = self.by_collection.setdefault(collection_name, [])
                index = {
                    _dashed_y(record.id): position
                    for position, record in enumerate(store)
                    if getattr(record, "id", None) is not None
                }
                for point in points:
                    point_id = _dashed_y(point.id)
                    record = SimpleNamespace(id=point_id, payload=point.payload)
                    if point_id in index:
                        store[index[point_id]] = record
                    else:
                        index[point_id] = len(store)
                        store.append(record)

        base_y = cat_alog._catalogue_card_point_id("agent", None, "dash.txt")
        assert cat_alog._is_catalogue_card_record(
            {"source": "dash.txt", "is_catalogue_card": True}, base_y, "agent",
        ) is True
        assert cat_alog._is_catalogue_card_record(
            {"source": "dash.txt", "is_catalogue_card": True}, _dashed_y(base_y), "agent",
        ) is True, "(y) a dashed UUID card id must still be recognised"

        window_y = FakeEmbedder(37)

        async def _embedder_y(cat):
            return window_y

        async def _summary_y(cat, prompt):
            return " ".join(f"w{i}" for i in range(300))

        original_summary_y = cat_alog._run_summary_agent
        original_embedder_y = cat_alog._resolve_embedder
        cat_alog._run_summary_agent = _summary_y
        cat_alog._resolve_embedder = _embedder_y
        try:
            handler_y = QdrantCanonicalHandler({
                DEC: [SimpleNamespace(id=_uuid_y.uuid4().hex, payload={
                    "page_content": "DASH CHUNK",
                    "metadata": {"source": "dash.txt", "chunk_index": 0},
                })],
                EPS: [SimpleNamespace(id=_uuid_y.uuid4().hex, payload={
                    "page_content": "DASH CHAT CHUNK",
                    "metadata": {
                        "source": "dash.txt", "chat_id": "chatY", "chunk_index": 0,
                    },
                })],
            })
            fm_y = FakeFileManager([], files={
                ("agent", "dash.txt"): b"DASH PAGE " * 400,
                ("agent/chatY", "dash.txt"): b"DASH CHAT PAGE " * 200,
            })
            cat_y = FakeCat(fm_y, handler_y, {})

            def _cards_y(collection):
                return [
                    record
                    for record in handler_y.by_collection.get(collection, [])
                    if ((record.payload or {}).get("metadata", {}) or {}).get("source")
                    == "dash.txt"
                    and ((record.payload or {}).get("metadata", {}) or {}).get(
                        "is_catalogue_card"
                    ) is True
                ]

            def _ids_y(collection, chat_id):
                return {
                    _dashed_y(record.id)
                    for record in _cards_y(collection)
                    if ((record.payload or {}).get("metadata", {}) or {}).get("chat_id")
                    == chat_id
                }

            def _run_y():
                result = _run_phase("dash.txt", cat_y)
                assert result.status == "done", "(y) phase did not finish"

            _run_y()
            first_dec = _ids_y(DEC, None)
            first_eps = _ids_y(EPS, "chatY")
            assert len(first_dec) > 1 and len(first_eps) > 1, (
                "(y) window 37 did not segment the card"
            )

            # (a) changed window -> EXACTLY the new segment count, no orphan.
            window_y.max_input_tokens = 23
            _run_y()
            second_dec = _ids_y(DEC, None)
            second_eps = _ids_y(EPS, "chatY")
            assert len(second_dec) > len(first_dec), "(y) lower window must add segments"
            assert first_dec - second_dec, "(y) old agent-scoped segment ids survived"
            assert _ids_y(DEC, None) == second_dec, (
                "(y) old agent-scoped segments survived the lower-window re-run"
            )
            assert len(_cards_y(DEC)) == len(second_dec), (
                "(y) orphan agent-scoped segments left next to the new card",
                len(_cards_y(DEC)), len(second_dec),
            )
            assert len(_cards_y(EPS)) == len(second_eps), (
                "(y) orphan chat-scoped segments left next to the new card"
            )

            # (b) identical ids re-run -> count unchanged (no duplicates/orphans).
            _run_y()
            assert _ids_y(DEC, None) == second_dec, (
                "(y) an identical-id re-run changed the agent-scoped segment set"
            )
            assert len(_cards_y(DEC)) == len(second_dec), (
                "(y) an identical-id re-run changed the agent-scoped count"
            )
            assert len(_cards_y(EPS)) == len(second_eps), (
                "(y) an identical-id re-run changed the chat-scoped count"
            )

            # (c) the purge never deletes a non-card point carrying the card flags.
            flagged_id = "flagged-chunk-not-card"
            card_dashed = _dashed_y(
                cat_alog._catalogue_card_point_id("agent", None, "purgey.txt")
            )
            purge_handler = QdrantCanonicalHandler({DEC: [
                SimpleNamespace(id=card_dashed, payload={"metadata": {
                    "source": "purgey.txt", "is_catalogue_card": True,
                }}),
                SimpleNamespace(id=flagged_id, payload={"metadata": {
                    "source": "purgey.txt", "chunk_index": 0,
                    "is_catalogue_card": True, "full_card": "stale card body",
                }}),
            ]})
            removed_y = asyncio.run(cat_alog._purge_existing_cards(
                purge_handler, DEC, "purgey.txt", None, "test", "agent",
            ))
            remaining_y = {record.id for record in purge_handler.by_collection[DEC]}
            assert removed_y == 1 and card_dashed not in remaining_y, (
                "(y) the purge must delete a dashed-id card point"
            )
            assert flagged_id in remaining_y, (
                "(y) the purge must never delete a non-card point carrying the flags"
            )

            # (d) running twice more is idempotent in count.
            _run_y()
            _run_y()
            assert len(_cards_y(DEC)) == len(second_dec), (
                "(y) repeated runs must be idempotent"
            )
            assert len(_cards_y(EPS)) == len(second_eps), (
                "(y) repeated chat-scoped runs must be idempotent"
            )
        finally:
            cat_alog._run_summary_agent = original_summary_y
            cat_alog._resolve_embedder = original_embedder_y

        print("test_catalog_phase: ALL ASSERTIONS PASSED")

    finally:
        pass
