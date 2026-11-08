"""Tests of the Cat_alog plugin: its hooks and tool on fakes of the Cat, and the card through the RabbitHole of the
core (the hook chain, the chunks kept for a source and read back by the embedding).

Run from the root of grinning-cat-core, with its virtual environment:
``.venv/bin/pytest cat/plugins/cat_alog/tests -q -p no:cacheprovider``.

The Cat does not import the ``tests`` folder of a plugin, but its security scan reads it as every other ``.py`` file of
the plugin: no dynamic imports here. The plugin is loaded with the core loader, as in production.
"""
import ast
import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document

PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE = f"cat.plugins.{PLUGIN_DIR.name}.cat_alog"
BASE_HOOKS_DIR = PLUGIN_DIR.parents[1] / "core_plugins" / "base_plugin" / "hooks"

_loaded: Dict[str, Any] = {}


def _plugin():
    if "plugin" not in _loaded:
        from cat.looking_glass.mad_hatter.plugin import Plugin

        plugin = Plugin(str(PLUGIN_DIR))
        plugin._load_decorated_functions()
        _loaded["plugin"] = plugin
    return _loaded["plugin"]


@pytest.fixture(scope="module")
def cat_alog():
    _plugin()
    return sys.modules[MODULE]


def _run(coro):
    return asyncio.run(coro)


# -- fakes ------------------------------------------------------------------------------------------------------------


class FakeWorkflow:
    def __init__(self, reply: str = "A short summary.", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.prompts: List[str] = []

    async def run(self, task, llm):
        self.prompts.append(task.user_prompt)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output=self.reply)


class FakePluginManager:
    def __init__(self, settings: Dict | None, error: Exception | None = None):
        self._settings = settings
        self._error = error

    def get_plugin(self):
        manager = self

        class _Plugin:
            async def load_settings(self):
                if manager._error is not None:
                    raise manager._error
                return manager._settings

        return _Plugin()


class FakeFile:
    def __init__(self, name: str, when: str = "2026-11-01 10:00"):
        self.name = name
        self.last_modified = when


class FakeFileManager:
    def __init__(self, files: Dict[str, List[str]] | None = None, errors: set | None = None):
        self.files = files or {}
        self.errors = errors or set()

    def list_files(self, path: str):
        if path in self.errors:
            raise RuntimeError(f"cannot list {path}")
        return [FakeFile(name) for name in self.files.get(path, [])]


def _record(source=None, content="", **metadata):
    meta = dict(metadata)
    if source is not None:
        meta["source"] = source
    return SimpleNamespace(id="x", payload={"page_content": content, "metadata": meta})


class FakeHandler:
    """The vector memory: records per collection, filtered by equality on the metadata (as the core does)."""

    def __init__(self, records: Dict[str, List] | None = None, failing: set | None = None):
        self.records = records or {}
        self.failing = failing or set()
        self.calls: List = []

    async def get_all_tenant_points(self, collection, metadata=None, limit=None, with_vectors=False):
        self.calls.append((collection, dict(metadata or {}), limit))
        key = (collection, tuple(sorted((metadata or {}).keys())))
        if collection in self.failing or key in self.failing:
            raise RuntimeError("query not supported")
        found = [
            r for r in self.records.get(collection, [])
            if all((r.payload.get("metadata") or {}).get(k) == v for k, v in (metadata or {}).items())
        ]
        if limit is not None:
            found = found[:limit]
        return found, None


def _cat(settings=None, workflow=None, chat_id=None, files=None, handler=None, settings_error=None, fm_errors=None):
    return SimpleNamespace(
        agent_key="agent",
        id=chat_id,
        plugin_manager=FakePluginManager(settings if settings is not None else {}, settings_error),
        agentic_workflow=workflow or FakeWorkflow(),
        large_language_model=object(),
        file_manager=FakeFileManager(files, fm_errors),
        vector_memory_handler=handler if handler is not None else FakeHandler(),
    )


def _chunks(*texts, source="report.pdf", **extra):
    return [Document(page_content=t, metadata={"source": source, "when": 1.0, "hash": "h", **extra}) for t in texts]


def _stores(cat_alog, docs, cat):
    return _run(cat_alog.before_rabbithole_stores_documents.function(docs, cat))


def _suffix(cat_alog, prompt, cat):
    return _run(cat_alog.agent_prompt_suffix.function(prompt, cat))


# -- alignment with the core ------------------------------------------------------------------------------------------


def _core_hook_names() -> set:
    names = set()
    for path in BASE_HOOKS_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def test_every_hook_of_the_plugin_is_a_hook_of_the_core():
    plugin = _plugin()
    names = {h.name for h in plugin.hooks}
    assert names == {"before_rabbithole_stores_documents", "agent_prompt_suffix"}
    assert names <= _core_hook_names()


def test_the_plugin_has_one_tool():
    assert [t.name for t in _plugin().tools] == ["list_loaded_files"]


def test_no_reference_to_the_removed_efficient_ingestion(cat_alog):
    source = inspect.getsource(cat_alog)
    for gone in ("cat_efficient_ingestion", "ingestion_phase", "_phase_machine_active", "EffING", "CATALOGUES"):
        assert gone not in source


def test_the_settings_have_no_cache_tuning(cat_alog):
    fields = set(cat_alog.CatAlogSettings.model_fields)
    assert fields == {"max_document_chars", "max_summary_words", "append_file_list_to_prompt"}
    assert _plugin().settings_model() is cat_alog.CatAlogSettings


# -- the catalogue card, while the source is stored ------------------------------------------------------------------


def test_the_card_is_built_in_the_store_hook_alone(cat_alog):
    workflow = FakeWorkflow("It is a report.")
    docs = _chunks("Title and authors.", "Body of the report.")
    out = _stores(cat_alog, docs, _cat(workflow=workflow))

    assert out[:2] == docs and len(out) == 3
    card = out[2]
    assert card.metadata["is_catalogue_card"] is True
    assert card.metadata["full_card"] == card.page_content
    assert card.metadata["source"] == "report.pdf" and card.metadata["chunk_index"] == 0
    assert "## Source file or URL: report.pdf" in card.page_content
    assert "## Initial page\n\nTitle and authors.\n\n## Summary\n\nIt is a report.\n" in card.page_content
    assert card.id is None  # the core gives every point its id
    assert [d.metadata["chunk_index"] for d in out[:2]] == [0, 1]
    assert len(workflow.prompts) == 1
    assert "Title and authors.\n\nBody of the report." in workflow.prompts[0]


def test_a_chunk_index_set_by_the_chunker_is_kept(cat_alog):
    docs = _chunks("a", "b")
    docs[0].metadata["chunk_index"] = 7
    out = _stores(cat_alog, docs, _cat())
    assert [d.metadata["chunk_index"] for d in out[:2]] == [7, 1]
    assert out[2].metadata["chunk_index"] == 7


def test_every_run_summarizes_its_own_content(cat_alog):
    """A new upload of the same source (or a run after an interrupted one) never reuses an old summary."""
    workflow = FakeWorkflow()
    cat = _cat(workflow=workflow)
    _stores(cat_alog, _chunks("version one"), cat)
    workflow.reply = "second summary"
    out = _stores(cat_alog, _chunks("version two"), cat)
    assert len(workflow.prompts) == 2 and "version two" in workflow.prompts[1]
    assert "second summary" in out[-1].page_content and "version two" in out[-1].page_content


def test_the_chat_scopes_nothing_but_the_metadata_of_the_card(cat_alog):
    docs = _chunks("chat text", chat_id="chat-1")
    out = _stores(cat_alog, docs, _cat(chat_id="chat-1"))
    assert out[-1].metadata["chat_id"] == "chat-1"


def test_the_summary_input_is_bounded_and_escaped(cat_alog):
    workflow = FakeWorkflow()
    docs = _chunks("{x}" + "y" * 50)
    _stores(cat_alog, docs, _cat(settings={"max_document_chars": 10, "max_summary_words": 33}, workflow=workflow))
    prompt = workflow.prompts[0]
    assert "Maximum 33 words." in prompt
    assert "{{x}}yyyyyyy [truncated] " in prompt  # a template for the agentic workflow


def test_a_failing_llm_keeps_the_card_without_summary(cat_alog):
    out = _stores(cat_alog, _chunks("text"), _cat(workflow=FakeWorkflow(error=RuntimeError("down"))))
    assert "## Summary\n\n(summary not available)\n" in out[-1].page_content


def test_no_card_without_readable_text(cat_alog):
    workflow = FakeWorkflow()
    docs = [
        Document(page_content="[Image] report.pdf", metadata={"source": "report.pdf", "image": True}),
        Document(page_content="   ", metadata={"source": "report.pdf"}),
        Document(page_content="%PDF-1.7 binary", metadata={"source": "report.pdf"}),
        Document(page_content="old card", metadata={"source": "report.pdf", "is_catalogue_card": True}),
    ]
    out = _stores(cat_alog, docs, _cat(workflow=workflow))
    assert out == docs and workflow.prompts == []


def test_no_card_for_no_document(cat_alog):
    workflow = FakeWorkflow()
    assert _stores(cat_alog, [], _cat(workflow=workflow)) == []
    assert workflow.prompts == []


def test_image_chunks_never_reach_the_llm_nor_the_initial_page(cat_alog):
    workflow = FakeWorkflow()
    docs = [
        Document(page_content="[Image] report.pdf", metadata={"source": "report.pdf", "image_file": "a.png"}),
        *_chunks("Real text."),
    ]
    out = _stores(cat_alog, docs, _cat(workflow=workflow))
    assert "[Image]" not in workflow.prompts[0]
    assert "## Initial page\n\nReal text.\n" in out[-1].page_content
    assert out[-1].metadata.get("image_file") is None


@pytest.mark.parametrize("content, binary", [
    ("", False),
    ("plain prose", False),
    ("a\x00b", True),
    ("  %PDF-1.4", True),
    ("�" * 10 + "a" * 80, True),
    ("�" + "a" * 80, False),
])
def test_binary_text_detection(cat_alog, content, binary):
    assert cat_alog._looks_binary_text(content) is binary


# -- the loaded files in the prompt ------------------------------------------------------------------------------------


def test_the_prompt_lists_the_files(cat_alog):
    cat = _cat(files={"agent": ["b.txt", "a.pdf"]})
    out = _suffix(cat_alog, "SUFFIX  \n", cat)
    assert out.startswith("SUFFIX\n\n# Loaded files\n")
    assert "- **a.pdf** (Agent, uploaded 2026-11-01 10:00)" in out
    assert out.index("a.pdf") < out.index("b.txt")


def test_the_prompt_is_untouched_when_disabled(cat_alog):
    handler = FakeHandler()
    cat = _cat(settings={"append_file_list_to_prompt": False}, files={"agent": ["a.pdf"]}, handler=handler)
    assert _suffix(cat_alog, "SUFFIX", cat) == "SUFFIX"
    assert handler.calls == []


def test_the_prompt_is_untouched_without_files(cat_alog):
    assert _suffix(cat_alog, "SUFFIX", _cat()) == "SUFFIX"


def test_unreadable_settings_keep_the_list_on(cat_alog):
    cat = _cat(files={"agent": ["a.pdf"]}, settings_error=RuntimeError("redis down"))
    assert "a.pdf" in _suffix(cat_alog, "SUFFIX", cat)


def test_none_settings_keep_the_list_on(cat_alog):
    cat = _cat(files={"agent": ["a.pdf"]})
    cat.plugin_manager = FakePluginManager(None)
    assert "a.pdf" in _suffix(cat_alog, "SUFFIX", cat)


def test_a_failing_collection_leaves_the_prompt_untouched(cat_alog, monkeypatch):
    async def broken(cat):
        raise RuntimeError("boom")

    monkeypatch.setattr(cat_alog, "_collect_loaded_files", broken)
    assert _suffix(cat_alog, "SUFFIX", _cat(files={"agent": ["a.pdf"]})) == "SUFFIX"


def test_an_empty_listing_leaves_the_prompt_untouched(cat_alog, monkeypatch):
    async def files(cat):
        return {"a.pdf": {"source": "a.pdf", "when": None, "content": None, "type": "agent"}}

    monkeypatch.setattr(cat_alog, "_collect_loaded_files", files)
    monkeypatch.setattr(cat_alog, "_format_loaded_files", lambda files_dict, with_previews=False: "")
    assert _suffix(cat_alog, "SUFFIX", _cat()) == "SUFFIX"


def test_braces_in_file_names_are_escaped_in_the_prompt(cat_alog):
    out = _suffix(cat_alog, "S", _cat(files={"agent": ["{x}.txt"]}))
    assert "{{x}}.txt" in out


# -- the collector of the loaded files --------------------------------------------------------------------------------

DECL = "declarative"
EPIS = "episodic"


def _collect(cat_alog, cat):
    return _run(cat_alog._collect_loaded_files(cat))


def test_the_collector_reads_cards_first_chunks_and_urls(cat_alog):
    card = "# CATALOGUE CARD\n\n## Summary\n\nCard summary."
    handler = FakeHandler({
        DECL: [
            _record("a.pdf", "card text", is_catalogue_card=True, full_card=card, chunk_index=0),
            _record("a.pdf", "first chunk of a", chunk_index=0),
            _record("b.txt", "first chunk of b", chunk_index=0),
            _record("https://x.org/p", "url text", chunk_index=0, when=1_700_000_000),
            _record("https://x.org/q", "", chunk_index=0),
            _record(None, "no source", chunk_index=0),
            _record("ghost.pdf", "not on disk", chunk_index=0),
            _record("a.pdf", "[Image] a.pdf", chunk_index=0, image=True, image_file="a-1.png"),
        ],
        EPIS: [
            _record("c.md", "legacy card\n\n## Summary\n\nLegacy summary.", is_catalogue_card=True, chat_id="chat"),
            _record("c.md", "c text", chunk_index=0, chat_id="chat"),
        ],
    })
    cat = _cat(chat_id="chat", files={"agent": ["a.pdf", "b.txt"], "agent/chat": ["c.md"]}, handler=handler)
    files = _collect(cat_alog, cat)

    assert files["a.pdf"]["content"] == "Card summary." and files["a.pdf"]["type"] == "agent"
    assert files["b.txt"]["content"] == "first chunk of b"
    assert files["https://x.org/p"]["content"] == "url text" and files["https://x.org/p"]["when"] != "unknown"
    assert files["https://x.org/q"]["when"] == "unknown"
    assert files["c.md"] == {"source": "c.md", "when": "2026-11-01 10:00", "content": "Legacy summary.",
                             "type": "conversation"}
    assert "ghost.pdf" not in files


def test_a_card_without_summary_marker_falls_back_to_a_stored_text(cat_alog):
    handler = FakeHandler({DECL: [
        _record("a.pdf", "card without marker", is_catalogue_card=True, full_card="no marker"),
        _record("a.pdf", "", is_catalogue_card=True),
    ]})
    files = _collect(cat_alog, _cat(files={"agent": ["a.pdf"]}, handler=handler))
    assert files["a.pdf"]["content"] == "card without marker"


def test_the_legacy_fallback_previews_files_without_card_nor_first_chunk(cat_alog):
    handler = FakeHandler({DECL: [
        _record("old.txt", "old text"),
        _record("binary.bin", "%PDF-1.3 garbage"),
        _record("photo.png", "[Image] photo.png", image=True, image_file="photo.png"),
        _record("doc.pdf", "image of doc", image_file="extracted.png"),
    ]})
    cat = _cat(files={"agent": ["old.txt", "binary.bin", "photo.png", "extracted.png", "empty.txt"]}, handler=handler)
    files = _collect(cat_alog, cat)
    assert files["old.txt"]["content"] == "old text"
    assert files["binary.bin"]["content"] is None
    assert files["photo.png"]["content"] is None
    assert files["empty.txt"]["content"] is None
    assert "extracted.png" not in files or files["extracted.png"]["content"] is None


def test_derived_images_are_never_listed(cat_alog):
    handler = FakeHandler({DECL: [
        _record("doc.pdf", "[Image]", image=True, image_file="doc-1.png"),
        _record("photo.png", "[Image]", image=True, image_file="photo.png"),
        _record("other.pdf", "x", image="yes", image_file="other-1.png"),
    ]})
    cat = _cat(files={"agent": ["doc.pdf", "doc-1.png", "photo.png", "other-1.png"]}, handler=handler)
    files = _collect(cat_alog, cat)
    assert "doc-1.png" not in files and "photo.png" in files and "other-1.png" in files


def test_failures_of_storage_and_memory_are_tolerated(cat_alog):
    handler = FakeHandler(failing={DECL, EPIS})
    cat = _cat(chat_id="chat", files={"agent/chat": ["c.md"]}, handler=handler, fm_errors={"agent"})
    assert _collect(cat_alog, cat) == {
        "c.md": {"source": "c.md", "when": "2026-11-01 10:00", "content": None, "type": "conversation"},
    }
    cat = _cat(chat_id="chat", files={"agent": ["a.pdf"]}, handler=FakeHandler(), fm_errors={"agent/chat"})
    assert set(_collect(cat_alog, cat)) == {"a.pdf"}


def test_failing_queries_are_skipped_one_by_one(cat_alog):
    handler = FakeHandler(
        {DECL: [_record("a.pdf", "first", chunk_index=0)]},
        failing={(DECL, ("is_catalogue_card",)), (EPIS, ("chat_id", "is_catalogue_card"))},
    )
    files = _collect(cat_alog, _cat(files={"agent": ["a.pdf"]}, handler=handler))
    assert files["a.pdf"]["content"] == "first"

    handler = FakeHandler({DECL: [_record("a.pdf", "first")]}, failing={(DECL, ("chunk_index",))})
    files = _collect(cat_alog, _cat(files={"agent": ["a.pdf"]}, handler=handler))
    assert files["a.pdf"]["content"] == "first"  # from the legacy fallback


def test_the_collector_tolerates_a_broken_handler(cat_alog):
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("no handler")

    cat = _cat(files={"agent": ["a.pdf"]})
    cat.vector_memory_handler = Broken()
    assert _collect(cat_alog, cat)["a.pdf"]["content"] is None
    del cat.vector_memory_handler
    assert set(_collect(cat_alog, cat)) == {"a.pdf"}


def test_a_failing_fallback_query_is_tolerated(cat_alog):
    handler = FakeHandler({DECL: [_record("a.pdf", "x")]})
    original = handler.get_all_tenant_points

    async def failing_after_previews(collection, metadata=None, limit=None, with_vectors=False):
        if limit == 1:
            raise RuntimeError("fallback query failed")
        return await original(collection, metadata, limit, with_vectors)

    handler.get_all_tenant_points = failing_after_previews
    files = _collect(cat_alog, _cat(files={"agent": ["a.pdf"]}, handler=handler))
    assert files["a.pdf"]["content"] is None


# -- the formatter and the tool ---------------------------------------------------------------------------------------


def test_the_formatter_with_previews(cat_alog):
    files = {
        "a.pdf": {"source": "a.pdf", "when": "w", "content": "preview", "type": "agent"},
        "c.md": {"source": "c.md", "when": "w", "content": None, "type": "conversation"},
    }
    out = cat_alog._format_loaded_files(files, with_previews=True)
    assert out.startswith("# Loaded Files\n")
    assert "## FILE NAME: a.pdf\n" in out and "- **Type**: Agent" in out and "```\npreview...\n```" in out
    assert "- **Type**: Chat" in out and "**Content preview:** *(not available)*" in out
    assert cat_alog._format_loaded_files({}, with_previews=True).endswith(
        "No files loaded in this conversation or agent.")


def test_the_compact_formatter(cat_alog):
    files = {
        "a": {"when": None, "type": "agent"},
        "b": {"when": 1_700_000_000, "type": "conversation"},
        "c": {"when": "2026-01-01", "type": "agent"},
    }
    out = cat_alog._format_loaded_files(files)
    assert "- **a** (Agent, uploaded unknown)" in out
    assert "- **b** (Chat, uploaded 2023-11-1" in out
    assert "- **c** (Agent, uploaded 2026-01-01)" in out


def test_the_tool_lists_the_files_with_previews(cat_alog):
    handler = FakeHandler({DECL: [_record("a.pdf", "first", chunk_index=0)]})
    out = cat_alog.list_loaded_files.func(_cat(files={"agent": ["a.pdf"]}, handler=handler))
    assert "## FILE NAME: a.pdf" in out and "first" in out


def test_odd_records_are_skipped(cat_alog):
    handler = FakeHandler({DECL: [
        _record(None, "card of nothing", is_catalogue_card=True),
        _record("b.txt", "[Image] b.txt", chunk_index=0, image=True, image_file="b-1.png"),
        _record("e.txt", ""),
        _record("e.txt", "image text", image_file="e.txt"),
    ]})
    cat = _cat(files={"agent": ["b.txt", "b.txt", "e.txt"]}, handler=handler)
    files = _collect(cat_alog, cat)
    assert files["b.txt"]["content"] is None
    assert files["e.txt"]["content"] is None


def test_derived_images_only_from_image_points(cat_alog):
    """A vector database that does not filter on booleans still lists only real image points."""

    class Unfiltered(FakeHandler):
        async def get_all_tenant_points(self, collection, metadata=None, limit=None, with_vectors=False):
            return self.records.get(collection, []), None

    handler = Unfiltered({DECL: [
        _record("doc.pdf", "x", image="yes", image_file="doc-1.png"),
        _record("doc.pdf", "[Image]", image=True, image_file="doc-2.png"),
    ]})
    assert _run(cat_alog._derived_image_files(_cat(handler=handler))) == {"doc-2.png"}


# -- the card through the RabbitHole of the core ----------------------------------------------------------------------


class MemoryFileManager:
    """The storage of the files kept by the RabbitHole, in memory."""

    def __init__(self):
        self.files: Dict[str, Any] = {}

    def write_file(self, content, name, folder):
        self.files[f"{folder}/{name}"] = content
        return True

    def read_file(self, name, folder):
        return self.files.get(f"{folder}/{name}")


def _rabbit_hole_with_plugin(workflow):
    """A RabbitHole whose hooks are the ones of the plugin, run by the hook chain of the core (deep copy included)."""
    from cat.looking_glass.mad_hatter.mad_hatter import MadHatter
    from cat.rabbit_hole import RabbitHole

    hooks = {h.name: h for h in _plugin().hooks}
    chain = SimpleNamespace(context_execute_hook="cat", plugins={})

    class PluginManager(FakePluginManager):
        async def execute_hook(self, name, *args, caller):
            return await MadHatter._run_hooks(chain, [hooks[name]], *args, caller=caller)

    ccat = _cat(workflow=workflow)
    ccat.plugin_manager = PluginManager({})
    rabbit_hole = RabbitHole()
    rabbit_hole.cat = ccat
    rabbit_hole.stray = None
    return rabbit_hole


def _parsed(*texts):
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


def test_the_card_is_kept_with_the_chunks_and_read_back_without_the_llm():
    from cat.rabbit_hole import chunks

    workflow = FakeWorkflow("Kept summary.")
    rabbit_hole = _rabbit_hole_with_plugin(workflow)
    docs = _run(rabbit_hole._prepare_documents(_parsed("Title.", "Body."), "a.pdf", "h", {"owner": "u"}, None))
    assert len(docs) == 3 and docs[-1].metadata["is_catalogue_card"] is True
    assert docs[-1].metadata["owner"] == "u" and docs[-1].metadata["source"] == "a.pdf"

    file_manager = MemoryFileManager()
    chunks.write_chunks(file_manager, "agent", None, "a.pdf", docs, [])
    kept, images = chunks.read_chunks(file_manager, "agent", None, "a.pdf")

    # what the embedding phase reads at every re-embedding: the same card, no new LLM call
    assert [d.model_dump() for d in kept] == [d.model_dump() for d in docs] and images == []
    assert len(workflow.prompts) == 1


def test_every_parsing_of_a_source_carries_exactly_one_card():
    workflow = FakeWorkflow()
    rabbit_hole = _rabbit_hole_with_plugin(workflow)
    for _ in range(2):  # an upload, then a new parsing (a re-upload, a change of the chunker)
        docs = _run(rabbit_hole._prepare_documents(_parsed("Text."), "a.pdf", "h", {}, "chat"))
        assert sum(1 for d in docs if d.metadata.get("is_catalogue_card")) == 1
        assert docs[-1].metadata["chat_id"] == "chat"
    assert len(workflow.prompts) == 2
