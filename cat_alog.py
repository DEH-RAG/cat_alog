import asyncio
import hashlib
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, List, Tuple, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from cat import hook, plugin, tool, log, AgenticWorkflowTask
from cat.services.memory.models import VectorMemoryType, PointStruct
from langchain_core.documents.base import Document


class CatAlogSettings(BaseModel):
    max_document_chars: int = Field(
        default=8000,
        description="Maximum number of characters sent to the LLM for file summarization."
    )
    max_summary_words: int = Field(
        default=200,
        description="Maximum number of words for the summary."
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for cache entries in seconds."
    )
    cache_max_size: int = Field(
        default=100,
        description="Maximum number of entries in the LRU cache."
    )


@plugin
def settings_model():
    return CatAlogSettings


class TTL_LRU_Cache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict[Tuple[str, Optional[str], str], Tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _make_key(self, agent_id: str, chat_id: Optional[str], filename: str) -> Tuple[str, Optional[str], str]:
        return (agent_id, chat_id, filename)

    def get(self, agent_id: str, chat_id: Optional[str], filename: str) -> Optional[str]:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value
        return None

    def set(self, agent_id: str, chat_id: Optional[str], filename: str, summary: str) -> None:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (summary, time.time())
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def delete(self, agent_id: str, chat_id: Optional[str], filename: str) -> bool:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False


CATALOGUES = TTL_LRU_Cache()

#: Prompt template for the catalogue-card summary. Kept as a module constant
#: so the settings marker (ingestion_phase_settings_marker) can fingerprint it:
#: a change to the prompt invalidates previously generated cards.
_SUMMARY_PROMPT_TEMPLATE = """Write a short summary of the following file.
Focus on what the file is, what it is about, and what it contains.
Maximum {max_summary_words} words.

## Filename or source: {source}

## File content

{file_content}
"""


_EFFICIENT_INGESTION_CONFIG_NAME = "EfficientIngestionConfiguration"


async def _resolved_system_ingestion_config_name(cat) -> "Optional[str]":
    """Effective SYSTEM-level ingestion engine name.

    The ``ingestion`` category is one of the two system-level services (the
    other is ``embedder``), so its selection is read against
    ``DEFAULT_SYSTEM_KEY`` — never the agent key. ``resolved_config_name`` is
    the canonical resolver (signature: ``resolved_config_name(lizard) -> str``);
    a host that predates it is handled by reading the same single system-level
    category entry directly, so both paths return the same value.
    """
    try:
        from cat.services.factory.ingestion import resolved_config_name
    except ImportError:
        resolved_config_name = None

    if resolved_config_name is not None:
        return await resolved_config_name(cat)

    from cat.db.cruds import settings as crud_settings
    from cat.db.database import DEFAULT_SYSTEM_KEY

    saved = await crud_settings.get_settings_by_category(
        DEFAULT_SYSTEM_KEY, "ingestion"
    )
    return (saved or {}).get("name")


async def _phase_machine_active(cat=None) -> bool:
    """Whether the EffING phase path owns card generation for this ingest.

    Single source of truth for the two ingestion modes (same contract as
    GraphRAG's phase path). The phase path is active **iff**:

    ==========================  ======================  =============
    EffING installed/loadable   SYSTEM ``ingestion``    phase path
    ==========================  ======================  =============
    no                          any                     inactive
    yes                         core engine             inactive
    yes                         efficient engine        active
    ==========================  ======================  =============

    ``yes`` means the phases module is importable from the CANONICAL path
    ``cat.core_plugins.cat_efficient_ingestion.phases``. When the phase path is
    active the ``catalog_cards`` phase owns card generation and persistence, so
    the two ``before_rabbithole_*`` hooks are no-ops; in every other case the
    classic hooks run (TTL cache + card append) and the phase registrations are
    simply inert, so the two mechanisms can never run together and neither is
    dead. Import-safe (no side effects).
    """
    try:
        import cat.core_plugins.cat_efficient_ingestion.phases  # noqa: F401
    except Exception:
        return False

    if cat is None:
        return False

    try:
        name = await _resolved_system_ingestion_config_name(cat)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            f"cat_alog: could not resolve the system ingestion engine "
            f"({exc!r}); using the legacy card hooks."
        )
        return False

    return name == _EFFICIENT_INGESTION_CONFIG_NAME


def _build_summary_prompt(source: str, full_text: str, max_summary_words: int) -> str:
    """Build the LLM summary prompt for ``source`` (shared by hook and phase).

    The file content is brace-escaped so it can be interpolated with
    ``str.format`` without breaking the template placeholders.
    """
    safe_text = full_text.replace("{", "{{").replace("}", "}}")
    return _SUMMARY_PROMPT_TEMPLATE.format(
        max_summary_words=max_summary_words,
        source=source,
        file_content=safe_text,
    )


async def _run_summary_agent(cat, prompt: str) -> str:
    """Run the LLM summary agent for ``prompt`` (no exception swallowing).

    The legacy hook wraps this in try/except with a fallback; the
    ``catalog_cards`` phase lets exceptions propagate (fail-hard -> ERROR).
    """
    agent_input = AgenticWorkflowTask(user_prompt=prompt)
    summary_agent_output = await cat.agentic_workflow.run(
        task=agent_input,
        llm=cat.large_language_model,
    )
    return summary_agent_output.output


def _build_catalogue_card(source: str, metadata: Dict, abstract: str, summary: str) -> Tuple[str, Dict]:
    """Build the catalogue card text + metadata (shared by hook and phase)."""
    card = (
        "\n# CATALOGUE CARD\n\n"
        f"## Source file or URL: {source}\n\n"
        "## Initial page\n\n"
        + abstract + "\n\n"
        "## Summary\n\n"
        + summary + "\n"
    )

    card_metadata = {
        **metadata,
        "is_catalogue_card": True,
        # Store the FULL card text on every card point. When the card exceeds
        # the embedder's input window it is split into segments by the core
        # (each with its own embedding + metadata); with this field every
        # segment carries the complete card, so any similarity hit on any part
        # of the card retrieves the whole card.
        "full_card": card,
    }
    return card, card_metadata


async def _resolve_embedder(cat):
    """Resolve the active embedder (CheshireCat.lizard or StrayCat.embedder)."""
    lizard = getattr(cat, "lizard", None)
    if lizard is not None and hasattr(lizard, "embedder"):
        return await lizard.embedder()
    return await cat.embedder()


@hook(priority=10)
def ingestion_phase_specs(specs, cat) -> list:
    """Accumulator registrant declaring the ``catalog_cards`` phase.

    Appends ``PhaseSpec("catalog_cards", settings_category=None,
    depends_on=("parsing_chunking", "embedding"))`` so EffING computes
    staleness for the card phase: it re-runs when the parsing/embedding
    upstreams re-ran or when the plugin-defined settings marker changed.
    Lazy-imports ``PhaseSpec``: with EffING absent the specs are returned
    unchanged (import-safe).
    """
    try:
        from cat.core_plugins.cat_efficient_ingestion.phases import PhaseSpec
    except Exception:
        return specs
    specs = specs or []
    specs.append(
        PhaseSpec(
            "catalog_cards",
            settings_category=None,
            depends_on=("parsing_chunking", "embedding"),
        )
    )
    return specs


@hook(priority=10)
async def ingestion_phase_settings_marker(phase, cat) -> Any:
    """Material settings marker for the ``catalog_cards`` phase.

    SHA-256 fingerprint of the effective card config (max_document_chars,
    max_summary_words, cache_ttl_seconds, cache_max_size) plus the summary
    prompt template: any material change invalidates previously generated
    cards. ``None`` for unknown phases (conservative stale).
    """
    if phase != "catalog_cards":
        return None
    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    config = (
        int(settings.get("max_document_chars", 8000)),
        int(settings.get("max_summary_words", 200)),
        int(settings.get("cache_ttl_seconds", 3600)),
        int(settings.get("cache_max_size", 100)),
    )
    fingerprint_source = repr((config, _SUMMARY_PROMPT_TEMPLATE))
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


@hook(priority=10)
async def ingestion_phase_run(phase, source, completed_phases, cat) -> Any:
    """Run the ``catalog_cards`` phase for ``source`` (tri-state contract).

    - ``{"status": "done"}``: the card was embedded and stored as a vector
      point with ``is_catalogue_card: True`` + ``full_card`` metadata;
    - ``{"status": "not_ready", "retry_after": 5}``: the source has no stored
      points yet (parsing/embedding not done) — retry later;
    - raise: permanent failure (fail-hard -> ERROR);
    - ``None`` for unknown phases (fail-hard: a phase nobody implements is an
      error, never a silent success).
    """
    if phase != "catalog_cards":
        return None

    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    handler = cat.vector_memory_handler
    records, _ = await handler.get_all_tenant_points(
        str(VectorMemoryType.DECLARATIVE),
        metadata={"source": source},
        with_vectors=False,
    )

    # Collect the source's stored chunk texts, excluding any previously
    # stored catalogue card (a re-run must not summarize its own card).
    chunks: List[str] = []
    first_metadata: Dict = {}
    for rec in records or []:
        payload = rec.payload or {}
        meta = payload.get("metadata", {}) or {}
        if meta.get("is_catalogue_card"):
            continue
        content = payload.get("page_content", "") or ""
        if content.strip():
            if not chunks:
                first_metadata = meta
            chunks.append(content)
    if not chunks:
        return {"status": "not_ready", "retry_after": 5}

    full_text = "\n\n".join(chunks)
    if len(full_text) > max_document_chars:
        full_text = full_text[:max_document_chars] + " [truncated] "

    prompt = _build_summary_prompt(source, full_text, max_summary_words)
    summary = await _run_summary_agent(cat, prompt)

    abstract = chunks[0].strip()
    card, card_metadata = _build_catalogue_card(source, first_metadata, abstract, summary)

    embedder = await _resolve_embedder(cat)
    vector = (await asyncio.to_thread(embedder.embed_documents, [card]))[0]
    # Deterministic point id: re-runs upsert the same card instead of
    # accumulating duplicates (add_points_to_tenant is an upsert).
    point = PointStruct(
        id=uuid.uuid5(uuid.NAMESPACE_URL, f"catalogue_card:{source}").hex,
        payload={"page_content": card, "metadata": card_metadata},
        vector=vector,
    )
    await handler.add_points_to_tenant(str(VectorMemoryType.DECLARATIVE), points=[point])

    log.info(f"cat_alog: catalogue card stored for '{source}'")
    return {"status": "done"}


@hook(priority=10)
async def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    if not docs:
        return docs

    # With the EffING phase machine active, the catalog_cards phase owns card
    # generation: the in-memory TTL cache is no longer the persistence.
    if await _phase_machine_active(cat):
        return docs

    metadata = docs[0].metadata
    if 'source' not in metadata:
        return docs

    source = metadata['source'] 
    agent_id = cat.agent_key
    # chat_id is not added to metadata until store_documents(); fall back to
    # cat.id (StrayCat) which is the conversation ID when available.
    chat_id = metadata.get('chat_id') or getattr(cat, 'id', None)

    cached_summary = CATALOGUES.get(agent_id, chat_id, source)
    if cached_summary is not None:
        log.info(f"cat_alog: cache hit for '{agent_id}/{chat_id}/{source}'")
        return docs

    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    full_text = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()
    )
    if len(full_text) > max_document_chars:
        full_text = full_text[:max_document_chars] + ' [truncated] '

    full_prompt = _build_summary_prompt(source, full_text, max_summary_words)

    try:
        summary = await _run_summary_agent(cat, full_prompt)
    except Exception as e:
        log.warning(f"cat_alog: failed to summarize '{agent_id}/{chat_id}/{source}': {e}")
        summary = "(summary not available)"

    CATALOGUES.set(agent_id, chat_id, source, summary)

    log.info(f"cat_alog: summary added for '{agent_id}/{chat_id}/{source}' [{summary[:100]}...]")
    return docs


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """
    Retrieve the pre-computed summary from the cat and append the catalogue card.
    """
    if not docs:
        return docs

    # With the EffING phase machine active, the catalog_cards phase owns card
    # generation and persistence: this hook is a no-op.
    if await _phase_machine_active(cat):
        return docs

    metadata = docs[0].metadata
    if 'source' not in metadata:
        return docs

    # Add positional index to identify the first chunk (chunk_index=0)
    # Only set if not already present (respects prior settings from chunker/plugins).
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("chunk_index", i)

    source = metadata['source']
    agent_id = cat.agent_key
    chat_id = metadata.get('chat_id') or getattr(cat, 'id', None)

    summary = CATALOGUES.get(agent_id, chat_id, source)
    abstract = docs[0].page_content.strip()

    if not summary:
        log.warning(f"cat_alog: no summary found for {agent_id}/{chat_id}/{source}, skipping catalogue card.")
        return docs

    CATALOGUES.delete(agent_id, chat_id, source)

    card, card_metadata = _build_catalogue_card(source, metadata, abstract, summary)

    log.debug(f"cat_alog: added catalogue card for '{source}'")
    return docs + [Document(page_content=card, metadata=card_metadata)]


@tool(examples=[
    "What files are present",
    "List the files",
    "Tell me what files you know",
])
def list_loaded_files(cat):
    """List all files loaded in the current conversation and agent with their details.

    Returns a markdown formatted list showing files/URLs loaded in the current chat session
    and at the agent level, with their source, upload date, and content preview.
    """
    files_dict = asyncio.run(_collect_loaded_files(cat))
    return _format_loaded_files(files_dict, with_previews=True)


async def _collect_loaded_files(cat) -> Dict[str, Dict[str, Any]]:
    agent_id = cat.agent_key
    chat_id = getattr(cat, "id", None)

    files_dict: Dict[str, Dict[str, Any]] = {}

    # ──────────────────────────────────────────────
    # PHASE 1: Ground truth from FileManager (fast, no DB)
    #   Lists files stored on disk for both agent and conversation levels.
    # ──────────────────────────────────────────────
    async def add_files_from_manager():
        try:
            for fm_file in cat.file_manager.list_files(agent_id):
                if fm_file.name not in files_dict:
                    files_dict[fm_file.name] = {
                        "source": fm_file.name,
                        "when": fm_file.last_modified,
                        "content": None,
                        "type": "agent",
                    }
        except Exception as e:
            log.warning(f"cat_alog: FileManager error (agent): {e}")

        if chat_id:
            try:
                conv_path = os.path.join(agent_id, chat_id)
                for fm_file in cat.file_manager.list_files(conv_path):
                    files_dict[fm_file.name] = {
                        "source": fm_file.name,
                        "when": fm_file.last_modified,
                        "content": None,
                        "type": "conversation",
                    }
            except Exception as e:
                log.warning(f"cat_alog: FileManager error (conversation): {e}")

    # ──────────────────────────────────────────────
    # PHASE 2: Targeted DB queries for content previews
    #   Query 1: catalogue cards (is_catalogue_card=True) → get the summary.
    #   Query 2: first chunks (chunk_index=0) → only for sources WITHOUT a card.
    # ──────────────────────────────────────────────
    async def add_db_previews(collection_name: str, is_episodic: bool):
        try:
            handler = cat.vector_memory_handler
            meta_filter = {"chat_id": chat_id} if is_episodic else {}
            scope_type = "conversation" if is_episodic else "agent"

            results: List[Any] = []
            sources_with_cards: set = set()

            # ── Query 1: catalogue cards ──
            try:
                records, _ = await handler.get_all_tenant_points(
                    collection_name, metadata={**meta_filter, "is_catalogue_card": True}, with_vectors=False,
                )
                for rec in records:
                    src = ((rec.payload or {}).get("metadata", {}) or {}).get("source")
                    if src:
                        sources_with_cards.add(src)
                results.extend(records)
            except Exception:
                log.debug(f"cat_alog: catalogue card query for {collection_name} not supported, skipping")

            # ── Query 2: first chunks (only for sources WITHOUT a card) ──
            try:
                records, _ = await handler.get_all_tenant_points(
                    collection_name, metadata={**meta_filter, "chunk_index": 0}, with_vectors=False,
                )
                for rec in records:
                    src = ((rec.payload or {}).get("metadata", {}) or {}).get("source")
                    if src and src not in sources_with_cards:
                        results.append(rec)
            except Exception:
                log.debug(f"cat_alog: first chunk query for {collection_name} not supported, skipping")

            # ── Process results ──
            for record in results:
                payload = record.payload or {}
                meta = payload.get("metadata", {}) or {}
                source = meta.get("source")

                if not source:  # or source.startswith("http"):
                    continue

                page_content = payload.get("page_content", "") or ""

                if source not in files_dict:
                    # Only URLs are allowed without a FileManager entry
                    # (they have no file on disk).
                    if not source.startswith("http"):
                        continue
                    when_ts = meta.get("when", 0)
                    when_str = datetime.fromtimestamp(when_ts).strftime("%Y-%m-%d %H:%M") if when_ts else "unknown"
                    files_dict[source] = {
                        "source": source,
                        "when": when_str,
                        "content": page_content[:500],
                        "type": scope_type,
                    }

                is_card = meta.get("is_catalogue_card", False)
                if is_card:
                    # Prefer the full card stored in metadata: survives card
                    # splitting (every segment carries it) and is immune to
                    # the marker landing in a different segment.
                    full_card = meta.get("full_card", "")
                    if full_card:
                        summary_marker = "## Summary\n\n"
                        if summary_marker in full_card:
                            summary = full_card.split(summary_marker, 1)[1].strip()[:500]
                            files_dict[source]["content"] = summary
                            continue
                    # fallback: parse the page_content (legacy cards)
                    if page_content:
                        summary_marker = "## Summary\n\n"
                        if summary_marker in page_content:
                            summary = page_content.split(summary_marker, 1)[1].strip()[:500]
                            files_dict[source]["content"] = summary
                elif files_dict[source].get("content") is None:
                    files_dict[source]["content"] = page_content[:500]
        except Exception as e:
            log.warning(f"cat_alog: error reading collection {collection_name}: {e}")

    # ──────────────────────────────────────────────
    # PHASE 3: Legacy fallback
    #   For files in FileManager without a preview (no chunk_index /
    #   is_catalogue_card in metadata, e.g. pre-update files):
    #   query one chunk by source with limit=1.
    # ──────────────────────────────────────────────
    async def add_legacy_fallback(collection_name: str, is_episodic: bool):
        try:
            handler = cat.vector_memory_handler
            meta_filter = {"chat_id": chat_id} if is_episodic else {}
            scope_type = "conversation" if is_episodic else "agent"

            for source, info in list(files_dict.items()):
                if info.get("content") is not None:
                    continue
                if info["type"] != scope_type:
                    continue

                # Match either by metadata.source (text chunks) or by
                # metadata.image_file (images extracted from PDFs store the
                # saved image file name there).
                for match_key in ("source", "image_file"):
                    try:
                        records, _ = await handler.get_all_tenant_points(
                            collection_name,
                            metadata={**meta_filter, match_key: source},
                            limit=1,
                            with_vectors=False,
                        )
                        if records and records[0].payload:
                            content = (records[0].payload.get("page_content") or "")[:500]
                            if content:
                                files_dict[source]["content"] = content
                                log.debug(f"cat_alog: legacy fallback found preview for '{source}' ({match_key})")
                                break
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"cat_alog: error in legacy fallback for {collection_name}: {e}")

    # Execute all phases in order
    await add_files_from_manager()
    await add_db_previews(str(VectorMemoryType.DECLARATIVE), is_episodic=False)
    await add_db_previews(str(VectorMemoryType.EPISODIC), is_episodic=True)
    await add_legacy_fallback(str(VectorMemoryType.DECLARATIVE), is_episodic=False)
    await add_legacy_fallback(str(VectorMemoryType.EPISODIC), is_episodic=True)

    return files_dict


def _format_loaded_files(files_dict, with_previews: bool = False) -> str:
    pieces = ["# Loaded Files\n"] if with_previews else ["# Loaded files"]

    if with_previews:
        if not files_dict:
            pieces.append("No files loaded in this conversation or agent.")
        else:
            for source, info in sorted(files_dict.items()):
                file_type = "Chat" if info["type"] == "conversation" else "Agent"
                content = info.get("content", "")
                when = info["when"]

                pieces.append(f"## FILE NAME: {source}\n")
                pieces.append(f"- **Type**: {file_type}")
                pieces.append(f"- **Uploaded**: {when}\n")

                if content:
                    pieces.append("**Content preview:**\n")
                    pieces.append(f"```\n{content}...\n```\n")
                else:
                    pieces.append("**Content preview:** *(not available)*\n")
    else:
        for source, info in sorted(files_dict.items()):
            file_type = "Chat" if info["type"] == "conversation" else "Agent"
            when = info["when"]
            if not when:
                when = "unknown"
            elif isinstance(when, (float, int)):
                when = datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M")
            pieces.append(f"- **{source}** ({file_type}, uploaded {when})")

    return "\n".join(pieces)


@hook(priority=10)
async def agent_prompt_suffix(prompt_suffix: str, cat) -> str:
    """Append the list of files loaded in the agent and conversation to the end of the system prompt."""
    try:
        files = await _collect_loaded_files(cat)
    except Exception as e:
        log.warning(f"cat_alog: agent_prompt_suffix could not collect loaded files: {e}")
        return prompt_suffix
    if not files:
        return prompt_suffix
    listing = _format_loaded_files(files, with_previews=False).replace("{", "{{").replace("}", "}}")
    if not listing:
        return prompt_suffix
    log.info(f"cat_alog: appended {len(files)} loaded file(s) to prompt suffix")
    return prompt_suffix.rstrip() + "\n\n" + listing
