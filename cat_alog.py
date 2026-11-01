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
    append_file_list_to_prompt: bool = Field(
        default=True,
        description=(
            "Append the list of files loaded in the agent (and conversation) to "
            "the end of the system prompt. When False the prompt is left untouched."
        )
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

# EffING is optional: the classic CAT ingestion path must keep working when its
# phase module is absent. The class API is imported from the CANONICAL path
# (the host loads ``cat/core_plugins/`` and derives the module name from the
# folder, so any other path would be a second module identity).
try:
    from cat.core_plugins.cat_efficient_ingestion.phases import (
        IngestionPhase as _IngestionPhase,
        PhaseResult as _PhaseResult,
    )
except Exception:  # noqa: BLE001 - optional dependency, never break the import
    _IngestionPhase = None
    _PhaseResult = None


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
    """Build the catalogue card text + metadata (shared by hook and phase).

    The seed metadata comes from the first stored chunk. It is stripped of any
    card-only keys first so the card can never inherit ``is_catalogue_card`` /
    ``is_catalogue_card_segment`` / ``full_card`` from a chunk that already
    carries stale card metadata (the chunk keeps its real identity, the card
    only adds its own marker + ``full_card``). The caller's dict is never
    mutated: the card metadata is a fresh dict.
    """
    card = (
        "\n# CATALOGUE CARD\n\n"
        f"## Source file or URL: {source}\n\n"
        "## Initial page\n\n"
        + abstract + "\n\n"
        "## Summary\n\n"
        + summary + "\n"
    )

    card_metadata = {
        **_strip_card_metadata(metadata),
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


def _catalogue_card_point_id(
    agent_key: str,
    chat_id: "Optional[str]",
    source: str,
    index: int = 0,
    start_index: "Optional[int]" = None,
) -> str:
    """Deterministic, tenant/chat-scoped card point id shared by both modes.

    A re-run (or an engine switch) must UPSERT the card, never duplicate it, so
    the id is derived from the source. It is ALSO scoped by the agent and the
    conversation: the same file name uploaded to two agents (or to two chats of
    one agent) must yield two distinct cards instead of overwriting each other.

    A segmented (oversized) card derives its suffix from the segment's stable
    ``start_index`` inside the normalized card text when available, falling back
    to the ordinal ``index``. ``start_index == 0`` keeps the exact classic id,
    so re-runs and re-embeds address the same points even if the split is
    re-enumerated in a different order.
    """
    if start_index is not None:
        suffix = "" if start_index == 0 else f":s{int(start_index)}"
    else:
        suffix = "" if index == 0 else f":{index}"
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"catalogue_card:{agent_key}:{chat_id or ''}:{source}{suffix}",
    ).hex


def _is_url_source(source: str) -> bool:
    return str(source).startswith(("http://", "https://"))


def _decode_source(data: "Optional[bytes]") -> "Optional[str]":
    if data is None:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a preview is best-effort
        return None


async def _read_source_bytes(cat, source: str, chat_id: "Optional[str]") -> "Optional[bytes]":
    """Raw bytes of ``source`` as persisted on disk (file or URL copy).

    Files live under the agent root or the conversation root; URLs are
    persisted by the efficient-ingestion plugin under the deterministic name
    ``_persisted_url_file_name`` (same path: agent root / chat root). ``None``
    when neither copy exists (the caller then falls back to the stored points).
    """
    file_manager = getattr(cat, "file_manager", None)
    if file_manager is None:
        return None
    agent_id = cat.agent_key

    if _is_url_source(source):
        try:
            from cat.core_plugins.cat_efficient_ingestion.reembed import (
                _persisted_url_file_name,
            )
        except Exception:  # noqa: BLE001 - EffING absent/older: no persisted URL copy
            return None
        root = os.path.join(agent_id, str(chat_id)) if chat_id else agent_id
        try:
            persisted_name = _persisted_url_file_name(source, root)
            return file_manager.read_file(persisted_name, root)
        except Exception:  # noqa: BLE001
            return None

    # FILE: the chat copy (when chat-scoped) shadows the agent copy.
    roots = [agent_id]
    if chat_id:
        roots.insert(0, os.path.join(agent_id, str(chat_id)))
    for root in dict.fromkeys(roots):
        try:
            data = file_manager.read_file(source, root)
        except Exception:  # noqa: BLE001
            data = None
        if data is not None:
            return data
    return None


#: Metadata keys that belong to a catalogue card, not to a source chunk. A
#: source chunk can inherit them (a card's metadata is built FROM the first
#: chunk's metadata, and the store can carry stale keys on the chunk after a
#: card round-trip); they must never make an actual chunk look like a card.
_CARD_METADATA_KEYS = (
    "is_catalogue_card",
    "is_catalogue_card_segment",
    "full_card",
)


def _is_catalogue_card_record(meta: Dict, content: str) -> bool:
    """Whether a stored record IS a catalogue card (not a chunk flagged by stale metadata).

    The ``is_catalogue_card`` flag alone is not authoritative: a source chunk
    can carry it when stale card metadata was merged onto the chunk document.
    A real card/segment is, by construction, a slice of its own ``full_card``;
    a genuine source chunk is not. When the flag is set but the content is not
    part of ``full_card``, the record is a source chunk and MUST be summarized.
    Legacy cards without ``full_card`` keep the flag semantics.
    """
    if not meta.get("is_catalogue_card"):
        return False
    full_card = meta.get("full_card")
    text = (content or "").strip()
    if not isinstance(full_card, str) or not full_card.strip() or not text:
        return True
    return " ".join(text.split()) in " ".join(full_card.split())


def _strip_card_metadata(meta: Dict) -> Dict:
    """Drop card-only keys from a chunk that was wrongly flagged as a card.

    Also drops the card-derived ``start_index`` (a segment offset into the card
    text, not a position in the source) so ordering falls back to the chunk's
    real ``chunk_index``.
    """
    cleaned = {key: value for key, value in meta.items() if key not in _CARD_METADATA_KEYS}
    if meta.get("is_catalogue_card_segment"):
        cleaned.pop("start_index", None)
    return cleaned


def _point_order(record) -> Tuple[int, int]:
    """Ordering key for a stored chunk: ``start_index`` then ``chunk_index``."""
    meta = (record.payload or {}).get("metadata", {}) or {}
    for key in ("start_index", "chunk_index"):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            return int(value), 0
    return 1, 0


async def _load_source_chunk_groups(handler, source: str):
    """Stored chunks for ``source`` grouped by (collection, chat_id).

    Chunks live in DECLARATIVE for agent-scoped sources and in EPISODIC for
    chat-scoped ones. A source file with the SAME NAME can exist in several
    conversations (or in the agent KB and a conversation), so the lookup is
    grouped by the chunk's ``chat_id``: each group is summarized and stored as
    its own card, and one chat's text never leaks into another chat's card.

    Within a collection, chunks without a ``chat_id`` form the agent-scoped
    group. Genuine catalogue cards are excluded so a re-run never summarizes its
    own card (see :func:`_is_catalogue_card_record`: the flag alone is not
    authoritative when a real chunk carries stale card metadata), and derived
    image points are excluded because they are not source text. Returns a list
    of ``(collection, chat_id, points)`` sorted by ``start_index`` then
    ``chunk_index``.
    """
    groups = []
    for collection in (str(VectorMemoryType.DECLARATIVE), str(VectorMemoryType.EPISODIC)):
        records, _ = await handler.get_all_tenant_points(
            collection, metadata={"source": source}, with_vectors=False,
        )
        by_chat: Dict[Optional[str], list] = {}
        for record in records or []:
            meta = (record.payload or {}).get("metadata", {}) or {}
            content = (record.payload or {}).get("page_content") or ""
            if meta.get("image") is True:
                continue
            if _is_catalogue_card_record(meta, content):
                continue
            if not content.strip():
                continue
            if meta.get("is_catalogue_card"):
                # Stale card metadata on an actual source chunk: strip it so the
                # new card is built from the chunk's real metadata and the chunk
                # keeps participating in every later run.
                meta = _strip_card_metadata(meta)
                (record.payload or {})["metadata"] = meta
            by_chat.setdefault(meta.get("chat_id"), []).append(record)
        for chat_id in sorted(by_chat, key=lambda value: (value is not None, str(value))):
            groups.append(
                (collection, chat_id, sorted(by_chat[chat_id], key=_point_order))
            )
    return groups


def _segment_card(card: str, card_metadata: Dict, embedder) -> list:
    """Split an oversized card exactly like the classic store path.

    The classic path hands the card Document to the token-budget splitter, so
    every stored segment carries the original metadata (``full_card``
    included). Reuse that splitter; when it is unavailable (EffING absent) the
    card is stored whole.

    Every produced segment is marked ``is_catalogue_card_segment`` with a stable
    ``start_index`` inside the normalized card text. The re-embed path SKIPS
    splitting marked segments, so a later re-embed can never re-split the card
    into a different segment count (which would drop the ``:s<offset>`` id
    suffix); the offset also makes the id independent of enumeration order.
    """
    card_doc = Document(page_content=card, metadata=card_metadata)
    try:
        from cat.core_plugins.cat_efficient_ingestion.split import split_oversized
    except Exception:  # noqa: BLE001 - no splitter: store the card whole
        segments = [card_doc]
    else:
        segments = split_oversized([card_doc], embedder) or [card_doc]

    normalized = " ".join(card.split())
    cursor = 0
    for segment in segments:
        offset = normalized.find(segment.page_content, cursor)
        if offset < 0:
            offset = cursor
        segment.metadata = {
            **segment.metadata,
            "is_catalogue_card_segment": True,
            "start_index": offset,
        }
        cursor = offset + len(segment.page_content)
    return segments


async def _purge_existing_cards(handler, collection, source: str, chat_id, log_prefix: str) -> int:
    """Delete every stored card point of ``source`` in ``collection`` for ``chat_id``.

    A re-run of the ``catalog_cards`` phase can produce a DIFFERENT number of
    segments when the embedder window changed: the ids derive from
    ``start_index``, so the new ids do not address the stored ones and the old
    segments would survive next to the new card. Removing the source's cards in
    THIS collection and THIS chat first makes the write idempotent regardless
    of the previous segmentation.

    Scoped strictly to the source + ``is_catalogue_card`` + the exact
    ``chat_id``: other sources (and other conversations, including the
    agent-scoped ``None`` group) are never touched. The tenant is enforced by
    the handler itself.
    """
    records, _ = await handler.get_all_tenant_points(
        collection,
        metadata={"source": source, "is_catalogue_card": True},
        with_vectors=False,
    )
    stale_ids = []
    for record in records or []:
        payload = record.payload or {}
        meta = payload.get("metadata", {}) or {}
        if not _is_catalogue_card_record(meta, payload.get("page_content") or ""):
            continue
        if meta.get("chat_id") != chat_id:
            continue
        point_id = getattr(record, "id", None) or (record.payload or {}).get("id")
        if point_id is not None:
            stale_ids.append(point_id)
    if stale_ids:
        await handler.delete_tenant_points_by_ids(collection, stale_ids)
        log.info(f"{log_prefix}: replaced {len(stale_ids)} stale card segment(s)")
    return len(stale_ids)


async def _agent_llm_identity(cat) -> str:
    """Stable, secret-free identity of the agent's configured LLM.

    ``<config class>:<model>`` from the agent-scoped ``llm`` service setting;
    default/missing LLMs collapse to a stable sentinel so the phase marker
    stays deterministic. Never exposes credentials.
    """
    try:
        from cat.db.cruds import settings as crud_settings

        saved = await crud_settings.get_settings_by_category(cat.agent_key, "llm")
    except Exception:  # noqa: BLE001
        return "unknown"
    if not saved:
        return "none"
    name = str(saved.get("name") or "")
    value = saved.get("value") or {}
    model = value.get("model") if isinstance(value, dict) else None
    if name and name != "LLMDefaultConfig":
        return f"{name}:{model or ''}"
    return "default"


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

    SHA-256 fingerprint of the agent LLM identity, ``max_document_chars``,
    ``max_summary_words`` and the summary prompt template: any change that can
    alter the produced card invalidates previously generated cards. The cache
    tuning (``cache_ttl_seconds`` / ``cache_max_size``) is deliberately NOT
    part of the marker — it does not change the output. ``None`` for unknown
    phases (conservative stale).
    """
    if phase != "catalog_cards":
        return None
    return await _catalogue_settings_marker(cat)


async def _catalogue_settings_marker(cat) -> str:
    """Configuration-only material fingerprint for the catalogue cards phase.

    Undecorated so both the hook and :class:`CatalogCardsPhase` can call it
    (the ``@hook`` decorator replaces the module-level name with a ``CatHook``
    object, which is not directly callable). No LLM call.
    """
    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    config = (
        int(settings.get("max_document_chars", 8000)),
        int(settings.get("max_summary_words", 200)),
    )
    fingerprint_source = repr(
        (await _agent_llm_identity(cat), config, _SUMMARY_PROMPT_TEMPLATE)
    )
    return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()


_TRANSIENT_ERROR_MARKERS = (
    "Timeout",
    "ConnectionError",
    "ConnectError",
    "RateLimit",
    "ServiceUnavailable",
)


def _is_transient_phase_error(exc: BaseException) -> bool:
    """True only for failures that a bounded retry can plausibly fix.

    A timeout / connection reset / rate limit is retryable; a 404 or a
    model-not-found is permanent. Class-name matching avoids importing the
    provider SDK just to classify its exceptions.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    name = type(exc).__name__
    return any(marker in name for marker in _TRANSIENT_ERROR_MARKERS)


def _phase_failure_status(exc: BaseException) -> Dict[str, Any]:
    """Map an exception to a status dict: transient -> not_ready, else error."""
    reason = f"{type(exc).__name__}: {exc}".strip()[:500]
    if _is_transient_phase_error(exc):
        return {"status": "not_ready", "retry_after": 5, "reason": reason}
    return {"status": "error", "reason": reason}


@hook(priority=10)
async def ingestion_phase_run(phase, source, completed_phases, cat) -> Any:
    """Run the ``catalog_cards`` phase for ``source`` (tri-state contract).

    - ``{"status": "done"}``: the card was embedded and stored as a vector
      point with ``is_catalogue_card: True`` + ``full_card`` metadata;
    - ``{"status": "not_ready", "retry_after": 5}``: the source has no stored
      points yet, or the failure is transient (timeout/connection) — retry
      later;
    - ``{"status": "error", "reason": "..."}``: permanent failure (e.g. a 404 /
      model not found) — the machine marks the phase ERROR;
    - ``None`` for phases this plugin does not implement (hook abstention: the
      dispatcher keeps the value produced by the phase's real owner).
    """
    if phase != "catalog_cards" or CatalogCardsPhase is None:
        return None
    # Thin delegation: the phase class owns the work and its status mapping.
    return _phase_result_as_status(
        await CatalogCardsPhase(cat).process(source, cat)
    )


def _phase_result_as_status(result: Any) -> Any:
    """Translate a class ``PhaseResult`` back to the legacy status-dict contract.

    The legacy hook dispatcher (EffING ``_run_external_phase``) still expects a
    status dict, so the thin hook adapts the class result at the boundary
    instead of re-implementing the run logic. ``retry_after`` is not carried by
    ``PhaseResult``; the engine's default (5s) matches the previous value.
    """
    if _PhaseResult is not None and isinstance(result, _PhaseResult):
        status = {"status": result.status}
        if result.reason is not None:
            status["reason"] = result.reason
        return status
    return result


async def _delete_source_catalogue_cards(cat, source: str) -> int:
    """Delete ONLY ``source``'s catalogue-card points, across both collections.

    Scoped by ``source`` + ``is_catalogue_card`` so the source's chunks, images,
    graph artefacts and every other source are untouched (the tenant is
    enforced by the handler). Idempotent and safe when nothing exists: with no
    matching points no delete is issued and ``0`` is returned.
    """
    handler = cat.vector_memory_handler
    removed = 0
    for collection in (
        str(VectorMemoryType.DECLARATIVE),
        str(VectorMemoryType.EPISODIC),
    ):
        records, _ = await handler.get_all_tenant_points(
            collection,
            metadata={"source": source, "is_catalogue_card": True},
            with_vectors=False,
        )
        stale_ids = []
        for record in records or []:
            payload = record.payload or {}
            meta = payload.get("metadata", {}) or {}
            if not _is_catalogue_card_record(meta, payload.get("page_content") or ""):
                continue
            point_id = getattr(record, "id", None) or payload.get("id")
            if point_id is not None:
                stale_ids.append(point_id)
        if stale_ids:
            await handler.delete_tenant_points_by_ids(collection, stale_ids)
            removed += len(stale_ids)
    return removed


@hook(priority=10)
async def ingestion_phase_purge(phase, source, cat) -> Any:
    """Purge the catalogue-card points of ``source`` (phase purge hook).

    EffING calls this before re-running an abandoned (expired-lease)
    ``catalog_cards`` phase. Deletes ONLY this source's catalogue-card points
    across ALL segments and BOTH collections (declarative + episodic), scoped
    by ``is_catalogue_card`` so the source's chunks, images, graph artifacts
    and every other source are untouched. Returns a truthy status dict after
    purging, and ``None`` for phases this plugin does not implement.
    """
    if phase != "catalog_cards" or CatalogCardsPhase is None:
        return None
    # Thin delegation: the phase class owns its source-scoped delete.
    await CatalogCardsPhase(cat).delete(source, cat)
    return {"status": "done"}


async def _run_catalog_cards_phase(source, cat) -> Dict[str, Any]:
    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    handler = cat.vector_memory_handler

    # The chunks live in DECLARATIVE (agent KB) or EPISODIC (chat-scoped) and the
    # SAME source name can exist in several chats. Grouping by (collection,
    # chat_id) keeps each conversation's text isolated: every group becomes its
    # own card, in the SAME collection, with its own chat_id.
    groups = await _load_source_chunk_groups(handler, source)
    if not groups:
        return {"status": "not_ready", "retry_after": 5}

    embedder = await _resolve_embedder(cat)

    for collection, chat_id, points in groups:
        chunks: List[str] = [
            (rec.payload or {}).get("page_content", "") or "" for rec in points
        ]
        chunks = [content for content in chunks if content.strip()]

        first_metadata: Dict = dict(
            (points[0].payload or {}).get("metadata", {}) or {}
        )
        # EPISODIC scope is carried by the chunk metadata; without it the card
        # cannot be chat-scoped (log and keep going: the chunks still exist).
        if collection == str(VectorMemoryType.EPISODIC) and not chat_id:
            log.warning(
                f"cat_alog: EPISODIC source '{source}' has no chat_id in its stored "
                "points; the catalogue card cannot be chat-scoped."
            )
        if chat_id:
            first_metadata["chat_id"] = chat_id
        else:
            first_metadata.pop("chat_id", None)

        full_text = "\n\n".join(chunks)
        if len(full_text) > max_document_chars:
            full_text = full_text[:max_document_chars] + " [truncated] "

        # Initial page: the source's persisted bytes (file on disk / EffING URL
        # copy) truncated to max_document_chars. When neither is available, fall
        # back to the stored first chunk (start_index, then chunk_index) and say so.
        abstract = None
        source_text = _decode_source(await _read_source_bytes(cat, source, chat_id))
        if source_text is not None and source_text.strip():
            abstract = source_text[:max_document_chars].strip()
        if abstract is None:
            abstract = chunks[0].strip()
            log.warning(
                f"cat_alog: no persisted bytes for '{source}' (chat={chat_id}); "
                "using the stored first chunk as the initial page (best-effort preview)."
            )

        prompt = _build_summary_prompt(source, full_text, max_summary_words)
        summary = await _run_summary_agent(cat, prompt)

        card, card_metadata = _build_catalogue_card(source, first_metadata, abstract, summary)

        # Oversized cards are segmented exactly like the classic store path (the
        # token-budget splitter copies the metadata, so ``full_card`` survives on
        # every segment). Every segment is marked so a re-embed never re-splits it.
        segments = _segment_card(card, card_metadata, embedder)
        vectors = await asyncio.to_thread(
            embedder.embed_documents, [segment.page_content for segment in segments]
        )
        # Deterministic ids: re-runs/engine switches upsert the same card instead
        # of accumulating duplicates (add_points_to_tenant is an upsert). The id
        # is carried in the PAYLOAD too so a re-embed rebuild (which reconstructs
        # points from their payload) keeps the identity instead of minting a new
        # uuid4 — see EffING ``reembed._embed_phase``.
        card_points = []
        for index, (segment, vector) in enumerate(zip(segments, vectors)):
            segment_start = segment.metadata.get("start_index")
            card_id = _catalogue_card_point_id(
                cat.agent_key, chat_id, source, index, start_index=segment_start
            )
            card_points.append(
                PointStruct(
                    id=card_id,
                    payload={
                        "id": card_id,
                        "page_content": segment.page_content,
                        "metadata": segment.metadata,
                    },
                    vector=vector,
                )
            )
        # A re-run with a changed embedder window re-splits the card into a
        # different segment count, so the start_index-derived ids do not match
        # the stored ones. Replace this source's cards in THIS collection/chat
        # first: the write is then idempotent and one source never ends up with
        # two cards.
        await _purge_existing_cards(
            handler, collection, source, chat_id, "cat_alog"
        )
        await handler.add_points_to_tenant(collection, points=card_points)

        log.info(
            f"cat_alog: catalogue card stored for '{source}' "
            f"in {collection} (chat={chat_id}, {len(card_points)} segment(s))"
        )
    return {"status": "done"}


def _phase_result_from_status(phase_id: str, result: Any):
    """Map the legacy status dict to EffING's ``PhaseResult`` (never silent)."""
    if _PhaseResult is None:
        raise RuntimeError("the EffING phase API is not available")
    if isinstance(result, _PhaseResult):
        return result
    if isinstance(result, dict):
        status = result.get("status")
        if status not in ("done", "not_ready", "error", "skipped"):
            return _PhaseResult(
                status="error",
                reason=f"phase '{phase_id}' returned invalid status {status!r}",
            )
        return _PhaseResult(status=status, reason=result.get("reason"))
    if result is None:
        return _PhaseResult(
            status="error", reason=f"phase '{phase_id}' has no run implementation"
        )
    return _PhaseResult(
        status="error",
        reason=f"phase '{phase_id}' returned {result!r} (expected a status dict)",
    )


if _IngestionPhase is not None:

    class CatalogCardsPhase(_IngestionPhase):
        """The ``catalog_cards`` ingestion phase as an EffING phase object.

        Owns only the phase body (delegated to the SAME card-generation path
        the legacy hook uses), a source-scoped ``delete`` and the configuration
        fingerprint. No diary/state/lock/retry logic and no engine gate: the
        phase protocol belongs to EffING. Registered through
        ``ingestion_phase_objects``; the legacy hooks are thin delegations to
        these methods.
        """

        id = "catalog_cards"
        depends_on = ("parsing_chunking", "embedding")
        settings_category = None

        def __init__(self, cat=None):
            super().__init__()
            # Carries the cat that contributed this instance so ``tag`` (whose
            # ABC signature receives only ``settings``) can read the config
            # marker. Instances are created fresh by the registry on every
            # probe, so this is not shared mutable state.
            self._cat = cat

        async def applies_to(self, cat, deps_state) -> Tuple[bool, Optional[str]]:
            """Capability check only: is the catalogue enabled for this agent?

            The catalogue has no per-agent enable flag (its plugin settings are
            global tuning, not an on/off switch), and this phase object only
            exists when EffING is importable. Whether the phase machinery is
            actually active is the ENGINE's decision, computed and owned by
            EffING — never re-derived here.
            """
            return True, None

        async def process(self, source: str, cat):
            """Delegate to the EXISTING card-generation path (no reimplementation)."""
            used = cat if cat is not None else self._cat
            try:
                result = await _run_catalog_cards_phase(source, used)
            except Exception as exc:  # noqa: BLE001 - failure is a status, never a raise
                log.error(
                    f"cat_alog: phase catalog_cards failed for '{source}': "
                    f"{type(exc).__name__}: {exc}"
                )
                failure = _phase_failure_status(exc)
                return _phase_result_from_status(self.id, failure)
            return _phase_result_from_status(self.id, result)

        async def delete(self, source: str, cat) -> None:
            """Remove ONLY ``source``'s card points; idempotent and empty-safe."""
            used = cat if cat is not None else self._cat
            removed = await _delete_source_catalogue_cards(used, source)
            log.info(
                f"cat_alog: phase catalog_cards deleted {removed} "
                f"catalogue-card segment(s) for '{source}'"
            )

        async def tag(self, settings: Any = None) -> str:
            """Configuration-only invalidator: reuse the existing settings marker.

            No LLM call. When the registry already resolved the marker it is
            handed in as ``settings``; a direct call reads it from the cat.
            """
            if settings is None and self._cat is not None:
                settings = await _catalogue_settings_marker(self._cat)
            return await super().tag(settings)


else:
    # Keeps the module surface stable when EffING is absent, without teaching
    # the type checker a second, incompatible definition of the class.
    globals()["CatalogCardsPhase"] = None


@hook(priority=10)
def ingestion_phase_objects(cat=None, lizard=None) -> List[Any]:
    """Contribute the ``CatalogCardsPhase`` instance to EffING's registry.

    Import-safe: with EffING absent the class could not be defined and this
    returns an empty list, leaving the legacy hooks as the only path.
    """
    if CatalogCardsPhase is None:
        return []
    return [CatalogCardsPhase(cat if cat is not None else lizard)]


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

    # Share the phase's deterministic card identity so a re-ingest or an
    # engine switch targets the same card instead of adding a duplicate.
    card_doc = Document(page_content=card, metadata=card_metadata)
    card_doc.id = _catalogue_card_point_id(agent_id, chat_id, source)

    log.debug(f"cat_alog: added catalogue card for '{source}'")
    return docs + [card_doc]


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

    # An image extracted from a source file is SAVED as a file by the
    # multimodal-ingestion plugin, so the FileManager phase above lists it as a
    # file even though it is a derived artifact, never a source. Drop it.
    for image_file in await _derived_image_files(cat):
        files_dict.pop(image_file, None)

    return files_dict


async def _derived_image_files(cat) -> set:
    """Names of image files EXTRACTED from a source (derived, never sources).

    The multimodal-ingestion plugin stores every extracted image as a vector
    point whose metadata carries ``image is True`` and the saved file name in
    ``image_file``; the point also keeps the ORIGINAL source in ``source``.
    A point is a derived image iff its ``image_file`` differs from its
    ``source``: a source that is itself an image is stored whole with
    ``image_file == source`` and must stay listable.
    """
    derived: set = set()
    handler = getattr(cat, "vector_memory_handler", None)
    if handler is None:
        return derived
    for collection in (str(VectorMemoryType.DECLARATIVE), str(VectorMemoryType.EPISODIC)):
        try:
            records, _ = await handler.get_all_tenant_points(
                collection, metadata={"image": True}, with_vectors=False,
            )
        except Exception:
            log.debug(f"cat_alog: image query for {collection} not supported, skipping")
            continue
        for record in records or []:
            meta = (record.payload or {}).get("metadata", {}) or {}
            if meta.get("image") is not True:
                continue
            image_file = meta.get("image_file")
            source = meta.get("source")
            if image_file and image_file != source:
                derived.add(image_file)
    return derived


async def _collect_ingestion_sources(cat) -> Dict[str, Dict[str, Any]]:
    """Phase-path file list: the agent's ingestion status rows are the sources.

    With the EffING phase machine active the status registry is the ground
    truth of what has been ingested: one row per ingested SOURCE (file or URL),
    AGENT scope. Rows only ever describe sources, so images extracted from a
    file have no row and cannot leak into the list; duplicates are impossible
    because the dict is keyed by source (a source without a catalogue card
    still appears exactly once).
    """
    files: Dict[str, Dict[str, Any]] = {}
    try:
        from cat.db import crud
        from cat.db.cruds import settings as crud_settings
    except Exception as e:  # noqa: BLE001 - host without the status API
        log.warning(f"cat_alog: ingestion status API unavailable ({e!r})")
        return files

    agent_id = cat.agent_key
    try:
        keys = await crud_settings.get_ingestion_status_keys(agent_id)
    except Exception as e:  # noqa: BLE001 - best-effort list, never breaks the turn
        log.warning(f"cat_alog: could not list ingestion status keys: {e!r}")
        return files

    for key in keys or []:
        # The deletion marker shares the namespace and is not a status row.
        if str(key).rsplit(":", 1)[-1] == "delete":
            continue
        try:
            doc = await crud.read(key)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, list):
            doc = doc[0] if doc else None
        if not isinstance(doc, dict):
            continue
        source = doc.get("source")
        if not source or doc.get("scope") != "agent":
            continue
        when = doc.get("updated_at") or doc.get("created_at")
        files[source] = {
            "source": source,
            "when": when,
            "content": None,
            "type": "agent",
        }
    return files


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
    """Append the list of files loaded in the agent and conversation to the end of the system prompt.

    ``append_file_list_to_prompt`` is the master switch: when it is False the
    prompt is returned untouched and no file collection (no DB/vector query) is
    performed. When it is True the list is status-first — the EffING phase path
    builds it from the agent's ingestion-status rows, the classic path from the
    FileManager + vector query collector. Derived images never appear in either.
    """
    try:
        settings = await cat.plugin_manager.get_plugin().load_settings()
    except Exception as e:  # noqa: BLE001 - default ON, never break the turn
        log.warning(f"cat_alog: could not load settings for prompt suffix ({e!r})")
        settings = {}
    settings = settings or {}
    if not bool(settings.get("append_file_list_to_prompt", True)):
        return prompt_suffix

    try:
        if await _phase_machine_active(cat):
            files = await _collect_ingestion_sources(cat)
        else:
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
