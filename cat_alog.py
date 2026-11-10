import asyncio
import mimetypes
import os
import uuid
from typing import Any, List, Tuple, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from cat import hook, plugin, tool, log, AgenticWorkflowTask
from cat.rabbit_hole.chunks import read_source_bytes
from cat.rabbit_hole.phases import IngestionPhase, IngestionSource, PhaseResult
from cat.rabbit_hole.split import split_oversized
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


#: Prompt template for the catalogue-card summary. Part of the phase
#: settings: a change to the prompt invalidates previously generated cards.
_SUMMARY_PROMPT_TEMPLATE = """Write a short summary of the following file.
Focus on what the file is, what it is about, and what it contains.
Maximum {max_summary_words} words.

## Filename or source: {source}

## File content

{file_content}
"""


def _build_summary_prompt(source: str, full_text: str, max_summary_words: int) -> str:
    """Build the LLM summary prompt for ``source``.

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

    A failure propagates: the ``catalog_cards`` phase turns it into an ERROR
    result, so the source is not silently completed without its card.
    """
    agent_input = AgenticWorkflowTask(user_prompt=prompt)
    summary_agent_output = await cat.agentic_workflow.run(
        task=agent_input,
        llm=cat.large_language_model,
    )
    return summary_agent_output.output


def _build_catalogue_card(source: str, metadata: Dict, abstract: str, summary: str) -> Tuple[str, Dict]:
    """Build the catalogue card text + metadata.

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



#: Source extensions whose RAW BYTES are readable text, so decoding them gives
#: a usable card preview. Everything in :data:`_BINARY_SOURCE_EXTENSIONS` is a
#: container (PDF/Office/images/archives) whose decoded bytes are compressed
#: garbage — its parsed chunks are the only readable text.
_TEXTUAL_SOURCE_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".text", ".rst", ".rtf",
    ".html", ".htm", ".xhtml", ".xml", ".json", ".jsonl", ".ndjson",
    ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".log", ".tex", ".bib", ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx",
    ".css", ".scss", ".sql", ".sh", ".bash",
})

#: Container formats: never feed their decoded bytes to the LLM. A PDF decoded
#: as UTF-8 starts with ``%PDF`` and continues with binary streams; Office
#: documents are ZIP containers; images are binary by definition.
_BINARY_SOURCE_EXTENSIONS = frozenset({
    ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls", ".odt", ".ods",
    ".odp", ".odg", ".epub",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".svg", ".ico", ".heic", ".avif",
    ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".avi", ".mov", ".mkv",
    ".webm", ".woff", ".woff2", ".ttf", ".otf", ".eot",
})


def _source_extension(source: str) -> str:
    """Lower-case extension of a file name or URL (query/fragment stripped)."""
    path = str(source).split("?", 1)[0].split("#", 1)[0]
    return os.path.splitext(path)[1].lower()


def _is_textual_source(source: str) -> bool:
    """Whether decoding ``source``'s raw bytes yields readable text.

    Used to choose the card's "Initial page": textual sources keep the
    historical raw-bytes preview (their bytes ARE the text), while binary
    sources (PDF/Office/images/archives) must use the parsed chunks instead —
    decoding those bytes yields ~80k chars of compressed-stream garbage that
    the concept/relation LLM correctly refuses to summarize.

    Unknown extensions default to TEXTUAL (raw-bytes preview) so only a known
    binary container changes behaviour. A URL's extension is the path suffix;
    with no extension at all, ``mimetypes`` decides, then textual is the
    conservative fallback.
    """
    extension = _source_extension(source)
    if extension in _TEXTUAL_SOURCE_EXTENSIONS:
        return True
    if extension in _BINARY_SOURCE_EXTENSIONS:
        return False
    guessed, _ = mimetypes.guess_type(str(source).split("?", 1)[0])
    if guessed:
        if guessed.startswith("text/"):
            return True
        if guessed in ("application/json", "application/xml", "application/xhtml+xml"):
            return True
        # Any other concrete binary type (application/pdf, image/png, ...).
        return False
    return True


def _decode_source(data: "Optional[bytes]") -> "Optional[str]":
    if data is None:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a preview is best-effort
        return None


#: Above this share of U+FFFD the "text" is a binary container decoded as UTF-8
#: (PDF/Office/archive streams), never prose. Mirrors GraphRAG's extractor.
_BINARY_REPLACEMENT_RATIO = 0.10


def _looks_binary_text(content: str) -> bool:
    """Whether ``content`` is raw bytes decoded as text rather than prose.

    Such content must never be sent to an LLM (summary or card): a PDF decoded
    with ``errors="replace"`` starts with ``%PDF`` and/or floods U+FFFD and NUL
    bytes. Real text can contain the odd replacement char, so only a material
    share (>= 10%) of the sample marks the content binary.
    """
    if not content:
        return False
    if "\x00" in content:
        return True
    if content.lstrip().startswith("%PDF"):
        return True
    sample = content[:4096]
    if not sample:
        return False
    return (sample.count("\ufffd") / len(sample)) >= _BINARY_REPLACEMENT_RATIO


def _metadata_is_image(meta: Dict) -> bool:
    """Whether a point/document describes an IMAGE, not source text.

    Multimodal ingestion stores image points with ``image: True`` (and the
    saved file name in ``image_file`` / mime in ``image_mime_type``). Their
    ``page_content`` is only ``"[Image] <source>"``: an image chunk must never
    reach an LLM prompt.
    """
    return bool(
        meta.get("image") is True
        or meta.get("image_file")
        or meta.get("image_mime_type")
        or meta.get("image_base64")
    )


async def _read_source_bytes(cat, source: str, chat_id: "Optional[str]") -> "Optional[bytes]":
    """Raw bytes of ``source`` kept by the host file manager.

    The host stores the uploaded file and, for a URL, the bytes it downloaded
    (``cat.rabbit_hole.chunks``). ``None`` when no copy exists: the caller then
    falls back to the stored chunks.
    """
    file_manager = getattr(cat, "file_manager", None)
    if file_manager is None:
        return None
    try:
        return read_source_bytes(file_manager, cat.agent_key, chat_id, source)
    except Exception:  # noqa: BLE001 - no persisted copy is not an error
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


def _record_point_id(record) -> "Optional[str]":
    """The stored point id of a record (point level, payload fallback)."""
    point_id = getattr(record, "id", None) or (record.payload or {}).get("id")
    return str(point_id) if point_id is not None else None


def _normalize_point_id(value: Any) -> str:
    """Backend-agnostic spelling of a point id for identity comparison.

    The product mints card ids as ``uuid5(...).hex`` (undashed), while a
    vector store may hand the SAME uuid back in canonical dashed form (Qdrant
    does). Comparing the raw strings then fails to recognise a card the
    product itself wrote, so the phase purge would miss every stored segment
    and leave orphans behind. Normalising both sides keeps the identity check
    independent of the store's formatting.
    """
    return str(value).replace("-", "").lower()


def _catalogue_card_id_from_metadata(meta: Dict, agent_key: str) -> "Optional[str]":
    """The deterministic card id a record's OWN metadata describes.

    Mirrors the write path exactly: ``chat_id`` scopes the card, a segment's
    ``start_index`` selects its ``:s<start_index>`` suffix, and a plain card
    falls back to its ``chunk_index`` (always 0 for the first-chunk-derived
    card). Returns ``None`` when the record cannot describe a card at all.
    """
    source = meta.get("source")
    if not source:
        return None
    index = meta.get("chunk_index")
    if isinstance(index, bool) or not isinstance(index, int):
        index = 0
    start_index = meta.get("start_index")
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        start_index = None
    return _catalogue_card_point_id(
        agent_key, meta.get("chat_id"), str(source), index, start_index=start_index,
    )


def _is_catalogue_card_record(meta: Dict, point_id: "Optional[str]", agent_key: str) -> bool:
    """Whether a stored record IS a catalogue card, by IDENTITY (not content).

    The ``is_catalogue_card`` flag alone is not authoritative: an actual source
    chunk can carry it when stale card metadata was merged onto the chunk
    document. The ONLY authoritative test is the point id: a card (or one of
    its segments) is written at the id :func:`_catalogue_card_point_id` derives
    from its own ``source``/``chat_id``/``chunk_index``/``start_index``. Anything
    else that merely carries the flag is a source chunk, never a card.
    """
    if not meta.get("is_catalogue_card") or point_id is None:
        return False
    expected = _catalogue_card_id_from_metadata(meta, agent_key)
    return expected is not None and _normalize_point_id(point_id) == _normalize_point_id(expected)


def _strip_card_metadata_in_place(meta: Dict) -> Dict:
    """Drop card-only keys from a chunk wrongly flagged as a card, in place.

    The stored chunk's metadata dict is the one the handler returned; cleaning
    it IN PLACE (instead of replacing it with a copy) guarantees every other
    reference to that dict — the card seed and any consumer still holding the
    record — observes the cleaned metadata. Also drops the card-derived
    ``start_index`` (a segment offset into the card text, not a position in the
    source) so ordering falls back to the chunk's real ``chunk_index``.
    """
    was_card_segment = bool(meta.get("is_catalogue_card_segment"))
    for key in _CARD_METADATA_KEYS:
        meta.pop(key, None)
    if was_card_segment:
        meta.pop("start_index", None)
    return meta


def _strip_card_metadata(meta: Dict) -> Dict:
    """Return a copy of ``meta`` without the card-only keys (caller untouched).

    Used to seed the card so it can never inherit ``is_catalogue_card`` /
    ``is_catalogue_card_segment`` / ``full_card`` (and a segment's
    ``start_index``) from a chunk. The caller's dict is never mutated.
    """
    return _strip_card_metadata_in_place(dict(meta))


def _point_order(record) -> Tuple[int, int]:
    """Ordering key for a stored chunk: ``start_index`` then ``chunk_index``."""
    meta = (record.payload or {}).get("metadata", {}) or {}
    for key in ("start_index", "chunk_index"):
        value = meta.get(key)
        if isinstance(value, (int, float)):
            return int(value), 0
    return 1, 0


async def _load_source_chunk_groups(handler, source: str, agent_key: str):
    """Stored chunks for ``source`` grouped by (collection, chat_id).

    Chunks live in DECLARATIVE for agent-scoped sources and in EPISODIC for
    chat-scoped ones. A source file with the SAME NAME can exist in several
    conversations (or in the agent KB and a conversation), so the lookup is
    grouped by the chunk's ``chat_id``: each group is summarized and stored as
    its own card, and one chat's text never leaks into another chat's card.

    Within a collection, chunks without a ``chat_id`` form the agent-scoped
    group. Catalogue cards are excluded by IDENTITY (see
    :func:`_is_catalogue_card_record`: only a point at the deterministic card
    id is a card, the flag alone is not authoritative), and derived image
    points are excluded because they are not source text. Returns a list of
    ``(collection, chat_id, points)`` sorted by ``start_index`` then
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
            if _metadata_is_image(meta):
                continue
            if _is_catalogue_card_record(meta, _record_point_id(record), agent_key):
                continue
            if not content.strip():
                continue
            if _looks_binary_text(content):
                # A binary container decoded as text (legacy/broken parser):
                # its bytes must never be summarized or embedded in a card.
                continue
            if meta.get("is_catalogue_card"):
                # Stale card metadata on an actual source chunk (a legacy
                # card-write leak): strip it IN PLACE so this record's own
                # metadata dict — the card seed and any consumer still holding
                # the record — no longer carries the card flags. The chunk keeps
                # its real identity and participates in every later run.
                _strip_card_metadata_in_place(meta)
            by_chat.setdefault(meta.get("chat_id"), []).append(record)
        for chat_id in sorted(by_chat, key=lambda value: (value is not None, str(value))):
            groups.append(
                (collection, chat_id, sorted(by_chat[chat_id], key=_point_order))
            )
    return groups


def _segment_card(card: str, card_metadata: Dict, embedder) -> list:
    """Split an oversized card with the core token-budget splitter.

    Every stored segment carries the original metadata (``full_card``
    included). The budget is the active embedder's input window: a missing or
    non-positive limit disables the split and the card is stored whole.

    Every produced segment is marked ``is_catalogue_card_segment`` with a stable
    ``start_index`` inside the normalized card text. The re-embed path SKIPS
    splitting marked segments, so a later re-embed can never re-split the card
    into a different segment count (which would drop the ``:s<offset>`` id
    suffix); the offset also makes the id independent of enumeration order.
    """
    card_doc = Document(page_content=card, metadata=card_metadata)
    max_tokens = getattr(embedder, "max_input_tokens", None) or 0
    tokenizer = getattr(embedder, "_estimate_tokens", None)
    segments = split_oversized([card_doc], max_tokens, tokenizer) or [card_doc]

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


async def _purge_existing_cards(handler, collection, source: str, chat_id, log_prefix: str, agent_key: str) -> int:
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
        if not _is_catalogue_card_record(meta, _record_point_id(record), agent_key):
            continue
        if meta.get("chat_id") != chat_id:
            continue
        point_id = _record_point_id(record)
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


def _phase_failure_result(exc: BaseException) -> PhaseResult:
    """Map an exception to a phase result: transient -> not_ready, else error."""
    reason = f"{type(exc).__name__}: {exc}".strip()[:500]
    if _is_transient_phase_error(exc):
        return PhaseResult(status="not_ready", retry_after=5, reason=reason)
    return PhaseResult(status="error", reason=reason)


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
            if not _is_catalogue_card_record(meta, _record_point_id(record), cat.agent_key):
                continue
            point_id = _record_point_id(record)
            if point_id is not None:
                stale_ids.append(point_id)
        if stale_ids:
            await handler.delete_tenant_points_by_ids(collection, stale_ids)
            removed += len(stale_ids)
    return removed


async def _run_catalog_cards_phase(source: str, cat) -> PhaseResult:
    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    handler = cat.vector_memory_handler

    # The chunks live in DECLARATIVE (agent KB) or EPISODIC (chat-scoped) and the
    # SAME source name can exist in several chats. Grouping by (collection,
    # chat_id) keeps each conversation's text isolated: every group becomes its
    # own card, in the SAME collection, with its own chat_id.
    groups = await _load_source_chunk_groups(handler, source, cat.agent_key)
    if not groups:
        return PhaseResult(
            status="not_ready",
            retry_after=5,
            reason=f"no stored chunks for '{source}'",
        )

    embedder = await _resolve_embedder(cat)
    total_chunks = 0
    produced = 0

    for collection, chat_id, points in groups:
        chunks: List[str] = [
            (rec.payload or {}).get("page_content", "") or "" for rec in points
        ]
        chunks = [content for content in chunks if content.strip()]
        total_chunks += len(chunks)

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

        # Initial page. For a TEXTUAL source the raw persisted bytes (uploaded
        # file / host URL copy) are themselves readable text: keep the
        # historical raw-bytes preview, truncated to max_document_chars. For a
        # BINARY source (PDF/Office/image/archive) decoding those bytes yields
        # compressed-stream garbage (`%PDF-1.5 …`, replacement chars), so the
        # initial page MUST be the PARSED text instead: the concatenation of the
        # source's stored chunks, already fetched above and capped at
        # max_document_chars by ``full_text``. When a textual source has no
        # persisted copy, the parsed text is the best-effort fallback.
        abstract = None
        if _is_textual_source(source):
            source_text = _decode_source(await _read_source_bytes(cat, source, chat_id))
            if (
                source_text is not None
                and source_text.strip()
                and not _looks_binary_text(source_text)
            ):
                abstract = source_text[:max_document_chars].strip()
                log.info(
                    f"cat_alog: initial page for '{source}' (chat={chat_id}) from "
                    "the persisted raw bytes (textual source)."
                )
            elif source_text is not None and _looks_binary_text(source_text):
                log.warning(
                    f"cat_alog: persisted bytes for '{source}' (chat={chat_id}) look "
                    "binary despite a textual extension; using the parsed chunks."
                )
        if abstract is None:
            abstract = full_text.strip()
            log.info(
                f"cat_alog: initial page for '{source}' (chat={chat_id}) from the "
                f"parsed chunks (binary source or no persisted bytes), "
                f"{len(abstract)} chars."
            )

        prompt = _build_summary_prompt(source, full_text, max_summary_words)
        summary = await _run_summary_agent(cat, prompt)

        card, card_metadata = _build_catalogue_card(source, first_metadata, abstract, summary)

        # Oversized cards are segmented with the core token-budget splitter (it
        # copies the metadata, so ``full_card`` survives on every segment).
        # Every segment is marked so a re-embed never re-splits it.
        segments = _segment_card(card, card_metadata, embedder)
        vectors = await asyncio.to_thread(
            embedder.embed_documents, [segment.page_content for segment in segments]
        )
        # Deterministic ids: re-runs upsert the same card instead of accumulating
        # duplicates (add_points_to_tenant is an upsert). The id is carried in
        # the PAYLOAD too, so a re-embed rebuild (which reconstructs points from
        # their payload) keeps the identity instead of minting a new uuid4.
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
            handler, collection, source, chat_id, "cat_alog", cat.agent_key
        )
        await handler.add_points_to_tenant(collection, points=card_points)
        produced += len(card_points)

        log.info(
            f"cat_alog: catalogue card stored for '{source}' "
            f"in {collection} (chat={chat_id}, {len(card_points)} segment(s))"
        )

    # The phase RAN (the source's chunks exist and every call succeeded) but no
    # card came out: a tiny/structural document may legitimately extract to
    # nothing. That is a WARNING on a ``done`` result, not an error — it is
    # complete, so it is not retried and no dependent is blocked. "Could not
    # run" (no chunks above, LLM failure) keeps its not_ready/error path.
    if produced == 0:
        return PhaseResult(
            status="done",
            warnings=[
                {
                    "code": "empty-extraction",
                    "message": (
                        f"no catalogue card produced for '{source}' "
                        f"from {total_chunks} chunk(s)"
                    ),
                }
            ],
            processed=0,
        )
    return PhaseResult(status="done")


class CatalogCardsPhase(IngestionPhase):
    """The ``catalog_cards`` phase of the core ingestion protocol.

    Runs after ``embedding``: it reads the source's stored chunks, summarizes
    them and stores one catalogue card per (collection, chat). It owns only its
    body and its source-scoped cleanup — the core owns the diary, staleness,
    retries and locks.
    """

    id = "catalog_cards"
    label = "Catalogue cards"
    depends_on = ("embedding",)

    async def settings(self, cat) -> Dict[str, Any]:
        """The material configuration: when it changes, the card is regenerated.

        The agent LLM identity, the size limits and the summary prompt can all
        change the produced card, so they are part of the phase tag; the
        prompt-list switch does not.
        """
        settings = await cat.plugin_manager.get_plugin().load_settings()
        settings = settings or {}
        return {
            "max_document_chars": int(settings.get("max_document_chars", 8000)),
            "max_summary_words": int(settings.get("max_summary_words", 200)),
            "llm": await _agent_llm_identity(cat),
            "prompt": _SUMMARY_PROMPT_TEMPLATE,
        }

    async def applies_to(self, source: IngestionSource, cat) -> Tuple[bool, Optional[str]]:
        """Capability check only: the catalogue has no per-agent enable flag."""
        return True, None

    async def process(self, source: IngestionSource, cat) -> PhaseResult:
        """Build and store the source's cards (same generator as before)."""
        try:
            return await _run_catalog_cards_phase(source.name, cat)
        except Exception as exc:  # noqa: BLE001 - failure is a result, never a raise
            log.error(
                f"cat_alog: phase catalog_cards failed for '{source.name}': "
                f"{type(exc).__name__}: {exc}"
            )
            return _phase_failure_result(exc)

    async def delete(self, source: IngestionSource, cat) -> None:
        """Remove ONLY ``source``'s card points; idempotent and empty-safe."""
        removed = await _delete_source_catalogue_cards(cat, source.name)
        log.info(
            f"cat_alog: phase catalog_cards deleted {removed} "
            f"catalogue-card segment(s) for '{source.name}'"
        )


@hook(priority=10)
def rabbithole_ingestion_phases(phases: List, cat) -> List:
    """Register the ``catalog_cards`` phase on the core ingestion protocol.

    The hook receives and returns phase CLASSES (the host instantiates them
    without arguments for every run), so the class is appended, never an
    instance.
    """
    phases = list(phases or [])
    phases.append(CatalogCardsPhase)
    return phases


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

                if _metadata_is_image(meta):
                    # An image point's page_content is only "[Image] <source>":
                    # an image chunk must never be forwarded to the LLM (this
                    # preview becomes tool output in the conversation).
                    continue

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
                            record_meta = (records[0].payload.get("metadata") or {})
                            if _metadata_is_image(record_meta):
                                # Never preview an image point's "[Image] …"
                                # placeholder: image chunks are not source text.
                                continue
                            content = (records[0].payload.get("page_content") or "")[:500]
                            if _looks_binary_text(content):
                                continue
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
    """File list for the prompt suffix: the agent's ingestion status rows.

    The status registry is the ground truth of what has been ingested: one row
    per ingested SOURCE (file or URL),
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
        if not source:
            continue
        # The key encodes the scope: ``:ingestion:agent:<sha>`` for the agent
        # knowledge base, ``:ingestion:chat:<chat_id>:<sha>`` for a chat. Only
        # the agent-scoped sources belong to the prompt list.
        if ":ingestion:chat:" in str(key):
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
    prompt is returned untouched and no file collection (no DB query) is
    performed. When it is True the list is built from the agent's ingestion
    status rows, so it contains sources only: derived images never appear.
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
        files = await _collect_ingestion_sources(cat)
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
