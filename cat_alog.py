import asyncio
import os
from typing import Any, List, Tuple, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from cat import hook, plugin, tool, log, AgenticWorkflowTask
from cat.services.memory.models import VectorMemoryType
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


#: Prompt template for the catalogue-card summary.
_SUMMARY_PROMPT_TEMPLATE = """Write a short summary of the following file.
Focus on what the file is, what it is about, and what it contains.
Maximum {max_summary_words} words.

## Filename or source: {source}

## File content

{file_content}
"""


def _build_summary_prompt(source: str, full_text: str, max_summary_words: int) -> str:
    """Build the LLM summary prompt for ``source``.

    The file content is brace-escaped: the agentic workflow reads the user
    prompt as a template (``HumanMessagePromptTemplate.from_template``), where a
    brace of the content would be a placeholder.
    """
    safe_text = full_text.replace("{", "{{").replace("}", "}}")
    return _SUMMARY_PROMPT_TEMPLATE.format(
        max_summary_words=max_summary_words,
        source=source,
        file_content=safe_text,
    )


async def _run_summary_agent(cat, prompt: str) -> str:
    """Run the LLM summary agent for ``prompt`` (the caller handles the failures)."""
    agent_input = AgenticWorkflowTask(user_prompt=prompt)
    summary_agent_output = await cat.agentic_workflow.run(
        task=agent_input,
        llm=cat.large_language_model,
    )
    return summary_agent_output.output


def _build_catalogue_card(source: str, metadata: Dict, abstract: str, summary: str) -> Tuple[str, Dict]:
    """Build the catalogue card text + metadata.

    The metadata come from the first text chunk of the source (its ``source``, ``chat_id``, ``when``, ``hash`` and the
    custom metadata of the upload). The caller's dict is never mutated: the card metadata is a fresh dict.
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
        **metadata,
        "is_catalogue_card": True,
        # Store the FULL card text on every card point. When the card exceeds
        # the embedder's input window it may be split into segments (the
        # ``finalize_oversized_chunks`` hook, each segment with its own embedding
        # and metadata); with this field every segment carries the complete card,
        # so any similarity hit on any part of the card retrieves the whole card.
        "full_card": card,
    }
    return card, card_metadata


#: Above this share of U+FFFD the "text" is a binary container decoded as UTF-8
#: (PDF/Office/archive streams), never prose.
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
    return (sample.count("�") / len(sample)) >= _BINARY_REPLACEMENT_RATIO


def _metadata_is_image(meta: Dict) -> bool:
    """Whether a point/document describes an IMAGE, not source text.

    A multimodal plugin stores image points with ``image: True`` (and the
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


def _metadata_is_catalogue_card(meta: Dict) -> bool:
    """Whether a point/document carries any catalogue-card marker/metadata."""
    return bool(meta.get("is_catalogue_card") or meta.get("full_card"))


def _is_llm_text_document(doc) -> bool:
    """Whether a chunk is source TEXT safe to send to an LLM."""
    meta = getattr(doc, "metadata", None) or {}
    content = getattr(doc, "page_content", "") or ""
    if not content.strip():
        return False
    if _metadata_is_image(meta) or _metadata_is_catalogue_card(meta):
        return False
    return not _looks_binary_text(content)


@hook(priority=10)
async def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """Summarize the chunks of the source and append its catalogue card.

    The RabbitHole fires this hook once per parsing of a source (its ``parsing_chunking`` phase: an upload, a change of
    the chunker, a re-ingestion), on the chunks it keeps for the source: the card is kept with them, so a re-embedding
    (e.g. a change of the embedder) embeds it again without calling the LLM, and every parsing replaces the card of the
    previous one with the points of the source. The chunks already carry their metadata (``source``, ``chat_id``, ...).
    ``RabbitHole.store_documents`` (for callers holding their documents already) fires it too: there the card is stored
    with the documents, not kept.
    """
    # ONLY the text chunks are summarized: image chunks (``[Image] …``
    # placeholders), catalogue cards and any binary-looking content must never
    # reach the summary LLM nor the card.
    text_docs = [doc for doc in docs if _is_llm_text_document(doc)]
    if not text_docs:
        if docs:
            log.warning(
                f"cat_alog: no readable text chunks for '{docs[0].metadata.get('source')}', skipping the catalogue card."
            )
        return docs

    # Add positional index to identify the first chunk (chunk_index=0)
    # Only set if not already present (respects prior settings from chunker/plugins).
    for i, doc in enumerate(docs):
        doc.metadata.setdefault("chunk_index", i)

    metadata = text_docs[0].metadata
    source = metadata["source"]

    settings = await cat.plugin_manager.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    full_text = "\n\n".join(doc.page_content for doc in text_docs)
    if len(full_text) > max_document_chars:
        full_text = full_text[:max_document_chars] + ' [truncated] '

    try:
        summary = await _run_summary_agent(cat, _build_summary_prompt(source, full_text, max_summary_words))
    except Exception as e:
        log.warning(f"cat_alog: failed to summarize '{source}': {e}")
        summary = "(summary not available)"

    # The card's initial page is source TEXT, never an image placeholder, a
    # previously built card, or decoded binary bytes.
    card, card_metadata = _build_catalogue_card(source, metadata, text_docs[0].page_content.strip(), summary)

    log.debug(f"cat_alog: added catalogue card for '{source}' [{summary[:100]}...]")
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

    # An image extracted from a source file may be SAVED as a file by a
    # multimodal plugin, so the FileManager phase above lists it as a
    # file even though it is a derived artifact, never a source. Drop it.
    for image_file in await _derived_image_files(cat):
        files_dict.pop(image_file, None)

    return files_dict


async def _derived_image_files(cat) -> set:
    """Names of image files EXTRACTED from a source (derived, never sources).

    A multimodal plugin stores every extracted image as a vector
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
    performed. When it is True the list (names, type, upload date) comes from the
    FileManager + vector query collector. Derived images never appear in it.
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
