# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

The catalog card contains:
- the file metadata
- the first chunk of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

## Two ingestion modes and the gate rule

Card generation has two mutually exclusive implementations, selected per ingestion by `_phase_machine_active(cat)`:

| [EffING](https://github.com/DEH-RAG/cat-efficient-ingestion) phases importable from the canonical path | SYSTEM-level `ingestion` engine | Mode used |
|---|---|---|
| no | any | **classic** (in-memory hooks) |
| yes | core engine | **classic** |
| yes | `EfficientIngestionConfiguration` | **`catalog_cards` phase** |

The gate is true if and only if EffING is installed **and** the SYSTEM-level `ingestion` setting selects the efficient engine. `ingestion` is a system-level service (like `embedder`), so the decision is always read from the system settings, never from an agent's settings. When the gate is active, the two `before_rabbithole_*` hooks are no-ops and the phase machine owns card generation; otherwise the classic in-memory hooks run and the phase registrations are inert. The two paths can never run together.

## Catalog cards

A catalog card is stored in the vector database so the cat can discuss *about* the files loaded, not only their content. It contains:
- the file metadata
- the initial page of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

### The `catalog_cards` phase

On the efficient engine the card is produced by the **`catalog_cards`** ingestion phase, declared through `ingestion_phase_specs` with `settings_category=None` and `depends_on=("parsing_chunking", "embedding")`. It therefore re-runs automatically when the source's chunks or vectors change. The same phase is contributed as a `CatalogCardsPhase` instance through `ingestion_phase_objects`: `applies_to` is a **capability check only** — the catalogue has no per-agent enable flag and the phase object only exists when EffING is importable, so it returns true; whether the phase machinery is active is EffING's engine decision and is never re-derived inside the phase class (the classic-vs-phase gate stays in the classic hooks' `_phase_machine_active`). `process` delegates to the existing card-generation path, `delete` removes only that source's `is_catalogue_card` points in both collections (idempotent, empty-safe) and `tag` reuses the configuration-only settings marker below. The phase class owns no diary/state/lock/retry logic: the legacy `ingestion_phase_run` / `ingestion_phase_purge` hooks are thin delegations to the class methods (the marker hook shares the class's configuration fingerprint). `ingestion_phase_settings_marker` returns a SHA-256 fingerprint of the agent LLM identity, `max_document_chars`, `max_summary_words` and the summary prompt template; cache tuning is excluded because it cannot change the output. Any material change invalidates the cards produced before it.

The phase behaves as follows:

- **Initial page**: the source's persisted bytes (the file on disk, or the URL copy persisted by EffING) truncated to `max_document_chars`. When no copy exists it logs a warning and falls back to the stored first chunk (ordered by `start_index`, then `chunk_index`).
- **Collection scope**: the card is written to the SAME collection as its chunks — `DECLARATIVE` for agent-scoped sources, `EPISODIC` for chat-scoped ones — and copies `chat_id` from the chunk metadata. Stored chunks are grouped by `(collection, chat_id)`, so the SAME file name in two chats produces one card per chat containing only that chat's text (never a merged card). An EPISODIC source without `chat_id` logs a warning because the card cannot be chat-scoped.
- **Oversized cards** are segmented with EffING's `split_oversized` splitter, exactly like the classic store path; every segment carries `full_card` and is marked `is_catalogue_card_segment: True` with a stable `start_index`. A re-embed skips marked segments, so it can never re-split the card into a different segment count and lose its ids.
- **Deterministic id**: `uuid5("catalogue_card:<agent_key>:<chat_id>:<source>")`, with a stable `:s<start_index>` suffix for segments after the first (`:<index>` only when no offset is available). Re-runs, engine switches and re-embeds therefore UPSERT the same card instead of accumulating duplicates.
- **Tri-state contract**: `{"status": "done"}` when the card is stored, `{"status": "not_ready", "retry_after": 5}` while the source's points are still absent, and an exception is a permanent failure (ERROR).

## Settings

| Key | Default | Description |
|-----|---------|-------------|
| `append_file_list_to_prompt` | `True` | Append the list of files loaded in the agent and conversation to the end of the system prompt. When `False` the prompt is returned untouched and no file collection (no DB/vector query) is performed. |
| `max_document_chars` | `8000` | Max characters sent to the LLM for file summarization |
| `max_summary_words` | `200` | Max words for the summary |
| `cache_ttl_seconds` | `3600` | TTL for classic-path cache entries in seconds |
| `cache_max_size` | `100` | Max entries in the classic-path LRU cache |

With `append_file_list_to_prompt` enabled, `agent_prompt_suffix` appends a compact status-first list (names, type, upload date; no previews) to the END of the system prompt. On the phase path the list is built from the agent's ingestion-status rows (`_collect_ingestion_sources`); on the classic path from the FileManager plus vector-query collector. Images extracted from files are never listed: they are derived artifacts, not sources, and are filtered by the multimodal-ingestion marker (`image is True` and `image_file != source`). The `list_loaded_files` tool is unchanged and reads the stored cards (falling back to first chunks) whichever path generated them.

`chunk_index` and `start_index` are stamped by the chunker, not by this plugin; Cat_alog only reads them to order chunks and find the first chunk.

## Commands

- Run the standalone test suite: `python test_catalog_phase.py` (prints `test_catalog_phase: ALL ASSERTIONS PASSED` on success)

