# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

The catalog card contains:
- the file metadata
- the first chunk of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

## Loaded files in the prompt

The list shows each file's name, type (agent or conversation) and upload date, with no content previews. It is a no-op when nothing is loaded. The `list_loaded_files` tool remains the on-demand source of full content previews.

## One ingestion path

Every source goes through the core phase ingestion (`cat.rabbit_hole.phases`).
This plugin contributes one phase, `catalog_cards`, through the core hook
`rabbithole_ingestion_phases(phases, cat)`: it depends on `embedding` and
therefore runs after the source's chunks have been parsed, chunked and
embedded. There is no second (in-memory hook) implementation and no engine
selector: the phase is the only card producer.

## Catalog cards

A catalog card is stored in the vector database so the cat can discuss *about* the files loaded, not only their content. It contains:
- the file metadata
- the initial page of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

### The `catalog_cards` phase

`CatalogCardsPhase` implements the core `IngestionPhase` ABC: `id = "catalog_cards"`, `depends_on = ("embedding",)`, `settings(cat)` returns the material configuration (agent LLM identity, size limits, summary prompt), `applies_to` is a capability check (the catalogue has no per-agent enable flag, so it always applies), `process` builds and stores the cards, and `delete` removes only that source's `is_catalogue_card` points in both collections (idempotent, empty-safe). The core owns the diary, staleness, retries and locks; the class owns only its body and cleanup.

The phase behaves as follows:

- **Initial page**: for a **textual** source (`.md`, `.txt`, `.html`, `.csv`, …) the persisted bytes (the uploaded file, or the copy the host keeps for a URL) truncated to `max_document_chars`; for a **binary** source (`.pdf`, Office, images, archives) the **parsed text** instead — the source's stored chunks concatenated and capped at `max_document_chars` — because decoding a container's bytes yields compressed-stream garbage that the LLM cannot summarize. Unknown extensions keep the textual behaviour. When a textual source has no persisted copy it also falls back to the parsed chunks.
- **Collection scope**: the card is written to the SAME collection as its chunks — `DECLARATIVE` for agent-scoped sources, `EPISODIC` for chat-scoped ones — and copies `chat_id` from the chunk metadata. Stored chunks are grouped by `(collection, chat_id)`, so the SAME file name in two chats produces one card per chat containing only that chat's text (never a merged card). An EPISODIC source without `chat_id` logs a warning because the card cannot be chat-scoped.
- **Oversized cards** are segmented with the core `cat.rabbit_hole.split.split_oversized` splitter, using the active embedder's input window; every segment carries `full_card` and is marked `is_catalogue_card_segment: True` with a stable `start_index`. The re-embed path skips marked segments, so a re-embed does not re-split the card into a different segment count and lose its ids.
- **Deterministic id**: `uuid5("catalogue_card:<agent_key>:<chat_id>:<source>")`, with a stable `:s<start_index>` suffix for segments after the first (`:<index>` only when no offset is available). Re-runs and re-ingests therefore UPSERT the same card instead of accumulating duplicates.
- **Card identity is the point id, not the metadata flags**: a stored record is a catalogue card only when its point id equals the deterministic id its own `source`/`chat_id`/`chunk_index`/`start_index` describes. A source chunk that merely carries `is_catalogue_card`/`full_card` (stale or inherited metadata) is a chunk: the card-only keys are stripped and it is summarized normally, so a single-chunk source can never make the phase loop `not_ready`. The same identity test guards both purge paths, so a chunk is never deleted as a card.
- **Result contract**: `PhaseResult(status="done")` when the card is stored; `PhaseResult(status="done", warnings=[{"code": "empty-extraction", "message": "..."}], processed=0)` when the phase RAN over existing chunks but produced no card (a tiny/structural document may legitimately yield none — the phase is complete, not retried, and dependents are not blocked); `PhaseResult(status="not_ready", retry_after=5)` while the source's points are still absent; and an exception (e.g. the LLM failing) is a permanent `PhaseResult(status="error", reason=...)`.

## Settings

| Key | Default | Description |
|-----|---------|-------------|
| `append_file_list_to_prompt` | `True` | Append the list of files loaded in the agent and conversation to the end of the system prompt. When `False` the prompt is returned untouched and no file collection (no DB/vector query) is performed. |
| `max_document_chars` | `8000` | Max characters sent to the LLM for file summarization |
| `max_summary_words` | `200` | Max words for the summary |

With `append_file_list_to_prompt` enabled, `agent_prompt_suffix` appends a compact list (names, type, upload date; no previews) to the END of the system prompt. The list is built from the agent's ingestion-status rows (`_collect_ingestion_sources`), so it contains sources only: images extracted from files are derived artifacts and never appear. The `list_loaded_files` tool reads the stored cards (falling back to first chunks) and filters the same derived images.

`chunk_index` and `start_index` are stamped by the chunker, not by this plugin; Cat_alog only reads them to order chunks and find the first chunk.

## Commands

- Run the standalone test suite: `python tests/test_catalog_phase.py` (prints `test_catalog_phase: ALL ASSERTIONS PASSED` on success)
- Run the real-plugin-manager integration test from the MyCAT integration checkout: `cd /root/git/grinning-cat-core && .venv/bin/python /root/git/plugins/cat_alog/tests/test_catalog_phase_integration.py`
- Run the end-to-end suite (real app, Redis db 15, Neo4j): `cd /root/git/grinning-cat-core && .venv/bin/python /root/git/plugins/cat_alog/tests/test_dual_mode.py` (prints `test_dual_mode: 13/13 passed` on success)
- Run the leaked-card-metadata cleanup (dry-run by default; `--yes` writes): `python scripts/fix_leaked_card_metadata.py --dry-run`

