# CAT-alog

Cheshire Cat plugin that adds "catalogue cards" to the vector DB so the AI can discuss
*about* loaded files (metadata, first chunk, summary). Also provides a tool to list files
and injects file references into `message.why` so RITA renders them as download links.

## Entrypoint

`cat_alog.py` — single-file plugin.

## Hooks & tools

| Priority | Hook/Tool | Purpose |
|---|---|---|
| 9 | `before_cat_sends_message` | Summarises files via LLM (insert_summary_if_not_exists) |
| 10 | `before_rabbithole_stores_documents` | Appends catalogue card with metadata + summary |
| 0 | `before_cat_sends_message` | Injects file names into `message.why.memory` for RITA clickable links |
| — | `@tool list_loaded_files` | Lists all files/URLs (agent + conversation) with upload date + preview |

## Quirks

- **`asyncio.run()` inside sync tool** — `list_loaded_files` uses `asyncio.run(_get_files())`
  to query the vector DB from a sync `@tool` function.
- **3-phase file listing**: FileManager (fast, no DB) → targeted DB queries (content previews)
  → legacy fallback (full DB scan via metadata.source substring match).
- **TTL_LRU_Cache** — summaries cached for 1 hour, keyed by `(agent_id, chat_id, filename)`.
  Cache entry is deleted once the catalogue card is appended to documents.
- **CAT priority = higher runs first**. Priority 9 (file summarization) and priority 1 (why core)
  run before priority 0 (why injection). The injection hook **must** run after the why hook
  because it appends to `message.why.memory`.
- **chat_id handling** — uses `getattr(cat, "id", None)` for current conversation, falls back
  to agent scope for file lookups.

## Dependencies

- `langchain-core>=1.3.0` (for `Document`)
- Runtime: Cheshire Cat (grinning-cat-core) with its plugin system

## Settings (pydantic)

- `max_document_chars` (8000) — chars sent to LLM for file summarisation
- `max_summary_words` (200) — max words in generated summary
- `cache_ttl_seconds` (3600) / `cache_max_size` (100) — summary cache

## Commands

No tests, no formatter, no typecheck. Installed by dropping the directory into CAT's
`plugins/` folder.
