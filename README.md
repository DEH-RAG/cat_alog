# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

The catalog card contains:
- the file metadata
- the first chunk of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

## Catalog cards as an ingestion phase

When the [EffING](https://github.com/DEH-RAG/cat-efficient-ingestion) phase machine is deployed, card generation runs as the **`catalog_cards`** ingestion phase instead of the legacy in-memory hooks. The phase is declared through `ingestion_phase_specs` with `settings_category=None` and `depends_on=("parsing_chunking", "embedding")`, so it re-runs automatically when the source's chunks or vectors change. `ingestion_phase_settings_marker` returns a SHA-256 fingerprint of the effective card config (max document chars, max summary words, cache TTL and size) plus the summary prompt template, so any material change invalidates previously generated cards. `ingestion_phase_run` builds the card and stores it as a vector point with `is_catalogue_card: True` and the full card in metadata, using a deterministic point id so re-runs upsert instead of duplicating.

With the machine active, the in-memory TTL cache is no longer the persistence: the two `before_rabbithole_*` hooks become no-ops and the durable phase diary (written by EffING) is what makes cards restartable. The `list_loaded_files` tool is unchanged — it reads the stored cards (and falls back to first chunks) the same way regardless of which path generated them.
