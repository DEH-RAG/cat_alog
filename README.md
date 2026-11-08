# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

## Catalog cards

A catalog card is stored in the vector database so the cat can discuss *about* the files loaded, not only their content. It contains:
- the file metadata
- the initial page of the document (its first text chunk, to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

The card is built by the `before_rabbithole_stores_documents` hook, which the RabbitHole fires once per parsing of a source (its `parsing_chunking` phase): an upload, a change of the chunker, a re-ingestion. The summary is computed there from the source's text chunks, capped at `max_document_chars`; when the LLM fails the card keeps `(summary not available)`. Only text reaches the LLM and the card: image chunks, previous cards and binary content decoded as text are left out, and a source without any readable text gets no card.

The card is kept by the RabbitHole with the other chunks of the source, so:

- a change of the **embedder** embeds the card again, without calling the LLM (a source ingested before the ingestion states existed is parsed again at that point, so it gets its card then);
- every new parsing of the source replaces the card of the previous one, together with the other points of the source;
- deleting the source deletes its card.

A change of the LLM or of the settings of the plugin does not rebuild the existing cards, and a failed summary is not retried: a new upload of the file, or a change of the chunker, does. Likewise, activating the plugin gives no card to the sources already loaded, and deactivating it leaves the existing cards in memory, until those sources are parsed again.

The hook also runs when a caller stores its documents directly (`RabbitHole.store_documents`): the card is stored with them.

Every point of the card carries the whole card in `full_card` (useful when the card is split into segments) and `is_catalogue_card: True`. `chunk_index` is set on the chunks that have none (their position), so the first chunk of a source can be found; Cat_alog reads it to find the first chunk.

## Loaded files in the prompt

With `append_file_list_to_prompt` enabled, `agent_prompt_suffix` appends a compact list of the files loaded in the agent and in the conversation (names, type, upload date; no previews) to the END of the system prompt. The list comes from the FileManager plus the vector memory (for the URLs); it is a no-op when nothing is loaded. Images extracted from files are never listed: they are derived artifacts, not sources (`image is True` and `image_file != source`). The `list_loaded_files` tool is the on-demand source of the content previews: the summary of the card, or the first chunk of a source without a card.

## Settings

| Key | Default | Description |
|-----|---------|-------------|
| `append_file_list_to_prompt` | `True` | Append the list of files loaded in the agent and conversation to the end of the system prompt. When `False` the prompt is returned untouched and no file collection (no DB/vector query) is performed. |
| `max_document_chars` | `8000` | Max characters sent to the LLM for file summarization |
| `max_summary_words` | `200` | Max words for the summary |

## Commands

- Run the tests, from the root of grinning-cat-core: `.venv/bin/pytest cat/plugins/cat_alog/tests -q -p no:cacheprovider`
