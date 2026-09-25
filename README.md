# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

The catalog card contains:
- the file metadata
- the first chunk of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.

## Loaded files in the prompt

The plugin appends a compact list of the files loaded in the agent and conversation to the END of the system prompt via the `agent_prompt_suffix` hook. The list shows each file's name, type (agent or conversation) and upload date, with no content previews. It is a no-op when nothing is loaded. The `list_loaded_files` tool remains the on-demand source of full content previews.
