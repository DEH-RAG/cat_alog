# Cat_alog

Normally the cat would answer about files only when asked about their *content*.

This plugin adds a "catalog card" to the vector database to be able to discuss *about* the files loaded.

The catalog card contains:
- the file metadata
- the first chunk of the document (to capture title and authors)
- a brief summary of the file content, automatically summarized with the currently configured LLM.
