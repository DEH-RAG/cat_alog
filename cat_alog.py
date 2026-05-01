from typing import List
from pydantic import BaseModel, Field
from cat import hook, plugin, log, AgenticWorkflowTask
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


@plugin
def settings_model():
    return CatAlogSettings

CATALOGUES = {}

@hook(priority=10)
async def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    if not docs:
        return docs

    metadata = docs[0].metadata
    if 'source' not in metadata: # XLSX files show no source (WHY?)
        return docs
    source = metadata['source']
    agent  = cat.agent_key
    # TODO: use the storage path of the file as key

    settings = await cat.mad_hatter.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words  = int(settings.get("max_summary_words",   200))

    full_text = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()
    )
    if len(full_text) > max_document_chars:
        full_text = full_text[:max_document_chars] + ' [truncated] '

    safe_text = full_text.replace('{','{{').replace('}','}}')

    full_prompt = f"""Write a short summary of the following file.
Focus on what the file is, what it is about, and what it contains.
Maximum {max_summary_words} words.

## Filename or source: {source}

## File content

{safe_text}
"""

    try:
        agent_input = AgenticWorkflowTask(user_prompt=full_prompt)
        summary_agent_output = await cat.agentic_workflow.run(
            task=agent_input,
            llm=cat.large_language_model,
        )
        summary = summary_agent_output.output
    except Exception as e:
        log.warning(f"cat_alog: failed to summarize '{agent}/{source}': {e}")
        summary = "(summary not available)"

    if agent not in CATALOGUES:
        CATALOGUES[agent] = {}
    CATALOGUES[agent][source] = summary

    log.info(f"cat_alog: summary added for '{agent}/{source}' [{summary[:100]}...]")
    return docs


@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """
    Retrieve the pre-computed summary from the cat and append the catalogue card.
    """
    if not docs:
        return docs

    metadata  = docs[0].metadata
    if 'source' not in metadata: # XLSX files show no source (WHY?)
        return docs

    # Card metadata
    source    = metadata['source']
    agent     = cat.agent_key
    # TODO: use the storage path of the file as key
    summary   = CATALOGUES.get(agent, {}).get(source,None)
    abstract  = docs[0].page_content.strip()

    if not summary:
        log.warning(f"cat_alog: no summary found for {agent}/{source}, skipping catalogue card.")
        return docs

    del CATALOGUES[agent][source]
    
    card_metadata = {
        **metadata,
        "is_catalogue_card": True,
    }

    card = (
        "\n# CATALOGUE CARD\n\n"
        f"## Source file or URL: {source}\n\n"
        "## Initial page\n\n"
        + abstract + "\n\n"
        "## Summary\n\n"
        + summary + "\n"
    )

    log.debug(f"cat_alog: added catalogue card for '{source}'")
    return  docs + [Document(page_content=card, metadata=card_metadata)]
