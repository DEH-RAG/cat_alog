from typing import List
from pydantic import BaseModel, Field
from cat import hook, plugin, log, AgenticWorkflowTask
from langchain_core.documents.base import Document


class CatAlogSettings(BaseModel):
    max_summary_chars: int = Field(
        default=8000,
        description="Maximum number of characters sent to the LLM for file summarization."
    )


@plugin
def settings_model():
    return CatAlogSettings


@hook(priority=10)
async def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    if not docs:
        return docs

    settings = await cat.mad_hatter.get_plugin().load_settings()
    settings = settings or {}
    max_summary_chars = int(settings.get("max_summary_chars", 8000))

    full_text = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()
    )

    source = docs[0].metadata.get("source", "unknown")

    full_prompt = f"""Write a short summary of the following file.
Focus on what the file is, what it is about, and what it contains.
Maximum {max_summary_chars} words.

Filename or source: {source}

{full_text}
"""

    try:
        agent_input = AgenticWorkflowTask(user_prompt=full_prompt)

        callbacks = await cat.plugin_manager.execute_hook("llm_callbacks", [], caller=cat)
        summary_agent_output = await cat.agentic_workflow.run(
            task=agent_input,
            llm=cat.large_language_model,
            callbacks=callbacks,
        )
        summary = summary_agent_output.output
    except Exception as e:
        log.warning(f"cat_alog: failed to summarize '{source}': {e}")
        summary = "(summary not available)"

    for doc in docs:
        doc.metadata["_catalogue_summary"] = summary

    log.debug(f"cat_alog: summary stored in metadata for '{source}'")
    return docs


@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """
    Retrieve the pre-computed summary from metadata and append the catalogue card.
    This hook is sync because no LLM call is needed here.
    """
    if not docs:
        return docs

    # The summary was stashed by the previous hook on the first chunk.
    # All chunks share the same source so any of them works.
    summary = docs[0].metadata.get("_catalogue_summary", None)

    if not summary:
        log.warning("cat_alog: no summary found in metadata, skipping catalogue card.")
        return docs

    metadata = docs[0].metadata.copy()
    source = metadata.get("source", "unknown")

    # Clean up the temporary key from all chunks before storing.
    for doc in docs:
        doc.metadata.pop("_catalogue_summary", None)

    card_metadata = {
        **metadata,
        "is_catalogue_card": True,
        "catalogue_for_source": source,
    }
    # Remove the temp key from card metadata too, in case it was copied above.
    card_metadata.pop("_catalogue_summary", None)

    card = (
        f"CATALOGUE CARD\n"
        f"Source: {source}\n"
        f"Summary:\n{summary}\n"
    )

    log.debug(f"cat_alog: appending catalogue card for '{source}'")
    return docs + [Document(page_content=card, metadata=card_metadata)]
