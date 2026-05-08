import threading
import time
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
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
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for cache entries in seconds."
    )
    cache_max_size: int = Field(
        default=100,
        description="Maximum number of entries in the LRU cache."
    )


@plugin
def settings_model():
    return CatAlogSettings


class TTL_LRU_Cache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict[Tuple[str, Optional[str], str], Tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

    def _make_key(self, agent_id: str, chat_id: Optional[str], filename: str) -> Tuple[str, Optional[str], str]:
        return (agent_id, chat_id, filename)

    def get(self, agent_id: str, chat_id: Optional[str], filename: str) -> Optional[str]:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value
        return None

    def set(self, agent_id: str, chat_id: Optional[str], filename: str, summary: str) -> None:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (summary, time.time())
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def delete(self, agent_id: str, chat_id: Optional[str], filename: str) -> bool:
        key = self._make_key(agent_id, chat_id, filename)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False


CATALOGUES = TTL_LRU_Cache()

@hook(priority=10)
async def before_rabbithole_splits_documents(docs: List[Document], cat) -> List[Document]:
    if not docs:
        return docs

    metadata = docs[0].metadata
    if 'source' not in metadata:
        return docs
    source = metadata['source']
    agent_id = cat.agent_key
    chat_id = metadata.get('chat_id', None)

    cached_summary = CATALOGUES.get(agent_id, chat_id, source)
    if cached_summary is not None:
        log.info(f"cat_alog: cache hit for '{agent_id}/{chat_id}/{source}'")
        return docs

    settings = await cat.mad_hatter.get_plugin().load_settings()
    settings = settings or {}
    max_document_chars = int(settings.get("max_document_chars", 8000))
    max_summary_words = int(settings.get("max_summary_words", 200))

    full_text = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content and doc.page_content.strip()
    )
    if len(full_text) > max_document_chars:
        full_text = full_text[:max_document_chars] + ' [truncated] '

    safe_text = full_text.replace('{', '{{').replace('}', '}}')

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
        log.warning(f"cat_alog: failed to summarize '{agent_id}/{chat_id}/{source}': {e}")
        summary = "(summary not available)"

    CATALOGUES.set(agent_id, chat_id, source, summary)

    log.info(f"cat_alog: summary added for '{agent_id}/{chat_id}/{source}' [{summary[:100]}...]")
    return docs


@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    """
    Retrieve the pre-computed summary from the cat and append the catalogue card.
    """
    if not docs:
        return docs

    metadata = docs[0].metadata
    if 'source' not in metadata:
        return docs

    source = metadata['source']
    agent_id = cat.agent_key
    chat_id = metadata.get('chat_id', None)

    summary = CATALOGUES.get(agent_id, chat_id, source)
    abstract = docs[0].page_content.strip()

    if not summary:
        log.warning(f"cat_alog: no summary found for {agent_id}/{chat_id}/{source}, skipping catalogue card.")
        return docs

    CATALOGUES.delete(agent_id, chat_id, source)

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
    return docs + [Document(page_content=card, metadata=card_metadata)]


@tool(examples=[
    "What files are present",
    "List the files",
    "Tell me what files you know",
])
def list_loaded_files(cat):
    """List all files loaded in the current conversation and agent with their details.

    Returns a markdown formatted list showing files loaded in the current chat session
    and at the agent level, with their source, upload date, and content preview.
    """
    agent_id = cat.agent_key
    chat_id = getattr(cat, "id", None)

    output = []
    output.append("# Loaded Files\n")

    async def _get_files():
        files_dict: Dict[str, Dict] = {}

        async def add_points_from_collection(collection_name: str, is_episodic: bool):
            try:
                points = await cat.vector_memory_handler.recall_tenant_memory(collection_name)
                for point in points:
                    doc = point.document
                    metadata = doc.metadata or {}
                    source = metadata.get("source", "unknown")
                    point_chat_id = metadata.get("chat_id")

                    if is_episodic and point_chat_id != chat_id:
                        continue
                    if not source or source.startswith("http"):
                        continue

                    if source not in files_dict:
                        when_ts = metadata.get("when", 0)
                        when_str = datetime.fromtimestamp(when_ts).strftime("%Y-%m-%d %H:%M") if when_ts else "unknown"
                        files_dict[source] = {
                            "source": source,
                            "when": when_str,
                            "content": doc.page_content[:500] if doc.page_content else "",
                            "type": "conversation" if is_episodic else "agent",
                        }
            except Exception as e:
                log.warning(f"cat_alog: error reading collection {collection_name}: {e}")

        await add_points_from_collection(str(VectorMemoryType.EPISODIC), is_episodic=True)
        await add_points_from_collection(str(VectorMemoryType.DECLARATIVE), is_episodic=False)

        return files_dict

    import asyncio
    files_dict = asyncio.run(_get_files())

    if not files_dict:
        output.append("No files loaded in this conversation or agent.")
        return "\n".join(output)

    for source, info in sorted(files_dict.items()):
        file_type = "Chat" if info["type"] == "conversation" else "Agent"
        content = info["content"]
        when = info["when"]

        output.append(f"## {source}\n")
        output.append(f"- **Type**: {file_type}")
        output.append(f"- **Uploaded**: {when}\n")
        output.append("**Content preview:**\n")
        output.append(f"```\n{content}...\n```\n")

    return "\n".join(output)
