from typing import List
from cat import hook, log
from langchain_core.documents.base import Document

@hook(priority=10)
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    '''Add a catalogue card describing the summarized documents'''
    LLM = cat
    # - extract metadata from documents
    metadata = docs[0].metadata
    # - rebuild file for summarization? Is it needed or I could just use the docs?
    texts = [doc.page_content for doc in docs]
    # - ask the LLM for summarization
    prompt = """Write a short summary (maximum 300 words) of the following document.
    """
    summary = LLM.generate_prompt([prompt] + texts)
    # - build catalogue card
    source = metadata.get("source", "unknown")
    title  = metadata.get("title", "unknown")
    # add other metadata? add all metadata? keep some metadata private?
    card = f'''Source (URL or filename): {source}
Title: {title}
Summary: 
{summary}
'''
    log.debug(f"Adding catalogue card to file \n{card}")
    # - add a new Document with same metadata and this catalogue card
    return docs + [Document(page_content=card, metadata=metadata)]
