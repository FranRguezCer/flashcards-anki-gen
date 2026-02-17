"""
Node 1 -- Parse Document.

Loads a PDF file using ``PyPDFLoader`` and writes the raw
``Document`` objects into the graph state.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from agent.state import FlashcardState


def parse_document(state: FlashcardState) -> dict:
    """Load a PDF file and return its pages as LangChain Documents.

    Parameters
    ----------
    state : FlashcardState
        Must contain ``file_path`` pointing to a valid PDF.

    Returns
    -------
    dict
        Keys: ``documents``, ``page_count``, ``log_messages``.

    Raises
    ------
    FileNotFoundError
        If the file at ``file_path`` does not exist.
    """
    path = Path(state["file_path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    page_count = len(documents)

    return {
        "documents": documents,
        "page_count": page_count,
        "log_messages": [
            f"Loaded {page_count} page(s) from {path.name}."
        ],
    }
