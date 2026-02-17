"""
Node 2 -- Chunk Text.

Splits loaded documents into overlapping text chunks using
``RecursiveCharacterTextSplitter`` for downstream LLM processing.
Chunk size is set to 2000 characters to minimize the number of LLM
calls while staying well within the model's context window.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.state import FlashcardState

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def chunk_text(state: FlashcardState) -> dict:
    """Split documents into overlapping text chunks.

    Parameters
    ----------
    state : FlashcardState
        Must contain ``documents`` (list of LangChain Documents).

    Returns
    -------
    dict
        Keys: ``chunks``, ``chunk_count``, ``log_messages``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(state["documents"])
    chunk_count = len(chunks)

    return {
        "chunks": chunks,
        "chunk_count": chunk_count,
        "log_messages": [
            f"Split into {chunk_count} chunk(s) "
            f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
        ],
    }
