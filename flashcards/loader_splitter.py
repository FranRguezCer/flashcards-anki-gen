"""
loader_splitter.py
==================
Utility functions to load a single file (PDF, DOCX, TXT, …)
and split its text into manageable chunks for downstream LLM
processing (flashcard generation).

The module uses the *lightweight* `PyPDFLoader` for PDFs (pure-Python,
no heavy vision/OCR dependencies) and falls back to
`UnstructuredFileLoader` for other formats.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,             # lightweight PDF parser (pypdf under the hood)
    UnstructuredFileLoader,  # generic loader for DOCX / TXT / HTML …
)


def _pick_loader(path: Path):
    """
    Return the appropriate LangChain loader for the given file.

    Parameters
    ----------
    path : pathlib.Path
        Absolute path to the file we want to ingest.

    Returns
    -------
    BaseLoader
        A loader instance ready to `.load()` the document.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        # Pure-Python PDF loader → avoids heavy onnx/tesseract stacks
        return PyPDFLoader(str(path))
    # Anything else (DOCX, TXT, HTML…) → generic Unstructured loader
    return UnstructuredFileLoader(str(path))


def load_and_split(
    file_path: str | Path,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[Document]:
    """
    Load a document and split it into overlapping chunks.

    Parameters
    ----------
    file_path : str | Path
        Relative or absolute path to the input file.
    chunk_size : int, default 400
        Maximum characters per chunk (≈ 300–350 tokens for GPT-3.5).
    overlap : int, default 50
        Characters of overlap between consecutive chunks to keep context.

    Returns
    -------
    list[Document]
        A list of LangChain `Document` objects, each containing:
         `.page_content` – the text chunk
         `.metadata`     – source information (ignored downstream)
    """
    # Normalize & validate the path early
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # 1  Load the raw document via the appropriate loader
    loader = _pick_loader(path)
    docs = loader.load()  # often returns a list of length 1

    # 2  Recursively split into chunks that fit the model context window
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(docs)

    return chunks
