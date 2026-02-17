"""
State schema for the flashcard generation graph.

Defines the ``FlashcardState`` TypedDict consumed by every node in the
LangGraph pipeline.  The ``log_messages`` field uses an append-only
reducer so that each node can push status updates without overwriting
previous entries.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class FlashcardState(TypedDict, total=False):
    """Shared state flowing through the flashcard generation graph.

    Attributes
    ----------
    file_path : str
        Path to the source PDF file.
    output_dir : str
        Directory where exported files are written.
    documents : list[Document]
        Raw pages returned by the document loader.
    page_count : int
        Number of pages loaded from the PDF.
    chunks : list[Document]
        Text chunks produced by the splitter.
    chunk_count : int
        Number of chunks after splitting.
    flashcards : list[dict[str, str]]
        Generated question/answer pairs before review.
    reviewed_flashcards : list[dict[str, str]]
        Flashcards that passed the quality review.
    cards_kept : int
        Number of cards approved during review.
    cards_removed : int
        Number of cards rejected during review.
    quality_approved : bool
        Whether the reviewer approved the current batch.
    retry_count : int
        Number of regeneration attempts (caps the review loop).
    csv_path : str
        Path to the exported CSV file.
    jsonl_path : str
        Path to the exported JSONL file.
    log_messages : Annotated[list[str], operator.add]
        Append-only log of status messages from each node.
    """

    file_path: str
    output_dir: str

    documents: list[Document]
    page_count: int

    chunks: list[Document]
    chunk_count: int

    flashcards: list[dict[str, str]]

    reviewed_flashcards: list[dict[str, str]]
    cards_kept: int
    cards_removed: int
    quality_approved: bool
    retry_count: int

    csv_path: str
    jsonl_path: str

    log_messages: Annotated[list[str], operator.add]
