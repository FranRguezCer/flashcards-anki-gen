"""
LangGraph state graph for the flashcard generation pipeline.

Assembles a 5-node linear graph with a conditional quality gate
that can loop back from ``review_quality`` to ``generate_flashcards``
at most once, making this an agentic workflow rather than a simple
sequential chain.
"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.state import FlashcardState
from agent.nodes import (
    parse_document,
    chunk_text,
    generate_flashcards,
    review_quality,
    export_cards,
)

MAX_RETRIES = 1


def quality_gate(state: FlashcardState) -> Literal["generate_flashcards", "export_cards"]:
    """Route after review: regenerate if rejected and retries remain, else export.

    Parameters
    ----------
    state : FlashcardState
        Must contain ``quality_approved`` and ``retry_count``.

    Returns
    -------
    str
        Next node name: ``"generate_flashcards"`` or ``"export_cards"``.
    """
    if not state.get("quality_approved", True) and state.get("retry_count", 0) <= MAX_RETRIES:
        return "generate_flashcards"
    return "export_cards"


def build_graph() -> CompiledStateGraph:
    """Build and compile the flashcard generation state graph.

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph ready to ``.invoke()`` or ``.stream()``.
    """
    builder = StateGraph(FlashcardState)

    builder.add_node("parse_document", parse_document)
    builder.add_node("chunk_text", chunk_text)
    builder.add_node("generate_flashcards", generate_flashcards)
    builder.add_node("review_quality", review_quality)
    builder.add_node("export_cards", export_cards)

    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "chunk_text")
    builder.add_edge("chunk_text", "generate_flashcards")
    builder.add_edge("generate_flashcards", "review_quality")
    builder.add_conditional_edges("review_quality", quality_gate)
    builder.add_edge("export_cards", END)

    return builder.compile()
