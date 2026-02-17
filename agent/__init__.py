"""
Flashcard generation agent built with LangGraph.

This package implements an agentic workflow for generating Anki flashcards
from PDF documents.  The pipeline uses a 5-node state graph with a
conditional quality gate that can loop back to regenerate cards if the
review node determines the quality is insufficient.
"""

from agent.graph import build_graph
from agent.state import FlashcardState

__all__ = ["build_graph", "FlashcardState"]
