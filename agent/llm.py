"""
Shared LLM utilities for the flashcard generation pipeline.

Provides the ``ChatOllama`` factory and JSON-cleaning helpers used
by the LangGraph node functions that require LLM interaction.
"""

import os
import re

from langchain_ollama import ChatOllama

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.MULTILINE)


def clean_json(raw: str) -> str:
    """Strip triple-backtick fences and surrounding whitespace.

    Parameters
    ----------
    raw : str
        Raw LLM output that may contain Markdown code fences.

    Returns
    -------
    str
        Cleaned string ready for ``json.loads()``.
    """
    return CODE_FENCE_RE.sub("", raw).strip()


def build_llm() -> ChatOllama:
    """Construct a ``ChatOllama`` instance from environment variables.

    Returns
    -------
    ChatOllama
        Configured chat model using ``OLLAMA_MODEL`` and ``OLLAMA_HOST``.
    """
    return ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "mistral-large-3:675b-cloud"),
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        temperature=0.15,
        format="json",
    )
