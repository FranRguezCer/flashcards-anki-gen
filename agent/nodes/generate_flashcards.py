"""
Node 3 -- Generate Flashcards.

Uses LCEL ``(prompt | llm).batch()`` to process all text chunks in
parallel, producing question/answer pairs as part of the LangGraph
agentic workflow. Parallel dispatch significantly reduces end-to-end
latency compared to sequential invocation.
"""

import json

from agent.llm import build_llm, clean_json
from agent.prompts import GENERATION_PROMPT
from agent.state import FlashcardState

MAX_CONCURRENCY = 2


def generate_flashcards(state: FlashcardState) -> dict:
    """Generate flashcards from all chunks in parallel via chain.batch().

    Parameters
    ----------
    state : FlashcardState
        Must contain ``chunks`` (list of LangChain Documents).

    Returns
    -------
    dict
        Keys: ``flashcards``, ``log_messages``.
    """
    llm = build_llm()
    chain = GENERATION_PROMPT | llm
    chunks = state["chunks"]
    total = len(chunks)

    print(f"  [generate] Dispatching {total} chunk(s) "
          f"(max_concurrency={MAX_CONCURRENCY}) ...", flush=True)

    inputs = [{"chunk": chunk.page_content} for chunk in chunks]
    results = chain.batch(inputs, config={"max_concurrency": MAX_CONCURRENCY})

    flashcards: list[dict[str, str]] = []
    errors = 0

    for result in results:
        try:
            parsed = json.loads(clean_json(result.content))
            if isinstance(parsed, list):
                cards = parsed
            elif isinstance(parsed, dict) and "flashcards" in parsed:
                cards = parsed["flashcards"]
            else:
                cards = []
                errors += 1
        except (json.JSONDecodeError, TypeError):
            cards = []
            errors += 1
        flashcards.extend(cards)

    print(f"  [generate] Done: {len(flashcards)} card(s) from {total} chunk(s).",
          flush=True)

    log = f"Generated {len(flashcards)} flashcard(s) from {total} chunk(s)."
    if errors:
        log += f" ({errors} chunk(s) failed to parse.)"

    return {
        "flashcards": flashcards,
        "log_messages": [log],
    }
