"""
Node 4 -- Review Quality.

Rule-based quality validation for the LangGraph agentic workflow.
Removes structurally invalid cards (missing/empty fields) and exact
duplicates. Sets ``quality_approved`` based on the pass rate to signal
the conditional quality gate edge -- no LLM call required.
"""

from agent.state import FlashcardState

QUALITY_THRESHOLD = 0.5


def review_quality(state: FlashcardState) -> dict:
    """Rule-based quality review: remove invalid/duplicate cards.

    Parameters
    ----------
    state : FlashcardState
        Must contain ``flashcards`` (list of Q/A dicts).

    Returns
    -------
    dict
        Keys: ``reviewed_flashcards``, ``cards_kept``, ``cards_removed``,
        ``quality_approved``, ``retry_count``, ``log_messages``.
    """
    input_cards = state["flashcards"]
    retry_count = state.get("retry_count", 0)

    seen: set[str] = set()
    reviewed: list[dict[str, str]] = []

    for card in input_cards:
        q = card.get("question", "").strip()
        a = card.get("answer", "").strip()
        if not q or not a:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        reviewed.append({"question": q, "answer": a})

    cards_kept = len(reviewed)
    cards_removed = len(input_cards) - cards_kept
    ratio = cards_kept / max(len(input_cards), 1)
    approved = ratio >= QUALITY_THRESHOLD

    log = (
        f"Review complete (attempt {retry_count + 1}): "
        f"{cards_kept} kept, {cards_removed} removed "
        f"({ratio:.0%} pass rate). Approved: {approved}."
    )
    print(f"  [review] {log}", flush=True)

    return {
        "reviewed_flashcards": reviewed,
        "cards_kept": cards_kept,
        "cards_removed": cards_removed,
        "quality_approved": approved,
        "retry_count": retry_count + 1,
        "log_messages": [log],
    }
