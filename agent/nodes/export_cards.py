"""
Node 5 -- Export Cards.

Writes the reviewed flashcards to both CSV (semicolon-delimited,
UTF-8 with BOM for Excel compatibility) and JSONL formats.
"""

import csv
import json
from pathlib import Path

from agent.state import FlashcardState


def export_cards(state: FlashcardState) -> dict:
    """Export reviewed flashcards to CSV and JSONL files.

    Parameters
    ----------
    state : FlashcardState
        Must contain ``reviewed_flashcards`` and ``output_dir``.

    Returns
    -------
    dict
        Keys: ``csv_path``, ``jsonl_path``, ``log_messages``.
    """
    output_dir = Path(state.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = state["reviewed_flashcards"]

    csv_path = output_dir / "flashcards.csv"
    jsonl_path = output_dir / "flashcards.jsonl"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        for card in cards:
            writer.writerow([card["question"], card["answer"]])

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for card in cards:
            json.dump(card, fh, ensure_ascii=False)
            fh.write("\n")

    return {
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
        "log_messages": [
            f"Exported {len(cards)} flashcard(s) to {csv_path} and {jsonl_path}."
        ],
    }
