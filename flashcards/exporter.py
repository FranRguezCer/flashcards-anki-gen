import csv
from pathlib import Path
import json
from typing import List, Dict

# ───────────────────────────── JSONL EXPORTER ──────────────────────────────
def export_to_jsonl(
    flashcards: List[Dict[str, str]],
    output_path: str = "flashcards.jsonl",
) -> None:
    """
    Save flashcards to JSON Lines (one JSON object per line).
    This is handy for ML pipelines or programmatic ingestion.
    """
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as fh:
        for card in flashcards:
            json.dump(card, fh, ensure_ascii=False)
            fh.write("\n")
    print(f"✅  Saved {len(flashcards)} flashcards → {out.resolve()}")


# ───────────────────────────── CSV EXPORTER ─────────────────────────────────
def export_to_csv(
    flashcards: List[Dict[str, str]],
    output_path: str = "flashcards.csv",
    delimiter: str = ";",       
):
    """
    Save flashcards to CSV (UTF-8 with BOM) so Excel opens it flawlessly.
    Columns: Front | Back |
    """
    out = Path(output_path)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        for card in flashcards:
            writer.writerow([card["question"], card["answer"]])

    print(f"✅  Saved {len(flashcards)} flashcards → {out.resolve()}")


# ────────────────────────── EXPORTER DISPATCHER ──────────────────────────────
def export_flashcards(
    flashcards: List[Dict[str, str]],
    output_path: str,
    fmt: str = "csv",
    **kwargs,
) -> None:
    """
    Send the flashcards to the right exporter based on `fmt`.
    """
    fmt = fmt.lower()
    if fmt == "csv":
        export_to_csv(flashcards, output_path, **kwargs)
    elif fmt == "jsonl":
        export_to_jsonl(flashcards, output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")