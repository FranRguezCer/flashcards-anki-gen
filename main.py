"""
main.py ─ CLI entry-point

Usage examples:
    python main.py docs/example.pdf -o example.csv                  # CSV (default)
    python main.py docs/example.pdf -o example.jsonl --format jsonl # JSONL
"""

import argparse
from flashcards.loader_splitter import load_and_split
from flashcards.flashcard_chain import generate_flashcards
from flashcards.exporter import export_flashcards


def run(file_path: str, output: str, fmt: str = "csv") -> None:
    """Full pipeline: load → split → generate → export."""
    chunks = load_and_split(file_path)
    flashcards = generate_flashcards(chunks)
    export_flashcards(flashcards, output_path=output, fmt=fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from a document."
    )
    parser.add_argument("file", help="Path to PDF/DOCX/TXT")
    parser.add_argument(
        "-o",
        "--output",
        default="flashcards.csv",
        help="Output file name (with extension)",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "jsonl"],
        help="Output format",
    )

    args = parser.parse_args()
    run(args.file, args.output, args.format)
