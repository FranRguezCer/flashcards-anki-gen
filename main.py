"""
Entry point for the Anki flashcard generator (LangGraph agentic workflow).

Dispatches to the LangGraph agentic pipeline in CLI mode, or launches
the Gradio web interface in UI mode.  Supports two modes:

- ``cli``: Process a PDF file from the command line and export flashcards.
- ``ui``:  Launch the Gradio web interface.

Usage
-----
CLI mode::

    python main.py --mode cli --file docs/ru_vocabulary.pdf --output ./output

UI mode::

    python main.py --mode ui
"""

import argparse
import os

from dotenv import load_dotenv


def run_cli(file_path: str, output_dir: str) -> None:
    """Run the flashcard generation pipeline from the command line.

    Parameters
    ----------
    file_path : str
        Path to the input PDF file.
    output_dir : str
        Directory for exported CSV and JSONL files.
    """
    from agent.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "file_path": file_path,
        "output_dir": output_dir,
    })

    for msg in result.get("log_messages", []):
        print(msg)

    print(f"\nCSV:   {result.get('csv_path', 'N/A')}")
    print(f"JSONL: {result.get('jsonl_path', 'N/A')}")
    print(f"Total: {result.get('cards_kept', 0)} flashcard(s)")


def run_ui() -> None:
    """Launch the Gradio web interface."""
    from app import create_app

    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


def main() -> None:
    """Parse arguments and dispatch to the appropriate mode."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Anki Flashcard Generator -- LangGraph Agentic Workflow",
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "ui"],
        default="ui",
        help="Run mode: 'cli' for command-line or 'ui' for Gradio (default: ui).",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to input PDF file (required for cli mode).",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("OUTPUT_DIR", "./output"),
        help="Output directory for exported files (default: ./output).",
    )

    args = parser.parse_args()

    if args.mode == "cli":
        if not args.file:
            parser.error("--file is required in cli mode.")
        run_cli(args.file, args.output)
    else:
        run_ui()


if __name__ == "__main__":
    main()
