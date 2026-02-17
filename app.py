"""
Gradio web interface for the Anki flashcard generator.

Provides a simple UI for uploading PDF files, generating flashcards
via the LangGraph agentic workflow, and downloading the results as
CSV or JSONL. Executes nodes manually (rather than via graph.stream)
to yield per-chunk progress during generation.
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv

import gradio as gr

from agent.llm import build_llm, clean_json
from agent.prompts import GENERATION_PROMPT
from agent.nodes import parse_document, chunk_text, review_quality, export_cards

MAX_RETRIES = 1

CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}
.prose h1, .prose p {
    text-align: center !important;
}
"""

DARK_JS = """
function() {
    document.body.classList.add('dark');
}
"""


def process_document(file: str | None, progress=gr.Progress()):
    """Run the flashcard generation pipeline with per-chunk progress.

    Executes graph nodes manually so the UI can yield after every
    chunk during flashcard generation, giving real-time feedback.

    Parameters
    ----------
    file : str or None
        File path string from the Gradio file component, or None.
    progress : gr.Progress
        Gradio progress tracker for the progress bar.

    Yields
    ------
    tuple
        ``(log_text, dataframe, csv_path, jsonl_path)`` updated after
        each processing step.
    """
    if file is None:
        yield "No file uploaded.", pd.DataFrame(), None, None
        return

    file_path = file.name if hasattr(file, "name") else str(file)
    output_dir = os.environ.get("OUTPUT_DIR", "./output")

    log_lines: list[str] = []
    dataframe = pd.DataFrame()
    csv_path = None
    jsonl_path = None

    def _yield():
        return "\n".join(log_lines), dataframe, csv_path, jsonl_path

    # -- Node 1: parse_document --
    progress(0.05, desc="Loading PDF...")
    log_lines.append("Loading PDF...")
    yield _yield()

    state = {"file_path": file_path, "output_dir": output_dir}
    result = parse_document(state)
    state.update(result)
    log_lines.extend(result["log_messages"])
    yield _yield()

    # -- Node 2: chunk_text --
    progress(0.10, desc="Splitting text into chunks...")
    result = chunk_text(state)
    state.update(result)
    log_lines.extend(result["log_messages"])
    yield _yield()

    # -- Node 3: generate_flashcards (manual, per-chunk) --
    llm = build_llm()
    chain = GENERATION_PROMPT | llm
    chunks = state["chunks"]
    total = len(chunks)
    retry_count = 0

    while True:
        flashcards: list[dict[str, str]] = []
        errors = 0
        attempt = retry_count + 1

        if retry_count > 0:
            log_lines.append(f"Regenerating flashcards (attempt {attempt})...")
            yield _yield()

        for i, chunk in enumerate(chunks):
            pct = 0.15 + 0.65 * (i / total)
            progress(pct, desc=f"Generating chunk {i + 1}/{total} "
                              f"(attempt {attempt})...")
            log_lines.append(f"  Chunk {i + 1}/{total}: processing...")
            yield _yield()

            raw = chain.invoke({"chunk": chunk.page_content})
            try:
                parsed = json.loads(clean_json(raw.content))
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
            log_lines[-1] = (f"  Chunk {i + 1}/{total}: "
                             f"+{len(cards)} cards (total: {len(flashcards)})")
            yield _yield()

        gen_log = (f"Generated {len(flashcards)} flashcard(s) "
                   f"from {total} chunk(s).")
        if errors:
            gen_log += f" ({errors} chunk(s) failed to parse.)"
        log_lines.append(gen_log)
        yield _yield()

        state["flashcards"] = flashcards

        # -- Node 4: review_quality --
        progress(0.85, desc="Reviewing quality...")
        result = review_quality(state)
        state.update(result)
        log_lines.extend(result["log_messages"])
        retry_count = result["retry_count"]

        if "reviewed_flashcards" in result:
            dataframe = pd.DataFrame(
                result["reviewed_flashcards"],
                columns=["question", "answer"],
            )
        yield _yield()

        # Quality gate
        if state.get("quality_approved", True) or retry_count > MAX_RETRIES:
            break

        log_lines.append("Quality gate: batch rejected, retrying...")
        yield _yield()

    # -- Node 5: export_cards --
    progress(0.95, desc="Exporting flashcards...")
    result = export_cards(state)
    state.update(result)
    log_lines.extend(result["log_messages"])
    csv_path = result["csv_path"]
    jsonl_path = result["jsonl_path"]

    progress(1.0, desc="Done!")
    yield _yield()


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns
    -------
    gr.Blocks
        Configured Gradio application ready to launch.
    """
    load_dotenv()

    with gr.Blocks(title="Anki Flashcard Generator") as app:
        gr.Markdown("# Anki Flashcard Generator\nUpload a PDF and generate "
                     "high-quality flashcards using a LangGraph agentic workflow.")

        with gr.Row():
            file_input = gr.File(
                file_types=[".pdf"],
                file_count="single",
                label="Upload PDF",
            )

        generate_btn = gr.Button("Generate Flashcards", variant="primary")

        progress_log = gr.Textbox(
            label="Progress Log",
            lines=12,
            interactive=False,
            placeholder="Upload a PDF and click Generate to start...",
        )

        results_table = gr.Dataframe(
            headers=["question", "answer"],
            label="Generated Flashcards",
            interactive=False,
        )

        with gr.Row():
            csv_download = gr.File(label="Download CSV", interactive=False)
            jsonl_download = gr.File(label="Download JSONL", interactive=False)

        gr.on(
            triggers=[generate_btn.click],
            fn=process_document,
            inputs=[file_input],
            outputs=[progress_log, results_table, csv_download, jsonl_download],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(
            primary_hue="blue",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_lg,
        ),
        css=CUSTOM_CSS,
        js=DARK_JS,
    )
