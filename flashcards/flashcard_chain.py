"""
flashcard_chain.py
~~~~~~~~~~~~~~~~~~
Generate Q-A flashcards from text *chunks* using LangChain ≥ 0.2
and OpenAI.  The model is asked to return **raw JSON** (no ```json
code fences).  A tiny helper strips any stray fences just in case.
"""

from typing import List, Dict
import json
import re

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document  # only for typing

# ────────────────────────────────────────────────────────────
# 1. Prompt template
#     system  -> global instruction
#     user    -> passes the chunk + output example
# ────────────────────────────────────────────────────────────
FLASHCARD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        # NOTE: “raw JSON” = no Markdown fences, no commentary.
        "You are an expert educator. Produce EXACTLY three Q-A flashcards "
        "in raw JSON list format (keys: question, answer).",
    ),
    (
        "user",
        'Text chunk:\n"""\n{chunk}\n"""\n\n'
        'Return *only* the JSON, e.g.:\n'
        '[{{"question": "…", "answer": "…"}}, {{…}}, {{…}}]',
    ),
])

# ────────────────────────────────────────────────────────────
# 2. Helper: remove ```json … ``` fences if the LLM adds them
# ────────────────────────────────────────────────────────────
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|```$", re.MULTILINE)


def _clean_json(raw: str) -> str:
    """Strip triple-backtick fences and leading/trailing spaces."""
    return CODE_FENCE_RE.sub("", raw).strip()


# ────────────────────────────────────────────────────────────
# 3. Main generator
# ────────────────────────────────────────────────────────────
def generate_flashcards(
    chunks: List[Document],
    model_name: str = "gpt-3.5-turbo", # OPENAI's chat model, you can use  another models like "gpt-4o" or some from anthropic like "claude-2" if you have access
    temperature: float = 0.15, # Parameter that controls the randomness of the model's output (0.0 = deterministic, 1.0 = creative)
) -> List[Dict[str, str]]:
    """
    Parameters
    ----------
    chunks : list[Document]
        Text chunks coming from loader_splitter.load_and_split().
    model_name : str
        OpenAI chat model to use.
    temperature : float
        Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns
    -------
    list[dict]
        [{"question": "...", "answer": "..."}, …]
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    flashcards: List[Dict[str, str]] = []

    for doc in tqdm(chunks, desc="Generating flashcards"):
        # 1) Format the prompt with the current chunk
        messages = FLASHCARD_PROMPT.format_messages(chunk=doc.page_content)

        # 2) Call the model
        raw = llm.invoke(messages).content

        # 3) Clean + parse JSON
        try:
            flashcards.extend(json.loads(_clean_json(raw)))
        except Exception as err:
            # If parsing fails, show 1st 120 chars for quick debugging
            print(f"[WARN] parse error: {err}\nRaw:\n{raw[:120]}…")

    return flashcards
