"""
Prompt templates for the LangGraph flashcard generation pipeline.

All LLM prompt templates are centralized here to keep node logic
focused on orchestration and state management within the agentic
workflow.
"""

from langchain_core.prompts import ChatPromptTemplate

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert educator who creates high-quality Anki flashcards. "
        "Given a text chunk, produce between 3 and 5 question-answer flashcards "
        "that capture the most important concepts.\n\n"
        "Quality criteria (apply these yourself, do NOT produce low-quality cards):\n"
        "- Each question must be specific, unambiguous, and self-contained\n"
        "- Each answer must be concise but complete\n"
        "- Do NOT produce vague, trivially obvious, or duplicate cards\n"
        "- Do NOT produce cards that require the original text to answer\n"
        "- If the chunk contains no meaningful content, return an empty list\n\n"
        "Return ONLY a JSON object with a single key \"flashcards\" containing "
        "a list of objects, each with \"question\" and \"answer\" keys.\n\n"
        "Example output:\n"
        "{{\n"
        '  "flashcards": [\n'
        '    {{"question": "What is X?", "answer": "X is ..."}},\n'
        '    {{"question": "How does Y work?", "answer": "Y works by ..."}}\n'
        "  ]\n"
        "}}",
    ),
    (
        "user",
        "Text chunk:\n"
        '"""\n'
        "{chunk}\n"
        '"""',
    ),
])
