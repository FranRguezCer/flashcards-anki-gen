# 🧠 Anki Flashcard Generator – LLM-powered CLI

This project is a **practical CLI tool** that converts any PDF document into *Anki-style* flashcards using **LangChain** and **OpenAI LLMs**. It handles document loading, smart chunking, robust prompt formatting, and outputs clean CSV/JSONL ready for spaced repetition.

---

## 🧭 Context

Active recall and spaced repetition are proven learning methods. This pipeline turns your own materials into ready-to-import flashcards — ideal for students, language learners, and lifelong learners.

---

## 💡 Why This Isn’t Just an OpenAI Wrapper

This is not a simple wrapper. It:

- 🔀 Splits text into smart overlapping chunks for better context
- 📄 Analyzes full PDF files locally — no size limit like the standard ChatGPT interface
- 📂 Handles multiple PDFs in a directory in one run
- 🗃️ Uses structured prompts for valid JSON output
- 🧹 Cleans and validates raw LLM output automatically
- 💾 Exports in formats suited for Anki, Excel, or ML pipelines
- ⚡ Runs fully automated from your terminal — repeatable and scalable.

---

## 🚀 How It Works

1. **Load & Split**: File is parsed and split into chunks.
2. **Generate**: Each chunk goes through an LLM with a strict Q-A prompt.
3. **Clean & Validate**: Output is stripped of stray code fences.
4. **Export**: Save as CSV (default) or JSONL.

---

## 📄 Example Datasets

- *Transformers Paper*: `docs/attention_is_all_you_need.pdf` → `transformers.csv` / `transformers.jsonl`
- *Russian Vocabulary*: `docs/ru_vocabulary.pdf` → `ru_vocab.csv` / `ru_vocab.jsonl`

Run them:

```bash
python main.py docs/attention_is_all_you_need.pdf -o transformers.csv
python main.py docs/ru_vocabulary.pdf -o ru_vocab.csv
```

---

## 📦 Project Structure

```bash
project-root/
├── assets/                 
│   └── cli-demo.gif                   # GIF demo showing the CLI in action (you can see it below)
├── docs/                        
│   ├── attention_is_all_you_need.pdf  # Example input: Transformers paper
│   └── ru_vocabulary.pdf              # Example input: Russian vocabulary
├── flashcards/                  
│   ├── __init__.py                    # Loads environment variables
│   ├── exporter.py                    # Exports flashcards to CSV/JSONL
│   ├── flashcard_chain.py             # Handles prompt + LLM call via LangChain
│   └── loader_splitter.py             # Loads files & splits into chunks
├── .env.example                       # Example .env file for your OpenAI key
├── LICENSE                            # MIT License
├── main.py                            # CLI entry point script
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── ru_vocab.csv                       # Example output: Russian vocab (CSV)
├── ru_vocab.jsonl                     # Example output: Russian vocab (JSONL)
├── transformers.csv                   # Example output: Transformers paper (CSV)
└── transformers.jsonl                 # Example output: Transformers paper (JSONL)
```

---

## ⚙️ Setup

```bash
git clone <repo-url>
cd project-root
cp .env.example .env  # Add your OPENAI_API_KEY
pip install -r requirements.txt
```

---

## 🖥️ Usage

```bash
# Generate CSV flashcards
python main.py docs/your_notes.pdf -o output.csv

# Or JSONL format
python main.py docs/your_notes.pdf -o output.jsonl --format jsonl
```

---

## 🎥 CLI Demo

<div align="center">
  <img src="assets/cli-demo.gif" alt="CLI Demo" width="600"/>
</div>

---

## 🧰 Tech Stack

- Python, LangChain ≥ 0.2
- OpenAI API (Chat models)
- dotenv, tqdm, argparse

---

## ⏭️ Next Steps

- Direct `.apkg` export for Anki decks
- Multilingual prompt templates
- More formats

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code for personal or commercial purposes, provided that proper credit is given.  
This software is provided **"as is"**, without warranty of any kind.

© 2025 Francisco José Rodríguez Cerezo

---

## 👨‍💻 Author

Created by **Francisco José Rodríguez Cerezo**
[Portfolio](https://franrguezcer.github.io/portfolio/) | [LinkedIn](https://linkedin.com/in/franciscojoserodriguezcerezo)
