# 🤖 Self-Healing RAG Agent

A Retrieval-Augmented Generation (RAG) agent built with **LangGraph**, **ChromaDB**, and **Groq (LLaMA 3.3 70B)** that automatically detects low-quality answers and retries with a rewritten query healing itself before giving up.

---

## How It Works

The agent follows a self-correcting loop with five nodes orchestrated by LangGraph:

```
START → Retrieve → Generate → Grade
                                 │
                        pass ────┤
                                 │
                        fail → Rewrite → Retrieve (retry)
                                              │
                                     (max 2 retries)
                                              │
                                          Give Up → END
```

1. **Retrieve** — searches ChromaDB for the top-3 most relevant document chunks using cosine similarity.
2. **Generate** — sends the chunks as context to LLaMA 3.3 70B (via Groq) to produce an answer.
3. **Grade** — a second LLM call evaluates whether the answer is fully grounded in the retrieved documents. Passes or fails.
4. **Rewrite** — if the grade fails, the original question is rephrased into a better search query.
5. **Give Up** — after 2 retries, the agent gracefully admits it couldn't find a good answer.

---

## Project Structure

```
self-healing-rag/
|
|
|---- docs/
|      |---- file_1.txt
|      |---- file_2.txt
|      .
|      .
|      |---- file_n.txt
|
|
|---- ingest.py
|---- rag_agent.py
|---- requirements.txt

```

---

## Setup

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/vamshi2011/self-healing-rag.git
cd self-healing-rag
pip install -r requirements.txt
```

### 2. Configure your API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Add your documents

Place any `.txt` files you want the agent to know about inside the `docs/` folder:

```
|---- docs/
|      |---- file_1.txt
|      |---- file_2.txt
|      .
|      .
|      |---- file_n.txt
```

### 4. Ingest the documents

This only needs to be run once (or again whenever you update `docs/`):

```bash
python ingest.py
```

You should see output like:

```
Loaded 6 chunks from python_basics.txt
Loaded 5 chunks from machine_learning.txt
Stored 11 chunks in ChromaDB.
Verification: ChromaDB now contains 11 chunks.
```

### 5. Run the agent

```bash
python rag_agent.py
```

```
🤖 Self-Healing RAG Agent Initialized!
Knowledge base loaded: 11 chunks ready.

Ask a question (or type 'quit'): What is machine learning?
```

---

## Requirements

```
langchain-community
langchain-text-splitters
langchain-groq
langgraph
chromadb
python-dotenv
```

Install all at once:

```bash
pip install langchain-community langchain-text-splitters langchain-groq langgraph chromadb python-dotenv
```

---

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `ingest.py` | `300` | Max characters per document chunk |
| `chunk_overlap` | `ingest.py` | `50` | Overlap between consecutive chunks |
| `n_results` | `rag_agent.py` | `3` | Number of chunks retrieved per query |
| `max retries` | `rag_agent.py` | `2` | How many rewrites before giving up |
| `model` | `rag_agent.py` | `llama-3.3-70b-versatile` | Groq model used for generation and grading |

---

## Troubleshooting

**"Found 0 chunks" on every query**

The knowledge base is empty. Make sure you've run `ingest.py` after placing `.txt` files in `docs/`:

```bash
python ingest.py
```

**Duplicate ID error when re-running `ingest.py`**

This is handled automatically — `ingest.py` clears the existing collection before re-ingesting. Just run it again cleanly.

**`GROQ_API_KEY` not found**

Make sure your `.env` file exists in the project root (same folder as `rag_agent.py`) and contains the correct key. Do not commit this file to version control.

**Agent always gives up without finding an answer**

Your `docs/` folder may not contain information relevant to your question. Add more `.txt` files covering the topic and re-run `ingest.py`.

---

## How to Add More Knowledge

1. Drop any new `.txt` files into the `docs/` folder.
2. Re-run `python ingest.py` — it clears the old collection and rebuilds from scratch.
3. Start the agent again with `python rag_agent.py`.

---

## License

MIT
