# Vector-Less — "Zero-Null" Vectorless RAG System

A high-performance, memory-efficient Retrieval-Augmented Generation (RAG) pipeline
that uses a **local LLM** (Qwen 2.5B via `llama.cpp`) for every stage — crawling,
translation, indexing, and retrieval — with **zero vector embeddings**.

---

## Architecture

```
Seed URLs
   │
   ▼
┌─────────────────────────────────┐
│  crawler/crawler.py             │  aiohttp + Playwright (stealth)
│  Bloom Filter deduplication     │  Async, configurable concurrency
└────────────────┬────────────────┘
                 │ raw HTML
                 ▼
┌─────────────────────────────────┐
│  parser/pruner.py               │  BeautifulSoup — strips <script>,
│  HTML Pruner & Splitter         │  <style>, <nav>, <svg>, <footer> …
└────────────────┬────────────────┘  Splits at 6 000-token boundary
                 │ clean HTML chunk(s)
                 ▼
┌─────────────────────────────────┐
│  indexer/translator.py          │  LLM call: HTML → Markdown
│  LLM Translation (Task 1)       │  Preserves tables + code blocks
└────────────────┬────────────────┘
                 │ Markdown
                 ▼
┌─────────────────────────────────┐
│  parser/chunker.py              │  Splits at # / ## headers
│  Semantic Chunker               │  Extracts first_sentence / last_sentence
└────────────────┬────────────────┘
                 │ MarkdownChunk list
                 ▼
┌─────────────────────────────────┐
│  indexer/signposter.py          │  LLM call: Dense Signpost (≤ 30 tokens)
│  Dense Signposting (Task 2)     │  Saves JSON ToC: chunk_id, signpost,
└────────────────┬────────────────┘  first/last sentence, raw_markdown
                 │ toc.json
                 ▼
┌─────────────────────────────────────────────────────────┐
│  retrieval/orchestrator.py  (Task 3)                    │
│                                                         │
│  Layer 1  DeepSieve  — LLM deconstructs vague query     │
│           <think> scratchpad hidden from user output    │
│  Layer 2  ToC Router — LLM selects relevant chunk_ids  │
│  Layer 3  Iterative Exploration + Synthesis             │
│           MCTS-lite: LLM can request "explore_parent"   │
│  Layer 4  BM25 Fallback (rank_bm25) → grounded LLM     │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| `llama.cpp` HTTP server | Must expose an OpenAI-compatible API on `http://127.0.0.1:8000/v1` |
| Qwen 2.5-3B-Instruct GGUF | Loaded by `llama.cpp` |
| Playwright browsers | Installed via `playwright install chromium` |

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Playwright's Chromium browser
playwright install chromium
```

---

## Running llama.cpp

```bash
# Example: serve Qwen 2.5-3B-Instruct GGUF on port 8000
./llama-server \
  -m qwen2.5-3b-instruct-q4_k_m.gguf \
  --port 8000 \
  --ctx-size 8192 \
  --alias qwen2.5
```

The system uses `http://127.0.0.1:8000/v1` and API key `sk-local` — it **never**
contacts the real OpenAI API.

---

## Usage

### Option A — CLI pipeline (all-in-one)

```bash
python pipeline.py \
  --urls https://example.com \
  --query "What does this site offer?" \
  --max-pages 30
```

### Option B — HTTP API server

```bash
# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Crawl & index
curl -X POST http://localhost:8080/crawl \
  -H "Content-Type: application/json" \
  -d '{"seed_urls": ["https://example.com"], "max_pages": 30}'

# Query (streamed answer)
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does this site offer?"}'

# Inspect the Table of Contents
curl http://localhost:8080/toc

# Health check
curl http://localhost:8080/health
```

---

## Project Structure

```
vector-less/
├── crawler/
│   ├── __init__.py
│   └── crawler.py        # Async crawling (aiohttp + Playwright + Bloom Filter)
├── parser/
│   ├── __init__.py
│   ├── pruner.py         # HTML pruning & token-aware splitting
│   └── chunker.py        # Markdown chunking + bookend metadata
├── indexer/
│   ├── __init__.py
│   ├── translator.py     # LLM: HTML → Markdown (Task 1)
│   └── signposter.py     # LLM: Dense Signposts + JSON ToC (Task 2)
├── retrieval/
│   ├── __init__.py
│   └── orchestrator.py   # 4-layer retrieval orchestrator (Task 3)
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI REST + SSE server
├── pipeline.py           # CLI entry point (no HTTP server needed)
├── requirements.txt
└── README.md
```

---

## Configuration

All LLM endpoint settings live in two places:

| File | Variable | Default |
|---|---|---|
| `indexer/translator.py` | `_LOCAL_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `indexer/signposter.py` | `_LOCAL_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `retrieval/orchestrator.py` | `_LOCAL_BASE_URL` | `http://127.0.0.1:8000/v1` |

Set `_MODEL` to the model alias registered in your `llama.cpp` server.

---

## Key Design Decisions

* **No vector embeddings** — `sentence-transformers` is never imported.
* **No html2text / markitdown** — all HTML→Markdown conversion is LLM-driven.
* **Memory-safe splitting** — HTML is split at structural boundaries before
  being sent to the LLM, preventing OOM on low-RAM machines.
* **`<think>` suppression** — the streaming generator in
  `retrieval/orchestrator.py` intercepts and discards `<think>…</think>` tokens
  so the user only sees clean output.
* **BM25 guarantee** — even if the LLM router returns `[]`, the system always
  produces an answer via lexical fallback.
