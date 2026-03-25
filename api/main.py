from __future__ import annotations

from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from retrieval.orchestrator import retrieve

app = FastAPI(title="Vector-less RAG")

ROOT = Path(__file__).resolve().parents[1]
TOC_PATH = ROOT / "processed" / "toc.json"

_toc: str | None = None


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def _startup() -> None:
    global _toc
    _toc = TOC_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Vector-less RAG</title>
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 900px;
        margin: 2rem auto;
        padding: 0 1rem;
      }
      textarea {
        width: 100%;
        min-height: 120px;
      }
      pre {
        white-space: pre-wrap;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        min-height: 140px;
      }
      button {
        margin-top: 0.5rem;
      }
    </style>
  </head>
  <body>
    <h1>Vector-less RAG</h1>
    <textarea id="q" placeholder="Ask a question..."></textarea>
    <br />
    <button id="ask">Ask</button>
    <h3>Answer</h3>
    <pre id="out"></pre>

    <script>
      const askBtn = document.getElementById("ask");
      const out = document.getElementById("out");
      const q = document.getElementById("q");

      askBtn.onclick = async () => {
        out.textContent = "";
        const res = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q.value }),
        });

        if (!res.ok || !res.body) {
          out.textContent = `Request failed: ${res.status}`;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          out.textContent += decoder.decode(value, { stream: true });
        }
        out.textContent += decoder.decode();
      };
    </script>
  </body>
</html>
"""


@app.post("/query")
def query(request: QueryRequest) -> StreamingResponse:
    """
    QUICK FIX (Option A):
    Use a synchronous generator so Starlette/FastAPI can execute streaming work
    in a threadpool, preventing the main event loop from being blocked by
    synchronous network calls inside retrieval.orchestrator.retrieve().
    """

    def _generate() -> Iterator[str]:
        for token in retrieve(request.query, _toc):
            yield token

    return StreamingResponse(_generate(), media_type="text/plain")
