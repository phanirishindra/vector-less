"""
FastAPI REST Interface for the Zero-Null Vectorless RAG System
==============================================================
Exposes endpoints to:
  /crawl      – trigger crawl + index pipeline
  /query      – run the multi-layer retrieval orchestrator (streamed SSE)
  /toc        – inspect the current Table of Contents
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from crawler.crawler import Crawler, CrawlResult
from indexer.signposter import ToCEntry, build_toc, load_toc
from indexer.translator import translate_html_to_markdown
from parser.chunker import MarkdownChunk, chunk_markdown
from parser.pruner import prune, split_html
from retrieval.orchestrator import retrieve

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_TOC_PATH = pathlib.Path("index/toc.json")

# Concurrency controls:
# - _PAGE_CONCURRENCY limits total in-flight page processing tasks
# - _LLM_CONCURRENCY limits concurrent LLM inference calls (protects llama.cpp)
_PAGE_CONCURRENCY = 4
_LLM_CONCURRENCY = 2

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_toc: list[ToCEntry] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load persisted ToC on startup if it exists."""
    global _toc
    if _TOC_PATH.exists():
        try:
            _toc = load_toc(_TOC_PATH)
            logger.info("Loaded ToC with %d entries from disk.", len(_toc))
        except Exception as exc:
            logger.warning("Could not load ToC: %s", exc)
    yield


app = FastAPI(
    title="Zero-Null Vectorless RAG",
    description=(
        "Crawl → Prune → Translate → Signpost → Retrieve pipeline "
        "powered entirely by a local LLM (no vector embeddings)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class CrawlRequest(BaseModel):
    seed_urls: list[str]
    max_pages: int = 50
    same_origin_only: bool = True


class CrawlResponse(BaseModel):
    pages_crawled: int
    chunks_indexed: int
    toc_path: str


class QueryRequest(BaseModel):
    query: str


class ToCResponse(BaseModel):
    entries: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/crawl", response_model=CrawlResponse)
async def crawl_and_index(request: CrawlRequest) -> CrawlResponse:
    """
    Crawl the given seed URLs, prune HTML, translate to Markdown via the local
    LLM, chunk, signpost, and persist the ToC.
    """
    global _toc

    # 1) Crawl
    crawler = Crawler(
        request.seed_urls,
        max_pages=request.max_pages,
        same_origin_only=request.same_origin_only,
    )
    results: list[CrawlResult] = await crawler.run()
    if not results:
        raise HTTPException(status_code=422, detail="No pages could be crawled.")

    # 2) Per-page processing with bounded concurrency:
    #    - CPU-bound ops in thread pool
    #    - LLM ops awaited directly, guarded by semaphore
    all_chunks: list[MarkdownChunk] = []
    loop = asyncio.get_running_loop()
    page_sem = asyncio.Semaphore(_PAGE_CONCURRENCY)
    llm_sem = asyncio.Semaphore(_LLM_CONCURRENCY)

    async def _process_page(page: CrawlResult) -> list[MarkdownChunk]:
        async with page_sem:
            try:
                # CPU-bound: prune and split in thread pool
                pruned = await loop.run_in_executor(None, prune, page.html)
                html_parts = await loop.run_in_executor(None, split_html, pruned)

                # LLM inference: awaited directly (async), with bounded concurrency
                async with llm_sem:
                    markdown = await translate_html_to_markdown(html_parts)

                # CPU-bound: chunking in thread pool
                chunks = await loop.run_in_executor(None, chunk_markdown, markdown)
                return chunks
            except Exception as exc:
                logger.warning("Processing error for %s: %s", page.url, exc)
                return []

    page_chunk_lists = await asyncio.gather(
        *(_process_page(page) for page in results),
        return_exceptions=False,
    )

    for chunks in page_chunk_lists:
        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=422, detail="No content could be extracted.")

    # 3) Signpost and persist ToC (awaited directly — async LLM calls)
    _toc = await build_toc(all_chunks, toc_path=_TOC_PATH)

    return CrawlResponse(
        pages_crawled=len(results),
        chunks_indexed=len(_toc),
        toc_path=str(_TOC_PATH),
    )


@app.post("/query")
async def query_rag(request: QueryRequest) -> StreamingResponse:
    """
    Run the multi-layer retrieval orchestrator and stream the answer back
    using Server-Sent Events. The <think> scratchpad is never surfaced.
    """
    if not _toc:
        raise HTTPException(
            status_code=503,
            detail="No indexed content available. POST /crawl first.",
        )

    async def _generate() -> AsyncIterator[str]:
        for token in retrieve(request.query, _toc):
            yield token
        yield "\n"

    return StreamingResponse(_generate(), media_type="text/plain")


@app.get("/toc", response_model=ToCResponse)
async def get_toc() -> ToCResponse:
    """Return the current in-memory Table of Contents (signposts only)."""
    return ToCResponse(
        entries=[
            {
                "chunk_id": e.chunk_id,
                "dense_signpost": e.dense_signpost,
                "first_sentence": e.first_sentence,
                "last_sentence": e.last_sentence,
            }
            for e in _toc
        ]
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "toc_entries": len(_toc)}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
