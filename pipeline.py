"""
End-to-end pipeline CLI
========================
Runs the full Zero-Null Vectorless RAG pipeline without the HTTP server:

    python pipeline.py --urls https://example.com --query "What is this site about?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


async def main(seed_urls: list[str], query: str, max_pages: int) -> None:
    from crawler.crawler import Crawler
    from indexer.signposter import build_toc
    from indexer.translator import translate_html_to_markdown
    from parser.chunker import chunk_markdown
    from parser.pruner import prune, split_html
    from retrieval.orchestrator import retrieve

    # ------------------------------------------------------------------ Step 1
    logger.info("Step 1/4 – Crawling %d seed URL(s) …", len(seed_urls))
    crawler = Crawler(seed_urls, max_pages=max_pages)
    results = await crawler.run()
    if not results:
        logger.error("No pages were crawled.  Aborting.")
        sys.exit(1)
    logger.info("Crawled %d page(s).", len(results))

    # ------------------------------------------------------------------ Step 2
    logger.info("Step 2/4 – Pruning → Translating → Chunking …")
    all_chunks = []
    for page in results:
        try:
            pruned = prune(page.html)
            parts = split_html(pruned)
            markdown = await translate_html_to_markdown(parts)
            all_chunks.extend(chunk_markdown(markdown))
        except Exception as exc:
            logger.warning("Skipping %s: %s", page.url, exc)

    if not all_chunks:
        logger.error("No content extracted.  Aborting.")
        sys.exit(1)
    logger.info("Produced %d chunk(s).", len(all_chunks))

    # ------------------------------------------------------------------ Step 3
    logger.info("Step 3/4 – Building Dense Signposts and ToC …")
    toc_path = pathlib.Path("index/toc.json")
    toc = await build_toc(all_chunks, toc_path=toc_path)
    logger.info("ToC written to %s", toc_path)

    # ------------------------------------------------------------------ Step 4
    logger.info("Step 4/4 – Retrieving answer for: %r", query)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    for token in retrieve(query, toc):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Null Vectorless RAG Pipeline")
    parser.add_argument(
        "--urls", nargs="+", required=True, metavar="URL", help="Seed URL(s) to crawl"
    )
    parser.add_argument(
        "--query", required=True, help="Question to answer from crawled content"
    )
    parser.add_argument(
        "--max-pages", type=int, default=50, help="Maximum pages to crawl (default 50)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.urls, args.query, args.max_pages))
