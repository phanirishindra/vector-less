"""
High-Performance Asynchronous Web Crawler
==========================================
Uses aiohttp for link discovery and Playwright (stealth) for JS-heavy pages.
A Bloom Filter deduplicates visited URLs to keep memory usage constant.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Iterable
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from pybloom_live import BloomFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic-style dataclass for crawl results
# ---------------------------------------------------------------------------
from pydantic import BaseModel, HttpUrl, field_validator


class CrawlResult(BaseModel):
    """Represents a single scraped page."""

    url: str
    html: str
    status_code: int
    rendered: bool  # True when Playwright was used


# ---------------------------------------------------------------------------
# Bloom-filter-backed URL frontier
# ---------------------------------------------------------------------------
class URLFrontier:
    """Thread-safe URL frontier backed by a Bloom Filter for O(1) deduplication."""

    def __init__(self, capacity: int = 100_000, error_rate: float = 0.001) -> None:
        self._seen: BloomFilter = BloomFilter(capacity=capacity, error_rate=error_rate)
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def seed(self, urls: Iterable[str]) -> None:
        for url in urls:
            self.push(url)

    def push(self, url: str) -> bool:
        """Enqueue URL if not previously seen.  Returns True when enqueued."""
        clean = _normalise(url)
        if not clean or clean in self._seen:
            return False
        self._seen.add(clean)
        self._queue.put_nowait(clean)
        return True

    async def pop(self) -> str:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()

    def task_done(self) -> None:
        self._queue.task_done()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"^https?://", re.I)


def _normalise(url: str) -> str | None:
    """Strip fragment, normalise scheme. Returns None for non-HTTP URLs."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return None
        # Drop fragment; keep query string (sites like wikis use it)
        normalised = parsed._replace(fragment="").geturl()
        return normalised
    except Exception:
        return None


def _same_origin(base: str, candidate: str) -> bool:
    b = urlparse(base)
    c = urlparse(candidate)
    return b.netloc == c.netloc


def _extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        absolute = urljoin(base_url, href)
        norm = _normalise(absolute)
        if norm:
            links.append(norm)
    return links


# ---------------------------------------------------------------------------
# Playwright stealth helpers
# ---------------------------------------------------------------------------
async def _apply_stealth(page: Page) -> None:
    """Minimal stealth: hide navigator.webdriver, spoof user-agent languages."""
    await page.add_init_script(
        """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
        """
    )


# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------
class Crawler:
    """
    Async crawler that:
    1. Uses aiohttp for fast, simple pages.
    2. Falls back to Playwright for JS-rendered content.
    3. Deduplicates via a Bloom Filter frontier.
    4. Stays within a single origin (configurable).
    """

    _STEALTH_UA = (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
    _JS_INDICATORS = ["__NEXT_DATA__", "window.__nuxt__", "ng-version", "react-root"]

    def __init__(
        self,
        seed_urls: list[str],
        *,
        max_pages: int = 200,
        concurrency: int = 8,
        same_origin_only: bool = True,
        request_timeout: float = 20.0,
        playwright_timeout: float = 30_000.0,
    ) -> None:
        self.seed_urls = seed_urls
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.same_origin_only = same_origin_only
        self.request_timeout = request_timeout
        self.playwright_timeout = playwright_timeout

        self._frontier = URLFrontier()
        self._frontier.seed(seed_urls)
        self._results: list[CrawlResult] = []
        self._pages_visited = 0
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self) -> list[CrawlResult]:
        """Execute full crawl and return all results."""
        async with async_playwright() as pw:
            self._browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            self._context = await self._browser.new_context(
                user_agent=self._STEALTH_UA,
                java_script_enabled=True,
                ignore_https_errors=True,
            )

            connector = aiohttp.TCPConnector(limit=self.concurrency, ssl=False)
            headers = {"User-Agent": self._STEALTH_UA}
            async with aiohttp.ClientSession(
                connector=connector, headers=headers
            ) as session:
                workers = [
                    self._worker(session) for _ in range(self.concurrency)
                ]
                await asyncio.gather(*workers)

            await self._context.close()
            await self._browser.close()

        logger.info("Crawl complete. Pages visited: %d", self._pages_visited)
        return self._results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _worker(self, session: aiohttp.ClientSession) -> None:
        while True:
            if self._frontier.empty() or self._pages_visited >= self.max_pages:
                break
            try:
                url = await asyncio.wait_for(self._frontier.pop(), timeout=2.0)
            except asyncio.TimeoutError:
                break

            try:
                result = await self._fetch(session, url)
                if result:
                    self._results.append(result)
                    self._pages_visited += 1

                    # Discover child links, respect same-origin constraint
                    for link in _extract_links(url, result.html):
                        if self.same_origin_only and not _same_origin(url, link):
                            continue
                        self._frontier.push(link)
            except Exception as exc:
                logger.warning("Error fetching %s: %s", url, exc)
            finally:
                self._frontier.task_done()

    async def _fetch(
        self, session: aiohttp.ClientSession, url: str
    ) -> CrawlResult | None:
        """Try aiohttp first; escalate to Playwright if JS rendering is needed."""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            ) as resp:
                if resp.status >= 400:
                    logger.debug("HTTP %d for %s", resp.status, url)
                    return None
                html = await resp.text(errors="replace")
                if self._needs_js(html):
                    logger.debug("JS rendering required for %s", url)
                    return await self._playwright_fetch(url)
                return CrawlResult(
                    url=url, html=html, status_code=resp.status, rendered=False
                )
        except aiohttp.ClientError as exc:
            logger.warning("aiohttp error for %s: %s – trying Playwright", url, exc)
            return await self._playwright_fetch(url)

    def _needs_js(self, html: str) -> bool:
        """Heuristic: page uses a JS framework that needs client-side rendering."""
        return any(indicator in html for indicator in self._JS_INDICATORS)

    async def _playwright_fetch(self, url: str) -> CrawlResult | None:
        if self._context is None:
            return None
        page: Page = await self._context.new_page()
        try:
            await _apply_stealth(page)
            response = await page.goto(
                url,
                timeout=self.playwright_timeout,
                wait_until="networkidle",
            )
            status = response.status if response else 200
            html = await page.content()
            return CrawlResult(url=url, html=html, status_code=status, rendered=True)
        except Exception as exc:
            logger.warning("Playwright error for %s: %s", url, exc)
            return None
        finally:
            await page.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def _main(urls: list[str], max_pages: int) -> None:
    crawler = Crawler(urls, max_pages=max_pages)
    results = await crawler.run()
    for r in results:
        print(f"[{'PW' if r.rendered else 'AH'}] {r.status_code}  {r.url}")


if __name__ == "__main__":
    import sys

    seed = sys.argv[1:] or ["https://example.com"]
    asyncio.run(_main(seed, max_pages=50))
