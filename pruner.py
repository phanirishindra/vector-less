"""
HTML Pruner & Splitter
======================
Strips boilerplate DOM elements and, when the remaining HTML exceeds 6 000 tokens
(measured with the Qwen2.5-3B-Instruct tokenizer), splits it at a safe structural
boundary so each piece can be processed individually by the LLM.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterator

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Tags whose entire subtree should be removed (boilerplate / non-content)
_STRIP_TAGS: tuple[str, ...] = (
    "script",
    "style",
    "nav",
    "footer",
    "svg",
    "noscript",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "header",
    "aside",
    "advertisement",
)

# Structural tags used as safe split boundaries (ordered by preference)
_SPLIT_BOUNDARIES: tuple[str, ...] = (
    "section",
    "article",
    "div",
    "main",
)

_TOKEN_LIMIT = 6_000
_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ---------------------------------------------------------------------------
# Tokenizer (lazy-loaded, cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_tokenizer():  # type: ignore[return]
    """Load the Qwen tokenizer once and cache it for the process lifetime."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(_MODEL_ID, use_fast=True)
        logger.info("Loaded tokenizer: %s", _MODEL_ID)
        return tok
    except Exception as exc:
        logger.warning(
            "Could not load %s tokenizer (%s). "
            "Falling back to whitespace-based token estimate.",
            _MODEL_ID,
            exc,
        )
        return None


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    if tok is None:
        # Rough approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    ids = tok.encode(text, add_special_tokens=False)
    return len(ids)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune(html: str) -> str:
    """
    Remove boilerplate tags and return cleaned HTML.

    Steps:
    1. Parse with lxml (fast, lenient).
    2. Decompose every tag in ``_STRIP_TAGS``.
    3. Strip HTML comments.
    4. Return the serialised outer HTML of ``<body>`` (or whole document if absent).
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate subtrees
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove HTML comments
    from bs4 import Comment  # local import to keep top-level clean

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove empty tags that carry no text
    for tag in soup.find_all(True):
        if (
            isinstance(tag, Tag)
            and not tag.get_text(strip=True)
            and tag.name not in ("img", "br", "hr", "td", "th", "tr", "table")
        ):
            tag.decompose()

    body = soup.find("body")
    result = str(body) if body else str(soup)
    logger.debug("Pruned HTML length: %d chars", len(result))
    return result


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_html(html: str) -> list[str]:
    """
    If the pruned HTML is within the token budget, return it as a single-element
    list.  Otherwise, split at structural boundaries and return multiple chunks,
    each within the limit.
    """
    if _count_tokens(html) <= _TOKEN_LIMIT:
        return [html]

    chunks: list[str] = []
    for chunk in _split_at_boundaries(html):
        if _count_tokens(chunk) > _TOKEN_LIMIT:
            # Last resort: hard character split
            for sub in _hard_split(chunk):
                chunks.append(sub)
        else:
            chunks.append(chunk)

    logger.info("HTML split into %d chunk(s) for LLM processing.", len(chunks))
    return chunks


def _split_at_boundaries(html: str) -> Iterator[str]:
    """Yield top-level boundary elements as individual HTML strings."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body") or soup

    boundary_children: list[Tag] = []
    remainder_children: list[Tag] = []

    for child in body.children:
        if not isinstance(child, Tag):
            continue
        if child.name in _SPLIT_BOUNDARIES:
            boundary_children.append(child)
        else:
            remainder_children.append(child)

    if not boundary_children:
        # No structural boundaries found — fall back to top-level divs or hard split
        yield html
        return

    buffer: list[str] = []
    buffer_tokens = 0

    for element in boundary_children:
        elem_html = str(element)
        elem_tokens = _count_tokens(elem_html)

        if buffer_tokens + elem_tokens > _TOKEN_LIMIT and buffer:
            yield "".join(buffer)
            buffer = []
            buffer_tokens = 0

        buffer.append(elem_html)
        buffer_tokens += elem_tokens

    if buffer:
        yield "".join(buffer)

    if remainder_children:
        remainder_html = "".join(str(c) for c in remainder_children)
        if remainder_html.strip():
            yield remainder_html


def _hard_split(html: str) -> list[str]:
    """Character-level hard split as last resort, respecting token budget."""
    # Approx chars-per-token for Qwen: ~4
    chars_per_chunk = _TOKEN_LIMIT * 4
    return [html[i : i + chars_per_chunk] for i in range(0, len(html), chars_per_chunk)]
