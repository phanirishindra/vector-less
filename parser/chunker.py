"""
Semantic Markdown Chunker with Bookend Metadata
================================================
Splits LLM-produced Markdown strictly at header boundaries (#, ##, ###, …).
For every chunk the module also extracts:
  * ``first_sentence`` – opening sentence (fallback retrieval anchor)
  * ``last_sentence``  – closing sentence (fallback retrieval anchor)
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class MarkdownChunk(BaseModel):
    """A single semantically bounded chunk of Markdown content."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    heading: str = ""           # The header line that opens this chunk (may be empty)
    raw_markdown: str = ""      # Full Markdown text of the chunk
    first_sentence: str = ""    # First sentence of plain text in the chunk
    last_sentence: str = ""     # Last sentence of plain text in the chunk


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Sentence boundary: period/!/? followed by whitespace or end-of-string,
# but not inside abbreviations like "e.g." or decimal numbers.
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")
# Strip Markdown formatting for plain-text sentence extraction
_MD_INLINE = re.compile(
    r"(\*{1,3}|_{1,3}|`{1,3}|~~|==|\[([^\]]*)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\))"
)
_MD_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_MD_TABLE_ROW = re.compile(r"^\|.*\|$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chunk_markdown(markdown: str) -> list[MarkdownChunk]:
    """
    Split *markdown* at every Markdown header boundary and return a list of
    :class:`MarkdownChunk` objects with bookend metadata populated.
    """
    chunks = list(_split(markdown))
    return [_annotate(c) for c in chunks]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _split(markdown: str) -> Iterator[MarkdownChunk]:
    """Yield raw (unannotated) chunks split at header lines."""
    lines = markdown.splitlines(keepends=True)
    current_heading = ""
    buffer: list[str] = []

    for line in lines:
        if _HEADER_RE.match(line.rstrip()):
            if buffer:
                raw = "".join(buffer).strip()
                if raw:
                    yield MarkdownChunk(heading=current_heading, raw_markdown=raw)
            current_heading = line.rstrip()
            buffer = [line]
        else:
            buffer.append(line)

    # Last chunk
    if buffer:
        raw = "".join(buffer).strip()
        if raw:
            yield MarkdownChunk(heading=current_heading, raw_markdown=raw)


def _annotate(chunk: MarkdownChunk) -> MarkdownChunk:
    """Populate ``first_sentence`` and ``last_sentence`` fields."""
    plain = _to_plain(chunk.raw_markdown)
    sentences = [s.strip() for s in _SENTENCE_SEP.split(plain) if s.strip()]
    if sentences:
        chunk.first_sentence = sentences[0]
        chunk.last_sentence = sentences[-1]
    return chunk


def _to_plain(md: str) -> str:
    """Strip Markdown formatting to get clean plain text for sentence extraction."""
    text = _MD_CODE_BLOCK.sub(" ", md)          # remove code blocks
    text = _MD_TABLE_ROW.sub(" ", text)          # remove table rows
    text = _MD_INLINE.sub(r"\2", text)           # unwrap inline markup
    text = re.sub(r"^[#\-*>]+\s*", "", text, flags=re.MULTILINE)  # headings/bullets
    text = re.sub(r"\s+", " ", text).strip()
    return text
