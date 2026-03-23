"""
Tests for _stream_hide_think in retrieval/orchestrator.py.

Verifies:
- <think>...</think> segments are removed from streamed final output.
- Normal text outside <think> tags continues streaming correctly.
- Behaviour is correct across chunk boundaries (e.g. '<th' in one chunk and 'ink>' in another).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator

import pytest

from retrieval.orchestrator import _stream_hide_think


def _make_stream(tokens: list[str]) -> Iterator[SimpleNamespace]:
    """Build a fake streaming iterator from a list of token strings."""
    for token in tokens:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
        )


def _collect(tokens: list[str]) -> str:
    """Run _stream_hide_think on the given token list and join the output."""
    return "".join(_stream_hide_think(_make_stream(tokens)))


# ---------------------------------------------------------------------------
# Normal text — no <think> tags
# ---------------------------------------------------------------------------

def test_plain_text_passes_through():
    result = _collect(["Hello", ", ", "world", "!"])
    assert result == "Hello, world!"


def test_empty_stream_yields_nothing():
    result = _collect([])
    assert result == ""


def test_single_empty_token():
    result = _collect([""])
    assert result == ""


# ---------------------------------------------------------------------------
# Complete <think>…</think> block in a single chunk
# ---------------------------------------------------------------------------

def test_think_block_removed():
    result = _collect(["<think>private reasoning</think>public answer"])
    assert result == "public answer"
    assert "<think>" not in result
    assert "private reasoning" not in result


def test_text_before_and_after_think():
    result = _collect(["Intro. <think>scratchpad</think> Conclusion."])
    assert result == "Intro.  Conclusion."


def test_multiple_think_blocks():
    result = _collect(["A<think>X</think>B<think>Y</think>C"])
    assert result == "ABC"


# ---------------------------------------------------------------------------
# <think> block split across chunk boundaries
# ---------------------------------------------------------------------------

def test_open_tag_split_across_chunks():
    """'<th' in one chunk, 'ink>' in the next — tag must still be suppressed."""
    result = _collect(["before <th", "ink>hidden</think> after"])
    assert result == "before  after"
    assert "hidden" not in result


def test_close_tag_split_across_chunks():
    """'</thi' in one chunk, 'nk>' in the next."""
    result = _collect(["<think>hidden</thi", "nk> visible"])
    assert result == " visible"
    assert "hidden" not in result


def test_think_content_split_across_many_chunks():
    """Content inside <think> arrives piece by piece over many chunks."""
    tokens = ["<", "t", "h", "i", "n", "k", ">", "sec", "ret", "</", "think", ">", "ok"]
    result = _collect(tokens)
    assert result == "ok"
    assert "secret" not in result


def test_text_before_split_open_tag():
    """Text before a split open tag must still be emitted."""
    result = _collect(["visible <th", "ink>hidden</think>"])
    assert result == "visible "
    assert "hidden" not in result


# ---------------------------------------------------------------------------
# Nested / malformed edge cases
# ---------------------------------------------------------------------------

def test_no_closing_tag_discards_content():
    """If </think> never arrives, content after <think> is silently discarded."""
    result = _collect(["before <think>no closing tag ever"])
    assert result == "before "
    assert "no closing" not in result


def test_think_only_stream():
    """Stream contains nothing but a <think> block."""
    result = _collect(["<think>only private stuff</think>"])
    assert result == ""
