"""
Dense Conceptual Signposting & JSON Table-of-Contents Builder
=============================================================
For every Markdown chunk the LLM generates a ≤30-token "Dense Signpost"
formatted as:

    [Core Theme] + [Key Entities] + [Questions Answered]

Results are persisted locally as a JSON file that acts as a lightweight
Table of Contents (ToC) for the retrieval layer.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Sequence

from openai import OpenAI
from pydantic import BaseModel, Field

from parser.chunker import MarkdownChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local LLM client
# ---------------------------------------------------------------------------
_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"

_DEFAULT_TOC_PATH = pathlib.Path("index/toc.json")


def _build_client() -> OpenAI:
    return OpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class ToCEntry(BaseModel):
    """One entry in the Table of Contents."""

    chunk_id: str
    dense_signpost: str        # ≤30 token compressed descriptor
    first_sentence: str
    last_sentence: str
    raw_markdown: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_SIGNPOST = """\
You are a conceptual indexer.  Given a Markdown chunk, output a single-line
Dense Signpost of at most 30 tokens in this exact format:

[Core Theme] + [Key Entities] + [Questions Answered]

Examples:
  [Machine Learning] + [PyTorch, GPUs] + [How to train a model?, What is backprop?]
  [Cookie Policy] + [GDPR, consent] + [What data is collected?, How to opt out?]

Output ONLY the signpost line.  No preamble, no explanation.
"""

_USER_SIGNPOST_TEMPLATE = """\
Markdown chunk:

{chunk}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_signpost(chunk: MarkdownChunk, client: OpenAI) -> str:
    """Generate a Dense Signpost for a single chunk via the local LLM."""
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_SIGNPOST},
                {
                    "role": "user",
                    "content": _USER_SIGNPOST_TEMPLATE.format(
                        chunk=chunk.raw_markdown[:3000]  # guard against huge chunks
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=80,
        )
        signpost = (response.choices[0].message.content or "").strip()
        # Ensure it truly is a single line
        return signpost.splitlines()[0] if signpost else ""
    except Exception as exc:
        logger.error("Signpost generation failed for chunk %s: %s", chunk.chunk_id, exc)
        return ""


def build_toc(
    chunks: Sequence[MarkdownChunk],
    *,
    toc_path: pathlib.Path = _DEFAULT_TOC_PATH,
) -> list[ToCEntry]:
    """
    Generate Dense Signposts for all chunks and persist the ToC as JSON.

    Parameters
    ----------
    chunks:
        Annotated :class:`~parser.chunker.MarkdownChunk` objects (with bookend
        metadata already populated).
    toc_path:
        File path where the JSON ToC will be written.

    Returns
    -------
    list[ToCEntry]
        The in-memory representation of the ToC.
    """
    client = _build_client()
    entries: list[ToCEntry] = []

    for i, chunk in enumerate(chunks):
        logger.info(
            "Signposting chunk %d/%d  [%s] …", i + 1, len(chunks), chunk.chunk_id
        )
        signpost = build_signpost(chunk, client)
        entry = ToCEntry(
            chunk_id=chunk.chunk_id,
            dense_signpost=signpost,
            first_sentence=chunk.first_sentence,
            last_sentence=chunk.last_sentence,
            raw_markdown=chunk.raw_markdown,
        )
        entries.append(entry)

    _persist(entries, toc_path)
    return entries


def _persist(entries: list[ToCEntry], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.model_dump() for e in entries]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("ToC written to %s  (%d entries)", path, len(entries))


def load_toc(toc_path: pathlib.Path = _DEFAULT_TOC_PATH) -> list[ToCEntry]:
    """Load a previously persisted ToC from disk."""
    with toc_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ToCEntry(**item) for item in raw]
