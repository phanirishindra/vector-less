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
import re
from typing import Sequence

from openai import AsyncOpenAI
from pydantic import BaseModel

from parser.chunker import MarkdownChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local LLM client
# ---------------------------------------------------------------------------
_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"

_DEFAULT_TOC_PATH = pathlib.Path("index/toc.json")


def _build_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class ToCEntry(BaseModel):
    """One entry in the Table of Contents."""

    chunk_id: str
    dense_signpost: str  # ≤30 token compressed descriptor
    first_sentence: str
    last_sentence: str
    raw_markdown: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_SIGNPOST = """\
You are a conceptual indexer. You receive a Markdown chunk and output a single-line Dense Signpost of at most 30 tokens in this exact format:

[Core Theme] + [Key Entities] + [Questions Answered]

Output rules:
1. Output the signpost line only. Start your response with "[". Do not write any introduction, explanation, or closing remark.
2. The entire output must fit on one line with no line breaks.
3. Keep the total length to 30 tokens or fewer.
4. Use the bracket-plus format exactly as shown.

Example:
--------
Input:
## Privacy Policy

We collect your name, email address, and usage data under GDPR.
You may request deletion of your data at any time by contacting support.

Output:
[Privacy Policy] + [GDPR, email, usage data] + [What data is collected?, How to delete data?]
--------
Follow this pattern exactly for every Markdown chunk you receive.
"""

_USER_SIGNPOST_TEMPLATE = """\
Markdown chunk:

{chunk}
"""

# ---------------------------------------------------------------------------
# Output hardening (sanitize + validate)
# ---------------------------------------------------------------------------
_RE_LEADING_CHATTER = re.compile(
    r"^\s*(sure|here(?:'s| is)|i(?:\s+have)?\s+(?:generated|created|formatted)|output:)\b",
    flags=re.IGNORECASE,
)

# Exact one-line bracket-plus-bracket-plus-bracket shape
_RE_SIGNPOST_EXACT = re.compile(
    r"^\s*\[[^\[\]\n]+\]\s\+\s\[[^\[\]\n]+\]\s\+\s\[[^\[\]\n]+\]\s*$"
)


def _approx_token_count(text: str) -> int:
    """
    Approximate token count for guardrail purposes.
    This is intentionally lightweight and model-agnostic.
    """
    # Splits words and punctuation as separate pieces.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _sanitize_signpost(text: str) -> str:
    """
    Sanitize model output:
    - trim
    - keep first non-empty line only
    """
    s = (text or "").strip()
    if not s:
        return ""
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _validate_signpost(signpost: str) -> tuple[bool, str]:
    """
    Validate strict signpost contract:
    - non-empty
    - no preface/chatter
    - single line
    - exact [A] + [B] + [C] format
    - <= ~30 tokens (approximate)
    """
    if not signpost:
        return False, "empty_output"

    if _RE_LEADING_CHATTER.match(signpost):
        return False, "leading_commentary"

    if "\n" in signpost or "\r" in signpost:
        return False, "multiline_output"

    if not _RE_SIGNPOST_EXACT.match(signpost):
        return False, "format_mismatch"

    token_count = _approx_token_count(signpost)
    if token_count > 30:
        return False, f"too_many_tokens_{token_count}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def build_signpost(
    chunk: MarkdownChunk,
    client: AsyncOpenAI,
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Generate a Dense Signpost for a single chunk via the local LLM with retries."""
    max_attempts = 3
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_SIGNPOST},
        {
            "role": "user",
            "content": _USER_SIGNPOST_TEMPLATE.format(
                chunk=chunk.raw_markdown[:3000]  # guard against huge chunks
            ),
        },
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=80,
            )

            raw = response.choices[0].message.content or ""
            signpost = _sanitize_signpost(raw)
            ok, reason = _validate_signpost(signpost)

            if ok:
                return signpost

            preview = signpost[:120].replace("\n", "\\n")
            logger.warning(
                "Signpost format violation (chunk %s/%s, id=%s, attempt %d/%d): %s | preview=%r",
                chunk_index if chunk_index is not None else "?",
                total_chunks if total_chunks is not None else "?",
                chunk.chunk_id,
                attempt,
                max_attempts,
                reason,
                preview,
            )

            if attempt < max_attempts:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response violated the contract. "
                            "Return EXACTLY one line in this format only: "
                            "[Core Theme] + [Key Entities] + [Questions Answered]. "
                            "No extra words, no labels, no line breaks."
                        ),
                    }
                )
                continue

            raise ValueError(
                f"Signpost generation failed validation after {max_attempts} attempts: {reason}"
            )

        except Exception as exc:
            logger.error(
                "Signpost generation failed (chunk %s/%s, id=%s, attempt %d/%d): %s",
                chunk_index if chunk_index is not None else "?",
                total_chunks if total_chunks is not None else "?",
                chunk.chunk_id,
                attempt,
                max_attempts,
                exc,
            )
            if attempt >= max_attempts:
                return ""

    return ""


async def build_toc(
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
            "Signposting chunk %d/%d [%s] ...", i + 1, len(chunks), chunk.chunk_id
        )
        signpost = await build_signpost(
            chunk,
            client,
            chunk_index=i + 1,
            total_chunks=len(chunks),
        )
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
    logger.info("ToC written to %s (%d entries)", path, len(entries))


def load_toc(toc_path: pathlib.Path = _DEFAULT_TOC_PATH) -> list[ToCEntry]:
    """Load a previously persisted ToC from disk."""
    with toc_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ToCEntry(**item) for item in raw]
