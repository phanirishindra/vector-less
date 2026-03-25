```python
"""
LLM-Driven HTML → Markdown Translator
=======================================
Passes pruned HTML to the local llama.cpp server (OpenAI-compatible API) and
returns clean, semantic Markdown. Never uses html2text or markitdown.

The client is pinned to the local inference endpoint so that real OpenAI is
never contacted.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local LLM client (pinned to llama.cpp / any OpenAI-compatible local server)
# ---------------------------------------------------------------------------
_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"  # model name as registered in the local server


def _build_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an HTML-to-Markdown converter. You receive raw HTML and output only the converted Markdown — nothing else.

Output rules:
1. Output Markdown only. Start your response with the first Markdown character. Do not write any introduction, explanation, or closing remark.
2. Use # for the page title, ## for top-level sections, ### for sub-sections.
3. Preserve tables as GitHub-Flavored Markdown pipe tables with a separator row containing |---|.
4. Preserve code blocks with triple backticks (```) and the original language hint.
5. Reproduce only the content present in the HTML — never fabricate content.

Example:
--------
Input:
```html
<h1>Getting Started</h1>
<p>Install the package with <code>pip install foo</code>.</p>
<h2>Configuration</h2>
<p>Set the <strong>API_KEY</strong> environment variable.</p>
```

Output:
# Getting Started

Install the package with `pip install foo`.

## Configuration

Set the **API_KEY** environment variable.
--------
Follow this pattern exactly for every HTML fragment you receive.
"""

_USER_TEMPLATE = """\
Convert the following HTML fragment to Markdown:

```html
{html}
```
"""

# ---------------------------------------------------------------------------
# Output hardening (sanitize + validate)
# ---------------------------------------------------------------------------
_RE_LEADING_CHATTER = re.compile(
    r"^\s*(sure|here(?:'s| is)|i(?:\s+have)?\s+(?:converted|translated)|the converted markdown|output:)\b",
    flags=re.IGNORECASE,
)
_RE_HTML_TAG_START = re.compile(r"^\s*<[^>]+>")
_RE_TOP_FENCE = re.compile(
    r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$",
    flags=re.IGNORECASE,
)


def _sanitize_markdown_output(text: str) -> str:
    """Trim output and unwrap an accidental single top-level markdown fence."""
    s = (text or "").strip()
    m = _RE_TOP_FENCE.match(s)
    if m:
        s = m.group(1).strip()
    return s


def _validate_markdown_output(text: str) -> tuple[bool, str]:
    """
    Validate output contract:
    - non-empty
    - no leading assistant chatter
    - not raw HTML
    """
    if not text.strip():
        return False, "empty_output"
    if _RE_LEADING_CHATTER.match(text):
        return False, "leading_commentary"
    if _RE_HTML_TAG_START.match(text):
        return False, "looks_like_html_not_markdown"
    return True, "ok"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def translate_html_to_markdown(html_chunks: Sequence[str]) -> str:
    """
    Translate one or more HTML chunks to Markdown and concatenate the results.

    Parameters
    ----------
    html_chunks:
        One or more pruned HTML strings (already split to fit within the token
        budget by :mod:`parser.pruner`).

    Returns
    -------
    str
        Concatenated Markdown from all chunks.
    """
    client = _build_client()
    parts: list[str] = []

    for i, chunk in enumerate(html_chunks):
        logger.debug("Translating HTML chunk %d/%d ...", i + 1, len(html_chunks))
        markdown = await _call_llm(
            client,
            chunk,
            chunk_index=i + 1,
            total_chunks=len(html_chunks),
        )
        parts.append(markdown)

    return "\n\n".join(parts)


async def _call_llm(
    client: AsyncOpenAI,
    html: str,
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """Single LLM call with strict validation and bounded retries."""
    max_attempts = 3
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(html=html)},
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
            content = _sanitize_markdown_output(raw)

            ok, reason = _validate_markdown_output(content)
            if ok:
                return content

            preview = content[:120].replace("\n", "\\n")
            logger.warning(
                "Translator format violation (chunk %s/%s, attempt %d/%d): %s | preview=%r",
                chunk_index if chunk_index is not None else "?",
                total_chunks if total_chunks is not None else "?",
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
                            "Your previous response violated the format contract. "
                            "Return ONLY raw Markdown. No commentary. No preface."
                        ),
                    }
                )
                continue

            raise ValueError(
                f"LLM output failed validation after {max_attempts} attempts: {reason}"
            )

        except Exception as exc:
            logger.error(
                "LLM translation error (chunk %s/%s, attempt %d/%d): %s",
                chunk_index if chunk_index is not None else "?",
                total_chunks if total_chunks is not None else "?",
                attempt,
                max_attempts,
                exc,
            )
            if attempt >= max_attempts:
                raise

    raise RuntimeError("Unreachable state in _call_llm")
```
