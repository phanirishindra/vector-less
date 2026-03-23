"""
LLM-Driven HTML → Markdown Translator
=======================================
Passes pruned HTML to the local llama.cpp server (OpenAI-compatible API) and
returns clean, semantic Markdown.  Never uses html2text or markitdown.

The client is pinned to the local inference endpoint so that real OpenAI is
never contacted.
"""

from __future__ import annotations

import logging
from typing import Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local LLM client (pinned to llama.cpp / any OpenAI-compatible local server)
# ---------------------------------------------------------------------------
_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"          # model name as registered in the local server


def _build_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an expert HTML-to-Markdown converter.
Your only task is to translate the raw HTML provided by the user into pristine,
semantic Markdown that faithfully preserves all informational content.

Rules:
- Preserve tables using GitHub-Flavored Markdown pipe syntax with separator rows (|---|).
- Preserve code blocks with triple backticks and language hints.
- Do NOT include any commentary, preamble, or explanation — output Markdown only.
- Use # for the page title, ## for top-level sections, ### for sub-sections.
- Do not fabricate content.  Translate only what is present in the HTML.
"""

_USER_TEMPLATE = """\
Convert the following HTML fragment to Markdown:

```html
{html}
```
"""


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
        logger.debug("Translating HTML chunk %d/%d …", i + 1, len(html_chunks))
        markdown = await _call_llm(client, chunk)
        parts.append(markdown)

    return "\n\n".join(parts)


async def _call_llm(client: AsyncOpenAI, html: str) -> str:
    """Single LLM call: HTML in → Markdown out."""
    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(html=html)},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception as exc:
        logger.error("LLM translation error: %s", exc)
        raise
