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

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local LLM client (pinned to llama.cpp / any OpenAI-compatible local server)
# ---------------------------------------------------------------------------
_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"          # model name as registered in the local server


def _build_client() -> OpenAI:
    return OpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are an HTML-to-Markdown converter. You receive raw HTML and output only the \
converted Markdown — nothing else.

Output rules:
1. Output Markdown only. Start your response with the first Markdown character. \
Do not write any introduction, explanation, or closing remark.
2. Use # for the page title, ## for top-level sections, ### for sub-sections.
3. Preserve tables as GitHub-Flavored Markdown pipe tables with a separator row \
containing |---|.
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
# Public API
# ---------------------------------------------------------------------------
def translate_html_to_markdown(html_chunks: Sequence[str]) -> str:
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
        markdown = _call_llm(client, chunk)
        parts.append(markdown)

    return "\n\n".join(parts)


def _call_llm(client: OpenAI, html: str) -> str:
    """Single LLM call: HTML in → Markdown out."""
    try:
        response = client.chat.completions.create(
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
