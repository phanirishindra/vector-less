"""
"Zero-Null" Multi-Layer Retrieval Orchestrator (ASYNC)
=======================================================
Async refactor:
- Uses AsyncOpenAI client
- All networked retrieval steps are async
- Streaming helpers consume async streams
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from indexer.signposter import ToCEntry

logger = logging.getLogger(__name__)

_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"


def _build_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# Streaming helper: hide <think>…</think> from user-facing output
# ---------------------------------------------------------------------------
async def _astream_hide_think(stream: AsyncIterator[Any]) -> AsyncIterator[str]:
    _OPEN = "<think>"
    _CLOSE = "</think>"
    inside_think = False
    buffer = ""

    def _safe_end_outside(buf: str) -> int:
        for prefix_len in range(len(_OPEN) - 1, 0, -1):
            if buf.endswith(_OPEN[:prefix_len]):
                return len(buf) - prefix_len
        return len(buf)

    def _safe_end_inside(buf: str) -> int:
        for prefix_len in range(len(_CLOSE) - 1, 0, -1):
            if buf.endswith(_CLOSE[:prefix_len]):
                return len(buf) - prefix_len
        return len(buf)

    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta

        while True:
            if not inside_think:
                open_pos = buffer.find(_OPEN)
                if open_pos == -1:
                    safe = _safe_end_outside(buffer)
                    if safe > 0:
                        yield buffer[:safe]
                    buffer = buffer[safe:]
                    break
                if open_pos > 0:
                    yield buffer[:open_pos]
                buffer = buffer[open_pos + len(_OPEN) :]
                inside_think = True
            else:
                close_pos = buffer.find(_CLOSE)
                if close_pos == -1:
                    safe = _safe_end_inside(buffer)
                    buffer = buffer[safe:]
                    break
                buffer = buffer[close_pos + len(_CLOSE) :]
                inside_think = False

    if not inside_think and buffer:
        yield buffer


# ---------------------------------------------------------------------------
# Layer 1 – DeepSieve: query deconstruction
# ---------------------------------------------------------------------------
_DEEPSIEVE_SYSTEM = """\
You are a query analyst. Use a <think>…</think> scratchpad to reason privately.

If the user query is vague or compound, rewrite it as 2-3 distinct, atomic
sub-queries. Output ONLY a JSON array of sub-query strings.

Example output:
["What is the return policy?", "How long does shipping take?"]

If the query is already specific, output a JSON array with just the original:
["<original query>"]
"""


async def deepsieve(query: str, client: AsyncOpenAI) -> list[str]:
    stream = await client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _DEEPSIEVE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=256,
        stream=True,
    )

    visible_tokens: list[str] = []
    async for t in _astream_hide_think(stream):
        visible_tokens.append(t)
    raw = "".join(visible_tokens).strip()

    try:
        sub_queries = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", raw)
        if match:
            try:
                sub_queries = json.loads(match.group(0))
            except json.JSONDecodeError:
                sub_queries = None
        else:
            sub_queries = None

    if isinstance(sub_queries, list) and sub_queries:
        return [str(q) for q in sub_queries]

    logger.warning("DeepSieve JSON parse error; using original query.")
    return [query]


# ---------------------------------------------------------------------------
# Layer 2 – ToC Routing
# ---------------------------------------------------------------------------
_TOC_ROUTER_SYSTEM = """\
You are a retrieval router. Given a list of Dense Signposts (as JSON) and a
set of queries, return a JSON array of the chunk_ids that are most likely to
contain the answer.

Output ONLY a JSON array of chunk_id strings, e.g.:
["abc-123", "def-456"]

If no chunk is relevant, output an empty array: []
"""


async def toc_route(
    sub_queries: list[str], toc: list[ToCEntry], client: AsyncOpenAI
) -> list[str]:
    signpost_index = [{"chunk_id": e.chunk_id, "dense_signpost": e.dense_signpost} for e in toc]
    user_payload = json.dumps({"queries": sub_queries, "signposts": signpost_index}, ensure_ascii=False)

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _TOC_ROUTER_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = (response.choices[0].message.content or "").strip()

        try:
            chunk_ids = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*?\]", raw)
            if not match:
                logger.warning("ToC router returned non-JSON output: %r", raw[:300])
                return []
            try:
                chunk_ids = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("ToC router JSON extraction failed: %r", raw[:300])
                return []

        if isinstance(chunk_ids, list):
            return [str(c) for c in chunk_ids]

        logger.warning("ToC router output was not a list: %r", raw[:300])
        return []
    except Exception as exc:
        logger.warning("ToC routing error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Layer 3 – Iterative Exploration & Synthesis
# ---------------------------------------------------------------------------
_FACT_EXTRACT_SYSTEM = """\
You are a precise fact extractor. Given a Markdown chunk and a question,
extract only the facts from the chunk that are directly relevant to the
question. Output as concise bullet points. Do not fabricate.

If the LLM needs broader context, output exactly:
{"action": "explore_parent", "target": "<heading or chapter>"}
"""

_SYNTHESIS_SYSTEM = """\
You are a synthesis engine. Combine the extracted facts below into a single,
coherent, well-structured answer. Be concise yet complete. Do not invent.
"""


async def iterative_explore(
    sub_queries: list[str],
    chunk_ids: list[str],
    toc: list[ToCEntry],
    client: AsyncOpenAI,
) -> AsyncIterator[str]:
    chunk_map = {e.chunk_id: e for e in toc}
    extracted_facts: list[str] = []

    for cid in chunk_ids:
        entry = chunk_map.get(cid)
        if entry is None:
            continue
        for query in sub_queries:
            facts = await _extract_facts(query, entry, toc, client)
            if facts:
                extracted_facts.append(f"**[{cid}]** {facts}")

    if not extracted_facts:
        return

    combined = "\n\n".join(extracted_facts)
    async for tok in _synthesise(sub_queries, combined, client):
        yield tok


async def _extract_facts(
    query: str,
    entry: ToCEntry,
    toc: list[ToCEntry],
    client: AsyncOpenAI,
    depth: int = 0,
) -> str:
    if depth > 2:
        return ""

    try:
        response = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _FACT_EXTRACT_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nMarkdown chunk:\n{entry.raw_markdown}"},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = (response.choices[0].message.content or "").strip()

        if raw.startswith("{"):
            try:
                action = json.loads(raw)
                if action.get("action") == "explore_parent":
                    target_heading = action.get("target", "")
                    parent = _find_parent(target_heading, toc)
                    if parent and parent.chunk_id != entry.chunk_id:
                        return await _extract_facts(query, parent, toc, client, depth + 1)
            except json.JSONDecodeError:
                pass

        return raw
    except Exception as exc:
        logger.error("Fact extraction error: %s", exc)
        return ""


def _find_parent(heading: str, toc: list[ToCEntry]) -> ToCEntry | None:
    heading_lower = heading.lower()
    for entry in toc:
        if heading_lower in entry.raw_markdown[:200].lower():
            return entry
    return None


async def _synthesise(
    queries: list[str], facts: str, client: AsyncOpenAI
) -> AsyncIterator[str]:
    try:
        stream = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": f"Questions:\n{chr(10).join(queries)}\n\nExtracted facts:\n{facts}"},
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        async for tok in _astream_hide_think(stream):
            yield tok
    except Exception as exc:
        logger.error("Synthesis error: %s", exc)
        yield facts


# ---------------------------------------------------------------------------
# Layer 4 – BM25 Fallback
# ---------------------------------------------------------------------------
_BM25_GROUNDING_SYSTEM = """\
You are a strict question-answering assistant. Answer the question using ONLY
the provided text excerpt. Do not add external knowledge. If the excerpt does
not contain the answer, say "I could not find that information in the indexed
content."
"""


async def bm25_fallback(query: str, toc: list[ToCEntry], client: AsyncOpenAI) -> AsyncIterator[str]:
    corpus_texts = [f"{e.first_sentence} {e.last_sentence} {e.raw_markdown}" for e in toc]
    tokenised = [_tokenise(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenised)

    query_tokens = _tokenise(query)
    scores = bm25.get_scores(query_tokens)
    best_idx = int(scores.argmax())
    best_entry = toc[best_idx]

    try:
        stream = await client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _BM25_GROUNDING_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nText excerpt:\n{best_entry.raw_markdown}"},
            ],
            temperature=0.1,
            max_tokens=512,
            stream=True,
        )
        async for tok in _astream_hide_think(stream):
            yield tok
    except Exception as exc:
        logger.error("BM25 LLM call error: %s", exc)
        yield best_entry.raw_markdown[:1000]


def _tokenise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
async def retrieve(query: str, toc: list[ToCEntry]) -> AsyncIterator[str]:
    client = _build_client()

    sub_queries = await deepsieve(query, client)
    chunk_ids = await toc_route(sub_queries, toc, client)

    if chunk_ids:
        yielded_any = False
        async for chunk in iterative_explore(sub_queries, chunk_ids, toc, client):
            if chunk:
                yield chunk
                yielded_any = True
        if yielded_any:
            return

    async for chunk in bm25_fallback(query, toc, client):
        yield chunk
