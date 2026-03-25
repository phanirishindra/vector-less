"""
"Zero-Null" Multi-Layer Retrieval Orchestrator
===============================================
Executes a cascading fallback strategy to guarantee an answer without OOM errors:

  Layer 1 – DeepSieve / Query Deconstruction
      LLM uses a <think> scratchpad.  Streaming output intercepts and hides
      <think>…</think> blocks so the user never sees them.

  Layer 2 – ToC Routing
      LLM receives all Dense Signposts and returns relevant chunk_ids.

  Layer 3 – Iterative Exploration & Synthesis
      - Multi-Path: loop through chunks, extract facts, run a synthesis call.
      - MCTS-lite: LLM can request parent context via {"action":"explore_parent",…}.

  Layer 4 – BM25 Fallback
      If routing fails completely, rank_bm25 searches first/last sentences and
      raw_markdown; the top hit is injected into the LLM with strict grounding.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Generator, Iterator

from openai import OpenAI
from rank_bm25 import BM25Okapi

from indexer.signposter import ToCEntry

logger = logging.getLogger(__name__)

_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"
_LOCAL_API_KEY = "sk-local"
_MODEL = "qwen2.5"


def _build_client() -> OpenAI:
    return OpenAI(base_url=_LOCAL_BASE_URL, api_key=_LOCAL_API_KEY)


# ---------------------------------------------------------------------------
# Streaming helper: hide <think>…</think> from user-facing output
# ---------------------------------------------------------------------------
def _stream_hide_think(stream: Iterator) -> Generator[str, None, None]:
    """
    Consume a streaming completion and yield tokens visible to the user,
    suppressing any content inside ``<think>…</think>`` tags.

    Handles tags split across chunk boundaries correctly by holding back any
    partial tag prefix until the next chunk confirms whether it is a real tag.
    """
    _OPEN = "<think>"
    _CLOSE = "</think>"
    inside_think = False
    buffer = ""

    def _safe_end_outside(buf: str) -> int:
        """Return the index up to which it is safe to yield when outside a think block.

        If ``buf`` ends with a partial prefix of ``<think>`` (e.g. ``<th``), the
        prefix is held back so it is not emitted prematurely — the next chunk may
        complete the tag and trigger suppression.
        """
        for prefix_len in range(len(_OPEN) - 1, 0, -1):
            if buf.endswith(_OPEN[:prefix_len]):
                return len(buf) - prefix_len
        return len(buf)

    def _safe_end_inside(buf: str) -> int:
        """Return the index up to which it is safe to discard when inside a think block.

        If ``buf`` ends with a partial prefix of ``</think>`` (e.g. ``</thi``),
        that prefix is kept so the next chunk can complete the closing tag and
        exit think mode correctly.
        """
        for prefix_len in range(len(_CLOSE) - 1, 0, -1):
            if buf.endswith(_CLOSE[:prefix_len]):
                return len(buf) - prefix_len
        return len(buf)

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta

        while True:
            if not inside_think:
                open_pos = buffer.find(_OPEN)
                if open_pos == -1:
                    # No complete open tag found — yield up to any partial tag prefix
                    safe = _safe_end_outside(buffer)
                    yield buffer[:safe]
                    buffer = buffer[safe:]
                    break
                # Yield text before the opening tag, then enter think mode
                yield buffer[:open_pos]
                buffer = buffer[open_pos + len(_OPEN):]
                inside_think = True
            else:
                close_pos = buffer.find(_CLOSE)
                if close_pos == -1:
                    # No complete close tag yet — discard up to any partial tag prefix
                    safe = _safe_end_inside(buffer)
                    buffer = buffer[safe:]
                    break
                # Skip everything up to and including </think>
                buffer = buffer[close_pos + len(_CLOSE):]
                inside_think = False

    # Flush any remaining safe content after the stream ends
    if not inside_think and buffer:
        yield buffer


# ---------------------------------------------------------------------------
# Layer 1 – DeepSieve: query deconstruction
# ---------------------------------------------------------------------------
_DEEPSIEVE_SYSTEM = """\
You are a query analyst.  Use a <think>…</think> scratchpad to reason privately.

If the user query is vague or compound, rewrite it as 2-3 distinct, atomic
sub-queries.  Output ONLY a JSON array of sub-query strings.

Example output:
["What is the return policy?", "How long does shipping take?"]

If the query is already specific, output a JSON array with just the original:
["<original query>"]
"""


def deepsieve(query: str, client: OpenAI) -> list[str]:
    """Layer 1: Deconstruct the user query, hide <think> scratchpad."""
    stream = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _DEEPSIEVE_SYSTEM},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=256,
        stream=True,
    )

    visible_tokens: list[str] = list(_stream_hide_think(stream))
    raw = "".join(visible_tokens).strip()

    try:
        sub_queries: list[str] = json.loads(raw)
        if isinstance(sub_queries, list) and sub_queries:
            logger.info("DeepSieve produced %d sub-quer(ies).", len(sub_queries))
            return [str(q) for q in sub_queries]
    except json.JSONDecodeError:
        logger.warning("DeepSieve JSON parse error; using original query.")

    return [query]


# ---------------------------------------------------------------------------
# Layer 2 – ToC Routing
# ---------------------------------------------------------------------------
_TOC_ROUTER_SYSTEM = """\
You are a retrieval router.  Given a list of Dense Signposts (as JSON) and a
set of queries, return a JSON array of the chunk_ids that are most likely to
contain the answer.

Output ONLY a JSON array of chunk_id strings, e.g.:
["abc-123", "def-456"]

If no chunk is relevant, output an empty array: []
"""


def toc_route(
    sub_queries: list[str], toc: list[ToCEntry], client: OpenAI
) -> list[str]:
    """Layer 2: Ask the LLM to pick relevant chunk_ids from the ToC."""
    signpost_index = [
        {"chunk_id": e.chunk_id, "dense_signpost": e.dense_signpost} for e in toc
    ]
    user_payload = json.dumps(
        {"queries": sub_queries, "signposts": signpost_index}, ensure_ascii=False
    )

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _TOC_ROUTER_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = (response.choices[0].message.content or "").strip()

        # 1) Fast path: exact JSON array
        try:
            chunk_ids = json.loads(raw)
        except json.JSONDecodeError:
            # 2) Fallback: extract first JSON array from mixed/prose/markdown output
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
            logger.info("ToC router selected %d chunk(s).", len(chunk_ids))
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
You are a precise fact extractor.  Given a Markdown chunk and a question,
extract only the facts from the chunk that are directly relevant to the
question.  Output as concise bullet points.  Do not fabricate.

If the LLM needs broader context, output exactly:
{"action": "explore_parent", "target": "<heading or chapter>"}
"""

_SYNTHESIS_SYSTEM = """\
You are a synthesis engine.  Combine the extracted facts below into a single,
coherent, well-structured answer.  Be concise yet complete.  Do not invent.
"""


def iterative_explore(
    sub_queries: list[str],
    chunk_ids: list[str],
    toc: list[ToCEntry],
    client: OpenAI,
) -> Generator[str, None, None]:
    """Layer 3: Extract facts from each chosen chunk, then synthesise."""
    chunk_map = {e.chunk_id: e for e in toc}
    extracted_facts: list[str] = []

    for cid in chunk_ids:
        entry = chunk_map.get(cid)
        if entry is None:
            logger.warning("chunk_id %s not found in ToC.", cid)
            continue

        for query in sub_queries:
            facts = _extract_facts(query, entry, toc, client)
            if facts:
                extracted_facts.append(f"**[{cid}]** {facts}")

    if not extracted_facts:
        return

    # Multi-path: synthesise all extracted facts
    combined = "\n\n".join(extracted_facts)
    yield from _synthesise(sub_queries, combined, client)


def _extract_facts(
    query: str,
    entry: ToCEntry,
    toc: list[ToCEntry],
    client: OpenAI,
    depth: int = 0,
) -> str:
    """Extract facts from a single chunk; handle MCTS-lite explore_parent action."""
    if depth > 2:
        # Prevent infinite recursion
        return ""

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _FACT_EXTRACT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nMarkdown chunk:\n{entry.raw_markdown}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = (response.choices[0].message.content or "").strip()

        # MCTS-lite: check if LLM wants to explore parent
        if raw.startswith("{"):
            try:
                action = json.loads(raw)
                if action.get("action") == "explore_parent":
                    target_heading = action.get("target", "")
                    parent = _find_parent(target_heading, toc)
                    if parent and parent.chunk_id != entry.chunk_id:
                        logger.info(
                            "MCTS-lite: exploring parent %s", parent.chunk_id
                        )
                        return _extract_facts(query, parent, toc, client, depth + 1)
            except json.JSONDecodeError:
                pass

        return raw
    except Exception as exc:
        logger.error("Fact extraction error: %s", exc)
        return ""


def _find_parent(heading: str, toc: list[ToCEntry]) -> ToCEntry | None:
    """Heuristically locate the chunk whose heading matches the requested parent."""
    heading_lower = heading.lower()
    for entry in toc:
        if heading_lower in entry.raw_markdown[:200].lower():
            return entry
    return None


def _synthesise(
    queries: list[str], facts: str, client: OpenAI
) -> Generator[str, None, None]:
    """Final synthesis call — streams output, filtering out <think> blocks."""
    try:
        stream = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Questions:\n{chr(10).join(queries)}\n\n"
                        f"Extracted facts:\n{facts}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        yield from _stream_hide_think(stream)
    except Exception as exc:
        logger.error("Synthesis error: %s", exc)
        yield facts  # Yield raw facts as fallback


# ---------------------------------------------------------------------------
# Layer 4 – BM25 Fallback
# ---------------------------------------------------------------------------
_BM25_GROUNDING_SYSTEM = """\
You are a strict question-answering assistant.  Answer the question using ONLY
the provided text excerpt.  Do not add external knowledge.  If the excerpt does
not contain the answer, say "I could not find that information in the indexed
content."
"""


def bm25_fallback(query: str, toc: list[ToCEntry], client: OpenAI) -> Generator[str, None, None]:
    """Layer 4: BM25 lexical search → top hit → LLM grounded answer (streamed)."""
    logger.info("Layer 4 triggered: BM25 fallback for query: %s", query)

    # Build corpus: combine first_sentence + last_sentence + raw_markdown for each entry
    corpus_texts = [
        f"{e.first_sentence} {e.last_sentence} {e.raw_markdown}" for e in toc
    ]
    tokenised = [_tokenise(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenised)

    query_tokens = _tokenise(query)
    scores = bm25.get_scores(query_tokens)
    best_idx = int(scores.argmax())
    best_entry = toc[best_idx]

    logger.info("BM25 top hit: chunk_id=%s (score=%.4f)", best_entry.chunk_id, scores[best_idx])

    try:
        stream = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _BM25_GROUNDING_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\nText excerpt:\n{best_entry.raw_markdown}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
            stream=True,
        )
        yield from _stream_hide_think(stream)
    except Exception as exc:
        logger.error("BM25 LLM call error: %s", exc)
        yield best_entry.raw_markdown[:1000]


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenisation for BM25."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def retrieve(query: str, toc: list[ToCEntry]) -> Generator[str, None, None]:
    """
    Full "Zero-Null" multi-layer retrieval.

    Yields answer tokens so callers can stream output to the user.
    The <think> scratchpad is intercepted and never yielded.
    """
    client = _build_client()

    # Layer 1: DeepSieve
    sub_queries = deepsieve(query, client)

    # Layer 2: ToC Routing
    chunk_ids = toc_route(sub_queries, toc, client)

    # Layer 3: Iterative exploration (if we have chunk hits)
    if chunk_ids:
        yielded_any = False
        for chunk in iterative_explore(sub_queries, chunk_ids, toc, client):
            if chunk:
                yield chunk
                yielded_any = True
        if yielded_any:
            return

    # Layer 4: BM25 fallback
    yield from bm25_fallback(query, toc, client)
