"""Knowledge chat — a chat that answers ONLY from the user's knowledge base.

Deliberately NOT a node in the main orchestration graph. The main assistant treated the
knowledge base as one external source among twenty-odd tools, and it served that badly:
a forced search on nearly every turn, a 4 x 600-character context budget, one search per
turn, and guards bolted on so REIT, brief and attachment questions weren't answered from
the wrong documents (platform doc 29). This module is the replacement: a small tool loop
whose whole world is the user's documents.

    question ─► Gemini with 3 tools ─► ≤ MAX_ROUNDS rounds of tool calls ─► cited answer
                 search_knowledge · read_document · list_documents

Rules it enforces:

* **Knowledge base only.** The system prompt forbids general knowledge; when retrieval
  finds nothing relevant the answer says so. Passages below KB_CHAT_MIN_SCORE never
  reach the model — the cross-encoder separates relevant (+2..+9) from unrelated
  (about -10) cleanly, and 0.0 is its decision boundary.
* **Every claim cites a numbered passage** ``[n]``. The stream ends with a ``sources``
  event listing only the passages the answer actually cited.
* **Memory within a conversation only.** Nothing is written anywhere. The UI sends the
  prior turns, and each assistant turn carries the documents it cited, so "what else did
  that report say?" resolves to the same document on the next turn.

SSE events (same framing as /intent/stream): ``{"token"}``, ``{"tool_call": {"name",
"args"}}``, ``{"sources": [...]}``, and ``{"status"}`` last.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types

from models import KnowledgeChatRequest
from services import kb as kbstore
from services import llm_ledger

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


MODEL = os.environ.get("KB_CHAT_MODEL", "gemini-2.5-flash")
# Rounds in which the model may call tools. One more, tool-less, round follows if it is
# still calling tools, so a turn always ends in an answer.
MAX_ROUNDS = _env_int("KB_CHAT_MAX_ROUNDS", 4)
MAX_CALLS_PER_ROUND = 6
SEARCH_TOP_K = _env_int("KB_CHAT_SEARCH_TOP_K", 8)
PASSAGE_CHARS = _env_int("KB_CHAT_PASSAGE_CHARS", 1500)
READ_MAX_CHUNKS = _env_int("KB_CHAT_READ_MAX_CHUNKS", 12)
READ_PASSAGE_CHARS = _env_int("KB_CHAT_READ_PASSAGE_CHARS", 2000)
# Total characters of retrieved text one turn may put in front of the model.
CONTEXT_CHARS = _env_int("KB_CHAT_CONTEXT_CHARS", 40_000)
MIN_SCORE = _env_float("KB_CHAT_MIN_SCORE", 0.0)
# Passages kept from a search scoped to named documents when none clears MIN_SCORE.
SCOPED_WEAK_MAX = 6
THINKING_BUDGET = _env_int("KB_CHAT_THINKING_BUDGET", 0)
HISTORY_TURNS = _env_int("KB_CHAT_HISTORY_TURNS", 20)
# A tool-less answer longer than this, on a turn with conversation history, is treated as
# answered-from-memory and sent back to search first (see _ungrounded). Short enough to
# catch a one-sentence "it doesn't mention that" (400 let exactly that through in the
# eval, 2026-09-17), long enough to let "You're welcome" and "Glad that helped" pass.
MEMORY_ANSWER_CHARS = 80
LIBRARY_TITLES = 200
EXCERPT_CHARS = 400

TOOL_NAMES = ("search_knowledge", "read_document", "list_documents")

_client_cache: tuple[str, genai.Client] | None = None


def _client() -> genai.Client | None:
    """Gemini client from GEMINI_API_KEY (memoized per key); None when unset."""
    global _client_cache
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    if _client_cache is None or _client_cache[0] != key:
        _client_cache = (key, genai.Client(api_key=key))
    return _client_cache[1]


# --- prompt -----------------------------------------------------------------------------


def _now_line(tzname: str | None) -> str:
    name = (tzname or os.environ.get("DEFAULT_TIMEZONE") or "UTC").strip()
    try:
        tz = ZoneInfo(name)
    except Exception:
        name, tz = "UTC", ZoneInfo("UTC")
    now = datetime.now(tz)
    return f"Today is {now.strftime('%A, %Y-%m-%d')} ({name})."


def _library_block(docs: list[dict] | None) -> str:
    if docs is None:
        return "The document list could not be loaded; use list_documents or search_knowledge.\n"
    if not docs:
        return "The knowledge base is EMPTY — there are no documents to answer from.\n"
    lines = [f"The knowledge base holds {len(docs)} document(s), newest first:"]
    for d in docs[:LIBRARY_TITLES]:
        added = str(d.get("created_at") or "")[:10]
        lines.append(f"- {(d.get('title') or 'Untitled').strip()}" + (f" (added {added})" if added else ""))
    if len(docs) > LIBRARY_TITLES:
        lines.append(f"- … and {len(docs) - LIBRARY_TITLES} more (use list_documents)")
    return "\n".join(lines) + "\n"


def _discussed_block(req: KnowledgeChatRequest | None) -> str:
    """Documents earlier answers in this conversation cited, most recent last.

    Kept in the system prompt, NOT appended to the assistant turns: annotated turns
    taught the model the format, and it began ending its own answers with
    "(Documents cited in this answer: …)" (seen in the browser conversations,
    2026-09-17)."""
    if req is None:
        return ""
    titles: list[str] = []
    for t in req.history[-HISTORY_TURNS:]:
        for src in t.sources if t.role == "assistant" else []:
            if src.title in titles:
                titles.remove(src.title)
            titles.append(src.title)
    if not titles:
        return ""
    return (
        "Documents earlier answers in this conversation cited (most recent last) — "
        "\"that report\", \"it\", \"the other one\" refer to these:\n"
        + "\n".join(f"- {t}" for t in titles[-10:])
        + "\n\n"
    )


def system_prompt(docs: list[dict] | None, tzname: str | None, req: KnowledgeChatRequest | None = None) -> str:
    return (
        "You are Knowledge chat: you answer questions using ONLY the user's own knowledge "
        "base — the research reports and documents they saved. You have no other source.\n\n"
        f"{_now_line(tzname)}\n"
        f"{_library_block(docs)}\n"
        f"{_discussed_block(req)}"
        "How to answer:\n"
        "1. Use the tools to read the knowledge base before answering any substantive "
        "question. search_knowledge finds passages for a question; read_document reads one "
        "named document for a summary or overview (pass `focus` for one topic within it). "
        "Call several tools in the same step when a question needs several documents — for "
        "example to compare two issues, read or search each one.\n"
        "2. Every factual statement in your answer must come from a numbered passage a tool "
        "returned IN THIS TURN, and must cite it like [3] or [2, 5]. Earlier answers in the "
        "conversation show what was discussed and which documents were used — they are not "
        "evidence. For a follow-up, search again (scoped to those documents when the user "
        "means them).\n"
        "3. Never use general knowledge, and never fill a gap with something plausible. If "
        "the passages do not answer the question, say plainly that the knowledge base does "
        "not cover it — and, when useful, which documents come closest. A statement that "
        "something is NOT covered carries no citation: cite only passages that support what "
        "you say.\n"
        "4. Choosing documents: when the user names a document or a kind of publication "
        "(\"the Agency MBS weekly\", \"the data center ABS note\"), find the title in the "
        "list above that matches it and pass THAT title to the tools — a search across the "
        "whole knowledge base returns whatever is nearest, which is often a different "
        "publication. \"The latest\" or \"most recent\" means the newest matching title (by "
        "the date in the title, else the date added). A recurring publication has many "
        "issues whose titles differ only by date; the date identifies the issue. \"That "
        "report\" and \"it\" refer to the documents cited earlier (listed above, when there "
        "are any). If a reference is genuinely ambiguous, ask which one they mean.\n"
        "5. Comparing documents: call read_document with `focus` set to the topic, once PER "
        "document and all in the same step, so one document's passages cannot crowd out the "
        "other's — read_document also returns each document's summary, which states its main "
        "points on the topic. Cover each document the question names, and weigh what each "
        "says about the topic itself rather than whichever tables happened to match.\n"
        "6. Before saying a document does not cover something, try once more — "
        "read_document with a `focus`, or search_knowledge with other wording scoped to that "
        "document. When several passages or documents bear on the question, use them all.\n"
        "7. Greetings, thanks and questions about what you can do need no tools — reply "
        "briefly.\n\n"
        "Format: Markdown. Lead with the answer. Bullets for several distinct points, prose "
        "otherwise. Keep citations right after the sentence they support. Do not add a "
        "sources list — the app shows the cited passages. No filler."
    )


def _tools() -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="search_knowledge",
                    description=(
                        "Search the knowledge base for passages relevant to a question. "
                        "Returns numbered passages with their document title and date. "
                        "Pass `documents` to search only inside specific documents."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to look for, phrased as a question or key terms.",
                            },
                            "documents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Optional. Titles (or distinctive parts of titles, "
                                    "including the date) of documents to restrict the search to."
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                ),
                types.FunctionDeclaration(
                    name="read_document",
                    description=(
                        "Read ONE document: its summary plus numbered passages sampled from "
                        "start to finish. Use for 'summarize X', 'what are the highlights of "
                        "X'. With `focus`, returns the passages of X most relevant to that topic."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "document": {
                                "type": "string",
                                "description": "The document's title or a distinctive part of it, including any date.",
                            },
                            "focus": {
                                "type": "string",
                                "description": "Optional topic within the document.",
                            },
                        },
                        "required": ["document"],
                    },
                ),
                types.FunctionDeclaration(
                    name="list_documents",
                    description="List every document in the knowledge base with its summary and date added.",
                    parameters_json_schema={"type": "object", "properties": {}},
                ),
            ]
        )
    ]


# --- retrieval bookkeeping ------------------------------------------------------------------


@dataclass
class Passage:
    n: int
    key: str
    document_id: str | None
    title: str
    chunk_index: int | None
    text: str
    weak: bool = False
    # What the reader is shown under the answer. For a search hit this is the matched
    # ~200-token passage, not the start of its parent block — the parent (what the model
    # reads) often opens with page furniture, which made the excerpt look unrelated to
    # the claim it supports (seen in the browser test, 2026-09-17).
    excerpt: str = ""


@dataclass
class TurnContext:
    """Passages retrieved during one turn, numbered in the order first seen."""

    passages: list[Passage] = field(default_factory=list)
    by_key: dict[str, Passage] = field(default_factory=dict)
    chars: int = 0
    # The library as listed at the start of the turn (title -> summary), so a scoped
    # search can show the model what each named document says overall.
    summaries: dict[str, str] = field(default_factory=dict)

    def add(self, key: str, document_id: str | None, title: str, chunk_index: int | None, text: str,
            excerpt: str | None = None, weak: bool = False) -> Passage | None:
        if key in self.by_key:
            return self.by_key[key]
        if self.chars + len(text) > CONTEXT_CHARS:
            return None
        p = Passage(len(self.passages) + 1, key, document_id, title, chunk_index, text, weak, excerpt or text)
        self.passages.append(p)
        self.by_key[key] = p
        self.chars += len(text)
        return p


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _not_found_text(payload: dict, what: str) -> str:
    alts: list[str] = []
    for u in payload.get("unresolved") or []:
        alts += [a.get("title") for a in u.get("alternatives") or [] if a.get("title")]
    for a in payload.get("alternatives") or []:
        if a.get("title"):
            alts.append(a["title"])
    msg = f"No document in the knowledge base matches {what}."
    if alts:
        msg += " Closest titles: " + "; ".join(dict.fromkeys(alts)) + "."
    return msg + " Check the document list and try again with a title from it."


async def _tool_search(ctx: TurnContext, user_id: str, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "error: query is required"
    documents = [str(d) for d in (args.get("documents") or []) if str(d).strip()]
    try:
        if len(documents) > 1:
            # One search PER named document, interleaved. A single search over several
            # documents let one crowd out the rest: asked to compare the Aug 28 and Sep 11
            # issues on CMBS, 7 of 8 passages came from Sep 11 — and the model kept issuing
            # the combined search even when told not to (2026-09-17), so the tool does it.
            per = max(3, SEARCH_TOP_K // len(documents))
            results = await asyncio.gather(*(
                kbstore.search(user_id, query, top_k=per, document_titles=[d]) for d in documents
            ))
            lists = [r.get("chunks") or [] for r in results]
            merged: list[dict] = []
            for i in range(max(len(x) for x in lists)):
                merged += [x[i] for x in lists if i < len(x)]
            payload = {"chunks": merged}
        else:
            payload = await kbstore.search(user_id, query, top_k=SEARCH_TOP_K, document_titles=documents or None)
    except kbstore.KbNotFound as exc:
        return _not_found_text(exc.payload, ", ".join(f'"{d}"' for d in documents) or "that reference")
    def weak(c: dict) -> bool:
        return isinstance(c.get("score"), (int, float)) and float(c["score"]) < MIN_SCORE

    all_chunks = payload.get("chunks") or []
    if documents:
        # Scoped to documents the user named: no relevance floor, same policy as
        # read_document. The floor exists so a corpus-wide search cannot cite unrelated
        # documents; inside a named document its best passages ARE the material, even
        # when the cross-encoder scores the wording low. Measured 2026-09-17: "regulatory
        # changes" scored every passage of the CLO report below 0, including the Basel
        # Endgame / NAIC one, and the answer became "it doesn't mention any".
        chunks = all_chunks[:SCOPED_WEAK_MAX] if all(weak(c) for c in all_chunks) else all_chunks
    else:
        chunks = [c for c in all_chunks if not weak(c)]
    if not chunks:
        return "No passages in the knowledge base are relevant to this query."
    lines: list[str] = []
    if documents:
        # What each named document says overall, so a thin passage set (a ratings table
        # for "CMBS") does not stand in for the document's actual view on the topic.
        for title in dict.fromkeys(c.get("title") for c in chunks if c.get("title")):
            summary = ctx.summaries.get(title)
            if summary:
                lines.append(
                    f"SUMMARY of {title} (not citable — search it with more specific terms "
                    f"to find the passages behind it): {_clip(summary, 400)}"
                )
    full = False
    for c in chunks:
        body = c.get("parent_content") or c.get("content") or ""
        added = ctx.add(
            f"chunk:{c.get('id')}", c.get("document_id"), (c.get("title") or "Untitled").strip(),
            c.get("chunk_index"), _clip(body, PASSAGE_CHARS),
            excerpt=c.get("content") or body, weak=weak(c),
        )
        if added is None:
            full = True
            break
        date = str(c.get("document_created_at") or "")[:10]
        tag = " — weak match, check it actually answers the question" if weak(c) else ""
        lines.append(f"[{added.n}] {added.title}" + (f" (added {date})" if date else "") + f"{tag}\n{added.text}")
    if full:
        lines.append("(Context budget for this turn is full — answer from the passages you have.)")
    return "\n\n".join(lines) if lines else "Context budget for this turn is full — answer from the passages you have."


async def _tool_read(ctx: TurnContext, user_id: str, args: dict) -> str:
    document = str(args.get("document") or "").strip()
    if not document:
        return "error: document is required"
    focus = str(args.get("focus") or "").strip() or None
    try:
        payload = await kbstore.read_document(user_id, document, focus=focus, max_chunks=READ_MAX_CHUNKS)
    except kbstore.KbNotFound as exc:
        return _not_found_text(exc.payload, f'"{document}"')
    doc = payload.get("document") or {}
    title = (doc.get("title") or "Untitled").strip()
    head = [f"DOCUMENT: {title}"]
    if doc.get("created_at"):
        head.append(f"Added: {str(doc['created_at'])[:10]}")
    if doc.get("summary"):
        head.append(f"Summary written when it was added (cite a passage, not this line): {doc['summary'].strip()}")
    lines = ["\n".join(head)]
    full = False
    for c in payload.get("chunks") or []:
        added = ctx.add(
            f"doc:{doc.get('id')}:{c.get('chunk_index')}", doc.get("id"), title,
            c.get("chunk_index"), _clip(c.get("content") or "", READ_PASSAGE_CHARS),
        )
        if added is None:
            full = True
            break
        lines.append(f"[{added.n}] {added.text}")
    if len(lines) == 1:
        lines.append("No readable passages are stored for this document.")
    if full:
        lines.append("(Context budget for this turn is full — answer from the passages you have.)")
    alts = [a.get("title") for a in payload.get("alternatives") or [] if a.get("title")]
    if alts:
        lines.append("Other documents with similar titles: " + "; ".join(alts))
    return "\n\n".join(lines)


async def _tool_list(user_id: str) -> str:
    payload = await kbstore.list_documents(user_id)
    docs = payload.get("documents") or []
    if not docs:
        return "The knowledge base has no documents."
    out = [f"{len(docs)} document(s):"]
    for d in docs:
        line = f"- {(d.get('title') or 'Untitled').strip()} (added {str(d.get('created_at') or '')[:10]})"
        if d.get("summary"):
            line += f": {_clip(d['summary'], 300)}"
        out.append(line)
    return "\n".join(out)


async def run_tool(ctx: TurnContext, user_id: str, name: str, args: dict) -> str:
    """Execute one tool call; never raises (a failure becomes text the model can read)."""
    try:
        if name == "search_knowledge":
            return await _tool_search(ctx, user_id, args)
        if name == "read_document":
            return await _tool_read(ctx, user_id, args)
        if name == "list_documents":
            return await _tool_list(user_id)
        return f"error: unknown tool {name!r}"
    except Exception as exc:  # KB unreachable etc. — tell the model, keep the turn alive
        logger.warning("knowledge tool %s failed: %s", name, type(exc).__name__)
        return f"error: the knowledge base could not be reached ({type(exc).__name__}). Tell the user to try again shortly."


# --- conversation -------------------------------------------------------------------------


def history_contents(req: KnowledgeChatRequest) -> list[types.Content]:
    """Prior turns + the new question as Gemini contents: starts with a user turn and
    roles alternate (consecutive same-role turns are merged). The documents earlier
    answers cited go in the system prompt (_discussed_block), not in these turns."""
    turns: list[tuple[str, str]] = []
    for t in req.history[-HISTORY_TURNS:]:
        role = "user" if t.role == "user" else "model"
        text = t.content.strip()
        if not text:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + text)
        else:
            turns.append((role, text))
    while turns and turns[0][0] != "user":
        turns.pop(0)
    if turns and turns[-1][0] == "user":
        turns[-1] = ("user", turns[-1][1] + "\n\n" + req.text.strip())
    else:
        turns.append(("user", req.text.strip()))
    return [types.Content(role=r, parts=[types.Part.from_text(text=x)]) for r, x in turns]


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sentences that are about the knowledge base itself, or decline — never a claim a
# passage could support. Used only to decide whether citations are shown as sources
# (see cited_sources); the answer text is never changed.
_META = re.compile(
    r"\b(knowledge base|my documents|your documents|cannot answer|can't answer|unable to (answer|find)"
    r"|does not (contain|cover|mention|include|discuss)|doesn't (contain|cover|mention|include|discuss)"
    r"|do not (contain|cover)|not covered|no (relevant )?information)\b",
    re.I,
)
_CONTRAST = re.compile(r"\b(but|however|although|though|whereas)\b", re.I)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def cited_sources(answer: str, ctx: TurnContext) -> list[dict]:
    """The passages the answer actually cited, in citation-number order."""
    wanted: set[int] = set()
    for m in _CITE.finditer(answer):
        wanted.update(int(x) for x in m.group(1).split(","))
    cited = [p for p in ctx.passages if p.n in wanted]
    # Citations attached only to statements ABOUT the knowledge base ("my knowledge base
    # contains research reports, not sports results [1]") or to a refusal are not
    # evidence, and showing that passage reads as support for "not covered". Seen in 1-2
    # of 9 out-of-scope eval answers: the model searched, found a numbers table that
    # happened to contain "2024", declined correctly, and cited it anyway.
    cited_sentences = [x for x in _SENTENCES.split(answer.strip()) if _CITE.search(x)]
    if cited_sentences and all(_META.search(x) and not _CONTRAST.search(x) for x in cited_sentences):
        return []
    return [
        {
            "n": p.n,
            "document_id": p.document_id,
            "title": p.title,
            "chunk_index": p.chunk_index,
            "excerpt": _clip(p.excerpt, EXCERPT_CHARS),
        }
        for p in cited
    ]


_GROUND_NUDGE = (
    "Before answering, search or read the knowledge base in this turn. Earlier answers in "
    "the conversation are not evidence, and their citation numbers do not refer to anything "
    "you have retrieved now."
)


def _ungrounded(answer: str, req: KnowledgeChatRequest) -> bool:
    """Did the model answer a substantive turn without retrieving anything?

    Observed in the live eval (2026-09-17): asked a follow-up, the model restated the
    previous answer and reused its "[1]" — a citation pointing at no passage retrieved in
    this turn. A tool-less answer is fine for "thanks" or "that isn't covered"; it is not
    fine when it cites passages, or when it is a long answer to a follow-up.
    """
    if _CITE.search(answer):
        return True
    return bool(req.history) and len(answer) > MEMORY_ANSWER_CHARS


def _config(system: str, tools_enabled: bool, force_tools: bool = False) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system,
        tools=_tools(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=("ANY" if force_tools else "AUTO") if tools_enabled else "NONE"
            )
        ),
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        temperature=0.2,
    )


async def run(req: KnowledgeChatRequest, user_id: str) -> AsyncIterator[dict]:
    """Run one Knowledge-chat turn, yielding SSE event dicts. Never raises."""
    client = _client()
    if client is None:
        yield {"token": "Knowledge chat is not configured (no model key)."}
        yield {"status": "error"}
        return

    try:
        docs: list[dict] | None = (await kbstore.list_documents(user_id)).get("documents") or []
    except Exception as exc:
        logger.warning("knowledge chat: document list unavailable: %s", type(exc).__name__)
        docs = None

    system = system_prompt(docs, req.timezone, req)
    contents = history_contents(req)
    ctx = TurnContext(summaries={
        (d.get("title") or "").strip(): d.get("summary") or "" for d in (docs or []) if d.get("title")
    })
    group = llm_ledger.new_group()
    answer_parts: list[str] = []
    tools_called = False
    force_tools = False

    for round_no in range(1, MAX_ROUNDS + 2):
        tools_enabled = round_no <= MAX_ROUNDS
        parts: list[types.Part] = []
        calls: list[types.FunctionCall] = []
        # Until a tool has run this turn, hold the text back: a tool-less answer may turn
        # out to be answered-from-memory (_ungrounded) and must then never reach the user.
        # Once retrieval has happened, text streams as it arrives.
        buffering = not tools_called and tools_enabled
        held: list[str] = []
        streamed_this_round = False
        try:
            with llm_ledger.attempt(
                "gemini", MODEL, "heartbeat.knowledge_chat",
                run_ref=f"chat:{req.chat_id}" if req.chat_id else None, group=group, attempt=round_no,
            ) as record:
                last_meta = None
                stream = await client.aio.models.generate_content_stream(
                    model=MODEL, contents=contents, config=_config(system, tools_enabled, force_tools)
                )
                async for chunk in stream:
                    if getattr(chunk, "usage_metadata", None) is not None:
                        last_meta = chunk.usage_metadata
                    for cand in chunk.candidates or []:
                        for part in (cand.content.parts if cand.content and cand.content.parts else []):
                            parts.append(part)
                            if part.function_call is not None:
                                calls.append(part.function_call)
                            elif part.text and not part.thought:
                                if buffering:
                                    held.append(part.text)
                                else:
                                    answer_parts.append(part.text)
                                    streamed_this_round = True
                                    yield {"token": part.text}
                record.ok(last_meta)
        except Exception as exc:
            logger.warning("knowledge chat model call failed: %s", type(exc).__name__)
            if answer_parts:
                break  # keep the partial answer the reader already has
            yield {"token": "The knowledge chat couldn't reach the model just now. Please try again in a moment."}
            yield {"status": "error"}
            return

        held_text = "".join(held)
        if not calls:
            if buffering and not force_tools and _ungrounded(held_text, req):
                # Answered from memory: discard it and make the next round retrieve.
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=held_text)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=_GROUND_NUDGE)]))
                force_tools = True
                continue
            if held_text:
                answer_parts.append(held_text)
                yield {"token": held_text}
            break
        if not tools_enabled:
            break

        force_tools = False
        tools_called = True
        calls = calls[:MAX_CALLS_PER_ROUND]
        contents.append(types.Content(role="model", parts=parts))
        for call in calls:
            yield {"tool_call": {"name": call.name, "args": dict(call.args or {})}}
        results = await asyncio.gather(*(run_tool(ctx, user_id, c.name or "", dict(c.args or {})) for c in calls))
        contents.append(types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(name=c.name or "", response={"result": r})
                for c, r in zip(calls, results)
            ],
        ))
        if streamed_this_round:
            answer_parts.append("\n\n")
            yield {"token": "\n\n"}

    answer = "".join(answer_parts).strip()
    if not answer:
        text = "I couldn't put together an answer from the knowledge base. Try rephrasing the question."
        yield {"token": text}
        yield {"sources": []}
        yield {"status": "completed"}
        return

    yield {"sources": cited_sources(answer, ctx)}
    yield {"status": "completed"}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"
