"""Live eval of the Knowledge chat agent, run INSIDE the deployed core-heartbeat container:

    docker exec -e KB_EVAL_USER=<user uuid> core-heartbeat python scripts/kb_chat_eval.py

Exercises the real loop (knowledge_chat.run) against real Gemini and the real knowledge
base, as that user. It does not go through HTTP auth — the route is covered by unit tests
and core-chat's browser test. Costs a few cents of Gemini; every call is in the ledger
under operation heartbeat.knowledge_chat.

Pass criteria (plan gate, 2026-09-17):
  * every answered question cites at least one passage, from an expected document;
  * every out-of-scope question cites nothing;
  * every follow-up stays on the document the first answer used;
  * median time to first token < 8 s.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import knowledge_chat as kc  # noqa: E402
from models import KnowledgeChatRequest  # noqa: E402

USER = os.environ["KB_EVAL_USER"]
TZ = "America/Chicago"

# (question, substrings any cited document title may contain)
ANSWERABLE = [
    ("How did 30-year conventional prepay speeds change in June?", ["June Prepay Speeds"]),
    ("Why does Morgan Stanley prefer colocation over hyperscaler data center ABS?", ["Data Center ABS"]),
    ("Summarize the latest Securitized Products Research issue.", ["September 11"]),
    ("What did the August 28 securitized products report say about multifamily CMBS?", ["August 28"]),
    ("Are borrowers whose FICO 10T score declines more likely to go delinquent?", ["Trended Credit Scores"]),
    ("Which securities did banks add most in 2Q26?", ["Bank Flows"]),
]

OUT_OF_SCOPE = [
    "Who won the 2024 World Series?",
    "Give me a recipe for roast chicken.",
    "What is the capital of Australia?",
]

# (first question, follow-up, substring the first answer's document must contain)
FOLLOW_UPS = [
    ("Summarize the Data Center ABS report.", "What else does that report say about spreads?", "Data Center ABS"),
    ("What does the June 2026 US Housing Tracker say about new home sales?", "What does it forecast for home prices?", "Housing Tracker"),
    ("What did the September 11 securitized products research say about CLOs?", "And what did it say about subprime auto in that same report?", "September 11"),
    ("What is the recommendation in Buy CLO AAAs vs. CMO Floaters?", "What regulatory changes does it mention?", "CLO AAAs"),
]


async def turn(text: str, history: list[dict] | None = None) -> dict:
    req = KnowledgeChatRequest(text=text, history=history or [], timezone=TZ)
    t0 = time.monotonic()
    ttft = None
    answer, sources, tools, status = "", [], [], None
    async for ev in kc.run(req, USER):
        if "token" in ev:
            if ttft is None and ev["token"].strip():
                ttft = time.monotonic() - t0
            answer += ev["token"]
        elif "sources" in ev:
            sources = ev["sources"]
        elif "tool_call" in ev:
            tools.append(ev["tool_call"])
        elif "status" in ev:
            status = ev["status"]
    return {"answer": answer, "sources": sources, "tools": tools, "status": status,
            "ttft": ttft, "total": time.monotonic() - t0}


def titles(r: dict) -> list[str]:
    return list(dict.fromkeys(s["title"] for s in r["sources"]))


async def main() -> int:
    results = {"answerable": [], "out_of_scope": [], "follow_ups": []}
    ttfts: list[float] = []
    ok_all = True

    for q, expect in ANSWERABLE:
        r = await turn(q)
        ttfts.append(r["ttft"] or r["total"])
        ok = r["status"] == "completed" and bool(r["sources"]) and all(
            any(e in t for e in expect) for t in titles(r)
        ) and any(any(e in t for e in expect) for t in titles(r))
        ok_all &= ok
        results["answerable"].append({"q": q, "ok": ok, "cited": titles(r), "tools": [t["name"] for t in r["tools"]], "ttft": round(r["ttft"] or 0, 1)})
        print(("PASS" if ok else "FAIL"), f"{r['ttft'] or 0:5.1f}s", q, "->", titles(r))
        if not ok:
            print("     ", r["answer"][:400].replace("\n", " "))

    for q in OUT_OF_SCOPE:
        r = await turn(q)
        ttfts.append(r["ttft"] or r["total"])
        ok = r["status"] == "completed" and not r["sources"]
        ok_all &= ok
        results["out_of_scope"].append({"q": q, "ok": ok, "answer": r["answer"][:200]})
        print(("PASS" if ok else "FAIL"), f"{r['ttft'] or 0:5.1f}s", "[out of scope]", q, "->", r["answer"][:120].replace("\n", " "))

    for first, follow, doc in FOLLOW_UPS:
        a = await turn(first)
        hist = [
            {"role": "user", "content": first},
            {"role": "assistant", "content": a["answer"], "sources": [{"document_id": s["document_id"], "title": s["title"]} for s in a["sources"]]},
        ]
        b = await turn(follow, hist)
        ttfts += [a["ttft"] or a["total"], b["ttft"] or b["total"]]
        ok = bool(b["sources"]) and all(doc in t for t in titles(b)) and any(doc in t for t in titles(a))
        ok_all &= ok
        results["follow_ups"].append({"first": first, "follow": follow, "ok": ok, "first_cited": titles(a), "follow_cited": titles(b)})
        print(("PASS" if ok else "FAIL"), "[follow-up]", follow, "->", titles(b))
        if not ok:
            print("      first:", titles(a), "| follow answer:", b["answer"][:300].replace("\n", " "))

    med = statistics.median(ttfts)
    ok_all &= med < 8
    summary = {
        "answerable": f"{sum(x['ok'] for x in results['answerable'])}/{len(ANSWERABLE)}",
        "out_of_scope_declined": f"{sum(x['ok'] for x in results['out_of_scope'])}/{len(OUT_OF_SCOPE)}",
        "follow_ups_same_doc": f"{sum(x['ok'] for x in results['follow_ups'])}/{len(FOLLOW_UPS)}",
        "median_ttft_s": round(med, 2),
        "max_ttft_s": round(max(ttfts), 2),
        "pass": ok_all,
    }
    print(json.dumps(summary))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
