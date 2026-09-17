"""Replay representative main-chat prompts through the REAL orchestrator, in-process.

    docker exec -i -e REPLAY_USER=<uuid> core-heartbeat python - < scripts/main_chat_replay.py

Written for the knowledge-base removal (2026-09-17): run it before and after, and compare
which tools each turn called and its time to first token. Memory extraction is disabled
for the run so test prompts never land in the user's saved preferences. Read-only
prompts only — nothing here creates, changes or deletes anything.
"""
import asyncio
import json
import os
import time

import orchestrator
from models import IntentPayload

orchestrator.schedule_memory_extraction = lambda *a, **k: None  # never write test turns to the profile

USER = os.environ["REPLAY_USER"]
PROMPTS = [
    "hi",
    "what's on my calendar tomorrow?",
    "What is the latest ARR report about?",
    "what's the weather forecast in Savannah this weekend?",
    "what topics are on my daily brief?",
    "how do I roast a chicken?",
    "summarize the August 28 securitized products report",
    "what did JPM say about subprime auto delinquencies in my saved research?",
    "what is 15% of 2,340?",
    "thanks!",
]


async def turn(text: str) -> dict:
    p = IntentPayload(intent="chat", confidence=0.95, raw_input=text, source="replay",
                      timezone="America/Chicago")
    t0 = time.monotonic()
    ttft, reply, tools, status = None, "", [], None
    async for ev in orchestrator.astream_run(p, USER):
        if "token" in ev:
            if ttft is None and ev["token"].strip():
                ttft = time.monotonic() - t0
            reply += ev["token"]
        elif "tool_call" in ev:
            tools.append(ev["tool_call"].get("name"))
        elif "status" in ev:
            status = ev["status"]
    return {"prompt": text, "tools": tools, "ttft_s": round(ttft or 0, 2),
            "total_s": round(time.monotonic() - t0, 2), "status": status,
            "reply": reply[:220].replace("\n", " ")}


async def main():
    out = []
    for p in PROMPTS:
        r = await turn(p)
        out.append(r)
        print(json.dumps(r))
    kb = sum(1 for r in out for t in r["tools"] if "knowledge" in (t or "") or t == "summarize_document")
    ttfts = sorted(r["ttft_s"] for r in out)
    print(json.dumps({"kb_calls": kb, "median_ttft_s": ttfts[len(ttfts) // 2], "sum_total_s": round(sum(r["total_s"] for r in out), 1)}))


asyncio.run(main())
