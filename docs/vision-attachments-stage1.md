# Vision attachments — Stage 1 (read-only)

*Plan only. Nothing in this document has been implemented.
Written 2026-08-05 against `core-heartbeat` `main` @ `436b5a6` and
`core-chat` `main` @ `c0ffa25`.*

---

## Goal

Let the user attach a screenshot and have the model **actually see it** — describe
it, and read data out of it — instead of only receiving OCR'd text.

**Stage 1 is deliberately read-only.** The model may look at an image and answer
about it. It may **not** create calendar events (or take any other write action)
from what it sees. That is Stage 2, and it is gated on trusting what Stage 1
reads. Rationale in [Why two stages](#why-two-stages).

## Current behaviour

`.png`, `.jpg`, `.jpeg` are already accepted by the uploader
(`core-chat/components/chat/ChatInput.tsx:28`) and upload succeeds. But an image
takes the same path as every document:

```
image → Docling OCR → export_to_markdown() → text → prompt
```

The model never receives the image. So "read the text in this receipt" works;
"what's in this screenshot" does not.

## What already exists (and makes this tractable)

| Piece | Where | Status |
|---|---|---|
| Image accepted by the uploader | `ChatInput.tsx:28` | ✅ done |
| Original bytes stored | `user-docs` bucket, key `<user_id>/<doc_id>/original` | ✅ done |
| **Backend fetch of original bytes** | `services/documents.py:21` `fetch_original(user_id, doc_id) -> bytes` | ✅ **already exists** |
| Content type recorded | `documents.content_type` column | ✅ done |
| Vision-capable supervisor | Anthropic Messages API | ✅ capable |
| Vision-capable compose model | `COMPOSE_MODEL=gemini-2.5-flash` | ✅ capable |
| Pillow for downscaling | in the image, v12.2.0 | ✅ available |

**No new storage, credentials, tables or dependencies are required.**

---

## The actual blocker

Prompts are **strings**, and both model calls pass them as scalar content:

| Call site | Line | Today |
|---|---|---|
| Supervisor (Anthropic) | `orchestrator.py:636` | `messages=[{"role":"user","content": prompt}]` |
| Compose stream (Gemini) | `orchestrator.py:1213` | `contents=prompt` |
| Memory extraction (Gemini) | `orchestrator.py:1484` | `contents=prompt` |
| Routing (Gemini path) | `orchestrator.py:565` | `contents=prompt` |

To attach an image these must become **content-part lists**. Both SDKs accept
that shape already; the change is ours, not theirs.

This is the hot path — every routing decision and every answer flows through it.
That is what makes this a day of work rather than an hour, and why the rollout
below is incremental.

---

## Design

### 1. Carry images in graph state, beside the text

Add one field to `GraphState` (`orchestrator.py:~219`, next to `documents: str`):

```python
# Attached IMAGE documents for this turn, already downscaled and base64-encoded,
# as [{"media_type": "image/png", "data": "<b64>", "filename": "..."}].
# Set-once, no reducer — mirrors `documents`.
document_images: list[dict]
```

Keeping it separate from `documents: str` means the existing text path is
untouched, and a turn with no images behaves **byte-identically to today**.

### 2. New loader beside `_load_documents`

`_load_documents` (`orchestrator.py:1670`) stays exactly as-is. Add a sibling:

```python
IMAGE_MEDIA_TYPES = {"image/png", "image/jpeg"}
MAX_IMAGES_PER_TURN = 3
MAX_IMAGE_EDGE_PX  = 1568   # Anthropic's recommended long-edge cap
MAX_IMAGE_BYTES    = 4_000_000

async def _load_document_images(user_id, document_ids, content_types) -> list[dict]:
    """Fetch, downscale and base64 attached images. Best-effort per doc; never raises."""
```

- selects docs whose `content_type` is in `IMAGE_MEDIA_TYPES`
- `docstore.fetch_original()` for bytes (already exists)
- Pillow: downscale so the long edge ≤ `MAX_IMAGE_EDGE_PX`, re-encode
- skips anything still over `MAX_IMAGE_BYTES` rather than sending it
- returns `[]` on any failure — **an image problem must never break a chat turn**

### 3. Keep OCR *and* the image

Docling still runs. The prompt gets both the extracted text and the image.

This is deliberate: for dense tables and receipts, OCR text is frequently more
accurate than vision alone, and the two together beat either. It also means that
if the image path fails, the turn degrades to exactly today's behaviour.

### 4. Prompt builders return content parts

Introduce a small helper rather than rewriting the builders:

```python
def _as_content_parts(prompt: str, images: list[dict], provider: str) -> list | str:
    """Return provider-shaped content. With no images, returns the plain string so
    the existing call path is completely unchanged."""
```

- **`anthropic`** → `[{"type":"text","text":prompt},
  {"type":"image","source":{"type":"base64","media_type":…,"data":…}}, …]`
- **`gemini`** → `[prompt, {"inline_data":{"mime_type":…,"data":…}}, …]`

> **The no-image case must return the bare string.** That keeps every existing
> code path, test and behaviour identical when nothing is attached, which is the
> overwhelming majority of turns.

### 5. Attach images on the first step only

```python
if state.get("step", 0) == 0 and state.get("document_images"):
```

The supervisor runs on **every** routing decision (up to `MAX_STEPS = 8`).
Re-sending a screenshot each time would multiply token cost ~8× for no benefit —
after step 0 the conversation already carries what the model read.

### 6. Ollama fallback

`OLLAMA_MODEL=qwen2.5:7b` **cannot see images**. When compose falls back to local
with images present, return an explicit message rather than silently answering
from OCR text as though it had looked:

> "I can read the text extracted from your image, but the local model can't see
> images. The cloud model is unavailable right now — try again shortly."

Optional later: pull a vision model (`llava` / `qwen2-vl`, ~4–8 GB; disk is at 9%
so there is room) and make the fallback real.

---

## Files touched

| File | Change | Size |
|---|---|---|
| `orchestrator.py` | `GraphState.document_images` field | XS |
| `orchestrator.py` | `_load_document_images()` + constants | S |
| `orchestrator.py` | `_as_content_parts()` helper | S |
| `orchestrator.py:636` | supervisor content parts | XS |
| `orchestrator.py:1213` | Gemini compose content parts | XS |
| `orchestrator.py` (run setup) | populate `document_images` beside `documents` | XS |
| `orchestrator.py` | Ollama-with-images guard | XS |
| `services/documents.py` | none — `fetch_original` already exists | — |
| `core-chat` | **none for Stage 1** | — |

`core-chat` needs no change: images already upload, and the answer renders as
normal text.

---

## Verification

`core-heartbeat` has 21 test files; `core-chat` has **zero**. Verification here is
mostly manual, so it must be deliberate.

**Regression — must be identical to today:**
1. Plain chat turn, no attachment
2. PDF attachment → still OCR'd and answered
3. KB question → unchanged
4. REIT question → unchanged
5. Calendar *read* → unchanged
6. `/health` and `/health/fund-pollers` → unchanged

**New behaviour:**
7. Screenshot of a schedule → describes it and lists the entries
8. Photo with no text → describes it (today: nothing useful)
9. Receipt → OCR text *and* visual layout both usable
10. 3 images at once → all seen, capped at `MAX_IMAGES_PER_TURN`
11. Oversized image → skipped gracefully, turn still answers
12. Corrupt/truncated image file → turn still answers, no crash
13. Cloud compose forced to fail with an image attached → clear message, not a silent OCR answer

**Cost check:** compare token usage for one image turn against a text turn, and
confirm the image is attached once, not once per step.

---

<a id="why-two-stages"></a>
## Why two stages

Stage 2 ("add this schedule to my calendar") means the model creates real events
from something it *read off a picture*. A misread screenshot could create a dozen
wrong events, and `create_calendar_event` has no undo beyond manual deletion.

Stage 1 gives most of the value — you can paste a screenshot and ask about it —
while keeping every write path exactly as it is today. Once you have seen how
accurately it reads real screenshots, Stage 2 becomes an informed decision rather
than a hopeful one.

Stage 2 should also add a **propose-then-confirm** step: the model lists the
events it intends to create and waits for approval, rather than writing directly.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hot-path regression breaks all chat | Medium | No-image case returns the bare string, so existing paths are untouched; regression list above |
| Token cost rises unexpectedly | Medium | Step-0-only attachment; downscaling; `MAX_IMAGES_PER_TURN = 3` |
| Model misreads a screenshot | **High — inherent** | Stage 1 is read-only; nothing is written from a misread |
| Ollama fallback silently degrades | Medium | Explicit guard message |
| Large image exhausts memory | Low | Downscale before encode; hard byte cap; skip rather than send |
| `core-chat` has no tests | — | Manual regression list; `core-chat` is unchanged in Stage 1 |

## Effort

Roughly **one day**, dominated by step 4 (content parts) and the regression pass —
not by the image handling itself, which is largely already built.

## Explicitly out of scope for Stage 1

- Any tool **write** driven by image content (calendar, KB, vault) — Stage 2
- A local vision model
- Image generation or editing
- PDF *page* images (PDFs keep the existing Docling text path)
- `core-chat` UI changes such as thumbnails or previews
