"""Re-open an image the user attached earlier in the conversation.

THE DEFECT THIS EXISTS FOR
``document_images`` is built from ``payload.document_ids``, which carries only the
CURRENT message's attachments. So the model sees an uploaded image on the turn it
arrives and never again. Asked "do you see fall break on the image?" two turns
later, it had no image — and answered anyway, from the OCR text and chat history.
On 2026-08-15 that produced a confident wrong date, then a second confident wrong
date when challenged to "check the image again". It could not check. Nothing in
the system let it.

Re-sending every attachment on every turn is the obvious fix and the wrong one: it
multiplies token cost on every hop for the overwhelming majority of turns that
never refer back. A tool makes the cost proportional to the need — paid only when
the model actually decides to look again.

The answer text returned here is UNTRUSTED: it is a description of a user-supplied
image, so it is contained the same way document text is before it goes back into
the conversation.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

VISION_MODEL_ENV = "REREAD_VISION_MODEL"
DEFAULT_VISION_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

MAX_ANSWER_CHARS = 4000


class AttachmentError(RuntimeError):
    """Raised for a condition worth telling the model about verbatim."""


def _vision_answer(image: dict, question: str) -> str:
    """Ask the vision model one question about one prepared image.

    Synchronous on purpose: catalog tools are sync, and this runs inside the
    tool-execution node rather than on the streaming path.
    """
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if not key or not key.strip():
        raise AttachmentError("no vision model is configured, so I cannot re-open images")
    try:
        from google import genai
    except Exception as exc:  # SDK absent — degrade like a missing key
        raise AttachmentError("the vision SDK is unavailable") from exc

    client = genai.Client(api_key=key)
    model = os.environ.get(VISION_MODEL_ENV) or DEFAULT_VISION_MODEL
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Answer only from what is visibly present in this image. "
                        "Quote the exact text you are reading for any date, time, "
                        "figure or name you report. If the image does not show the "
                        "answer, or the text is too small or unclear to read with "
                        "confidence, say so plainly instead of inferring it. Do not "
                        "use general knowledge to fill a gap.\n\n"
                        f"Question: {question}"
                    )
                },
                {
                    "inline_data": {
                        "mime_type": image["media_type"],
                        "data": image["data"],
                    }
                },
            ],
        }
    ]
    from services import llm_ledger

    # Re-reading an image is a full multimodal request and was never recorded.
    with llm_ledger.attempt("gemini", model, "heartbeat.vision") as record:
        resp = client.models.generate_content(model=model, contents=contents)
        record.ok(
            getattr(resp, "usage_metadata", None),
            model_served=getattr(resp, "model_version", None),
            request_id=getattr(resp, "response_id", None),
        )
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise AttachmentError("the vision model returned nothing for that image")
    return text[:MAX_ANSWER_CHARS]


def reread_attachment(user_id: str, args: dict) -> str:
    """Look at an attached image again and answer one question about it."""
    from services.untrusted import DOCUMENT_PREAMBLE, wrap
    from services import documents as docstore
    from services.images import (
        IMAGE_MEDIA_TYPES,
        KNOWN_UNVIEWABLE_MEDIA_TYPES,
        prepare_from_storage,
    )

    doc_id = (args.get("doc_id") or "").strip()
    question = (args.get("question") or "").strip()
    if not doc_id:
        return "error: reread_attachment needs doc_id (the id shown in the attachment list)."
    if not question:
        return "error: reread_attachment needs a question to answer about the image."

    types = docstore.fetch_content_types(user_id, [doc_id])
    if doc_id not in types:
        # Either it is not this user's document or it does not exist. Same answer
        # either way: never confirm the existence of another user's attachment.
        return f"error: no attachment with id {doc_id} in this conversation."

    media = (types.get(doc_id) or "").split(";")[0].strip().lower()
    if media not in IMAGE_MEDIA_TYPES:
        if media in KNOWN_UNVIEWABLE_MEDIA_TYPES:
            return (
                f"I cannot view {media} images — only PNG and JPEG. I have the text "
                "extracted from this file, but I have not seen the picture itself, "
                "so I should not describe what it looks like."
            )
        return (
            f"That attachment is {media or 'an unknown type'}, not an image I can "
            "look at. Answer from its extracted text, and say that is what you are doing."
        )

    image = prepare_from_storage(user_id, doc_id)
    if image is None:
        return (
            "I could not open that image (it may be corrupt, or too large even "
            "after resizing). I have not seen it, so I should not describe it."
        )

    try:
        answer = _vision_answer(image, question)
    except AttachmentError as exc:
        return f"error: {exc}"
    except Exception as exc:
        logger.warning("reread_attachment failed for doc_id=%s: %s", doc_id, type(exc).__name__)
        return "error: the image could not be examined just now."

    return wrap(
        answer,
        label="IMAGE READING",
        limit=MAX_ANSWER_CHARS,
        preamble=DOCUMENT_PREAMBLE,
    )


_DISPATCH = {"reread_attachment": reread_attachment}

ATTACHMENT_TOOL_REGISTRY = frozenset(_DISPATCH)


def run_attachment_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered attachment tool; never raises (graph contract)."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    if not user_id or not str(user_id).strip():
        return "error: no user context for this attachment lookup."
    try:
        return fn(user_id, args or {})
    except Exception as exc:  # never crash the graph on a bad tool call
        logger.warning("attachment tool %s failed: %s", name, type(exc).__name__)
        return f"error: {type(exc).__name__}"
