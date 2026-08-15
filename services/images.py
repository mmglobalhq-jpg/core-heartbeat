"""Preparing uploaded images for a model.

Extracted from orchestrator.py on 2026-08-15 so that BOTH the chat turn and the
``reread_attachment`` tool can use one implementation. The import direction makes
this necessary rather than merely tidy: ``orchestrator`` imports ``tools.*``, so a
tool cannot reach back into the orchestrator without a cycle. The logic is a
storage/media concern, not an orchestration one, so ``services`` is where it
belongs.

Behaviour is unchanged from the original ``_downscale_image``: decode, cap the
long edge, re-encode, reject anything still oversized.
"""

from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

# Formats we will hand to a model. Kept narrow deliberately: these are the two the
# providers all accept without transcoding surprises.
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})

# Formats a browser or phone will happily upload that are NOT in the set above.
# Tracked separately so the caller can say "I can't see this one" instead of
# silently falling back to OCR text and answering as though it had looked.
KNOWN_UNVIEWABLE_MEDIA_TYPES = frozenset(
    {"image/webp", "image/heic", "image/heif", "image/avif", "image/gif", "image/tiff"}
)

MAX_IMAGE_EDGE_PX = 1568     # Anthropic's recommended long edge; larger is downscaled
MAX_IMAGE_BYTES = 4_000_000  # skip anything still over this AFTER downscaling

# A decompression bomb is a small file that expands to an enormous bitmap. Pillow
# warns by default at ~89 Mpx and only raises at twice that; neither is a decision
# this service should leave to a default. 64 Mpx is far above any real photograph
# or page scan (a 40 MP phone camera is 40 Mpx) and far below anything that would
# exhaust a 14 GB box.
MAX_IMAGE_PIXELS = 64_000_000


def downscale_for_model(data: bytes) -> tuple[bytes, str] | None:
    """Return ``(bytes, media_type)`` ready for a model, or None if unusable.

    None means "cannot be shown to a model" for any reason — unsupported format,
    corrupt bytes, decompression bomb, or still too large after downscaling. The
    caller must treat that as *not seen*, never as *seen and unremarkable*.
    """
    from io import BytesIO

    from PIL import Image

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            if fmt not in ("PNG", "JPEG"):
                return None
            # Check the declared dimensions BEFORE decoding. Setting
            # Image.MAX_IMAGE_PIXELS is not sufficient on its own: Pillow only
            # *warns* at that threshold and does not raise until twice it, so a
            # 65 Mpx bomb against a 64 Mpx limit would sail through with a warning
            # and allocate anyway. `open()` reads the header only, so this costs
            # nothing and is decided before any pixels exist.
            w, h = im.size
            if w * h > MAX_IMAGE_PIXELS:
                logger.info("refusing image: %dx%d exceeds the pixel budget", w, h)
                return None
            im.load()
            longest = max(im.size)
            if longest > MAX_IMAGE_EDGE_PX:
                scale = MAX_IMAGE_EDGE_PX / longest
                im = im.resize(
                    (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                    Image.LANCZOS,
                )
            buf = BytesIO()
            if fmt == "PNG":
                im.save(buf, format="PNG", optimize=True)
                media = "image/png"
            else:
                # JPEG cannot carry alpha; convert so an RGBA source can't fail here.
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(buf, format="JPEG", quality=85, optimize=True)
                media = "image/jpeg"
            out = buf.getvalue()
    except Exception as exc:
        logger.info("image could not be prepared for the model: %s", type(exc).__name__)
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    if not out or len(out) > MAX_IMAGE_BYTES:
        return None
    return out, media


def prepare_from_storage(user_id: str, doc_id: str) -> dict | None:
    """Fetch an uploaded original and prepare it. Blocking (boto3 + Pillow)."""
    from services import documents as docstore

    try:
        raw = docstore.fetch_original(user_id, doc_id)
    except Exception:
        return None
    prepared = downscale_for_model(raw)
    if prepared is None:
        return None
    data, media = prepared
    return {
        "media_type": media,
        "data": base64.b64encode(data).decode("ascii"),
        "doc_id": doc_id,
    }
