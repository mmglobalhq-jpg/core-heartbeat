"""Extract plain text from an uploaded document.

Strategy:
- Cheap, dependency-free fast path for already-textual files (txt/md/csv/…).
- Everything else (PDF, DOCX, XLSX, PPTX, HTML, images) goes through **docling**,
  which is lazy-imported inside the worker so importing this module stays cheap and
  the sandbox/tests never need docling installed.

Resource safety (docling is CPU/memory-heavy and shares the box with Ollama): a
module-wide Semaphore(1) means at most one docling parse runs at a time, it's
offloaded to a worker thread, and it's bounded by a timeout. Parsing happens at
UPLOAD time, never on the chat latency path.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# One heavy parse at a time (protects memory on a single node running Ollama too).
_parse_sem = asyncio.Semaphore(1)
PARSE_TIMEOUT_S = 180

# Files that are already text — decode directly, no docling.
_FAST_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json"}


def _ext(filename: str) -> str:
    return ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""


def _fast_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _docling_extract(data: bytes, filename: str) -> str:
    """Blocking docling conversion → Markdown. Lazy-imports docling."""
    from io import BytesIO

    from docling.datamodel.base_models import DocumentStream
    from docling.document_converter import DocumentConverter

    # A named stream lets docling detect the format from the filename suffix without
    # a temp file on disk.
    source = DocumentStream(name=filename, stream=BytesIO(data))
    result = DocumentConverter().convert(source)
    return result.document.export_to_markdown()


async def parse_document(
    data: bytes, filename: str, content_type: str | None = None
) -> str:
    """Return extracted text for ``data``. Fast path for textual files; docling
    otherwise (bounded to one at a time, thread-offloaded, timed out). Raises on a
    docling failure so the caller can mark the document ``error``."""
    if _ext(filename) in _FAST_EXT:
        return _fast_text(data)
    async with _parse_sem:
        return await asyncio.wait_for(
            asyncio.to_thread(_docling_extract, data, filename),
            timeout=PARSE_TIMEOUT_S,
        )
