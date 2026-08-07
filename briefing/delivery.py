"""Getting a finished briefing to its reader.

PRODUCTION EMAIL IS NOT WIRED UP. The platform has a Resend key, but it lives
inside the Supabase Edge Function that handles signup approval — it is not in
this service's environment, and putting it there is a production credential
change. ``ResendSender`` below is complete and, given a key, would work; nothing
in this build supplies one.

The default provider is therefore ``file``, and the default is chosen so that a
half-configured environment writes to disk instead of emailing someone. A sender
that silently falls back to "send for real" is the wrong shape for this.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pathlib
from typing import Protocol

import httpx

from briefing import config

logger = logging.getLogger(__name__)


class DeliveryResult:
    def __init__(self, status: str, provider: str, detail: str = "") -> None:
        self.status = status  # 'sent' | 'failed' | 'skipped'
        self.provider = provider
        self.detail = detail

    def __repr__(self) -> str:
        return f"<DeliveryResult {self.provider} {self.status}: {self.detail}>"


class EmailSender(Protocol):
    name: str

    def send(self, *, to: str, subject: str, html: str, text: str) -> DeliveryResult: ...


class FileSender:
    """Write the rendered briefing to disk instead of sending it.

    This is what the end-to-end test uses. The output is the real rendered
    artefact — the same HTML that would have been the email body — so it can be
    opened in a browser and checked, without anything leaving the machine.
    """

    name = "file"

    def __init__(self, directory: str | None = None) -> None:
        self.directory = pathlib.Path(directory or config.EMAIL_OUTPUT_DIR)

    def send(self, *, to: str, subject: str, html: str, text: str) -> DeliveryResult:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        safe_to = "".join(c if c.isalnum() or c in "-_.@" else "_" for c in to)[:60]
        base = self.directory / f"briefing-{stamp}-{safe_to}"
        base.with_suffix(".html").write_text(html, encoding="utf-8")
        base.with_suffix(".txt").write_text(f"Subject: {subject}\nTo: {to}\n\n{text}",
                                            encoding="utf-8")
        logger.info("briefing written to %s.{html,txt}", base)
        return DeliveryResult("sent", self.name, str(base.with_suffix(".html")))


class ResendSender:
    """Send through Resend. Complete, but not wired to a credential in this build.

    Mirrors the request the existing signup-approval Edge Function makes, so the
    two use the same provider in the same way.
    """

    name = "resend"

    def __init__(self, api_key: str | None = None, sender: str | None = None) -> None:
        from services.secrets import secret

        self.api_key = api_key or secret("RESEND_API_KEY") or os.environ.get("RESEND_API_KEY")
        self.sender = sender or config.EMAIL_FROM

    def send(self, *, to: str, subject: str, html: str, text: str) -> DeliveryResult:
        if not self.api_key:
            # Not an exception: a missing key is a configuration state, and the
            # run should record "not delivered" and carry on rather than fail
            # after successfully producing a briefing.
            return DeliveryResult("skipped", self.name, "no RESEND_API_KEY configured")
        try:
            response = httpx.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"from": self.sender, "to": to, "subject": subject,
                      "html": html, "text": text},
                timeout=20.0,
            )
        except Exception as exc:  # noqa: BLE001
            return DeliveryResult("failed", self.name, f"{type(exc).__name__}")
        if response.status_code >= 400:
            # Never log the body verbatim — provider errors have been known to
            # echo request headers back.
            return DeliveryResult("failed", self.name, f"http {response.status_code}")
        return DeliveryResult("sent", self.name, (response.json() or {}).get("id", ""))


class NullSender:
    """Deliver nothing. For runs where email is off."""

    name = "none"

    def send(self, *, to: str, subject: str, html: str, text: str) -> DeliveryResult:
        return DeliveryResult("skipped", self.name, "delivery disabled")


SENDERS: dict[str, type] = {"file": FileSender, "resend": ResendSender, "none": NullSender}


def sender_for(name: str | None = None) -> EmailSender:
    """Build the configured sender. Unknown names fall back to the file sink.

    An unrecognised provider must not become "send it for real by default".
    """
    key = (name or config.EMAIL_PROVIDER or "file").lower()
    if key not in SENDERS:
        logger.warning("unknown email provider %r; writing to file instead", key)
        key = "file"
    return SENDERS[key]()
