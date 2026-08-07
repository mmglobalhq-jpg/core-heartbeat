"""Model access for the briefing pipeline: local first, escalation on a budget.

WHY LOCAL FIRST
A briefing touches every candidate story at least once — classifying, comparing,
summarising. Doing that against a hosted model means paying per story to discover
that most stories will not be used. The local model is adequate for those jobs
and free, so it does them.

Escalation is reserved for the two places quality is visible to the reader: the
Deep Dive and the editorial pass. It is capped per run by ``EscalationBudget``,
which RAISES on exhaustion rather than silently continuing. A cost ceiling that
degrades quietly is not a ceiling — it is a surprise on a bill.
"""

from __future__ import annotations

import logging
import os

import httpx

from briefing import config

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No model could serve the request."""


class EscalationExhausted(RuntimeError):
    """The per-briefing hosted-model budget is spent."""


class EscalationBudget:
    """Counts hosted-model calls for one briefing.

    Deliberately not a global: two briefings running concurrently must not share
    a budget, and a long-lived process must not accumulate one run's spend into
    the next.
    """

    def __init__(self, limit: int | None = None) -> None:
        self.limit = config.MAX_ESCALATIONS if limit is None else limit
        self.spent = 0

    def take(self) -> None:
        if self.spent >= self.limit:
            raise EscalationExhausted(
                f"escalation budget exhausted ({self.spent}/{self.limit} hosted calls used)"
            )
        self.spent += 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def __repr__(self) -> str:
        return f"<EscalationBudget {self.spent}/{self.limit}>"


# --- local (Ollama) ----------------------------------------------------------

# Module-level test seam, matching services/pending_plans.py. None -> real network.
_transport: httpx.BaseTransport | None = None


def _client(timeout: float | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=config.OLLAMA_URL,
        timeout=timeout or config.LLM_TIMEOUT_S,
        transport=_transport,
    )


def embed(text: str) -> list[float]:
    """Embed one string with the local embedding model."""
    if not (text or "").strip():
        return []
    with _client(timeout=30.0) as client:
        response = client.post(
            "/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": text},
        )
        response.raise_for_status()
        return response.json().get("embedding") or []


def generate_local(prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
    """Run a prompt on the local model.

    Low temperature by default: this pipeline summarises sourced material, and
    creative variance in that job shows up as invented detail.
    """
    payload: dict = {
        "model": config.LOCAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    try:
        with _client() as client:
            response = client.post("/api/generate", json=payload)
            response.raise_for_status()
            return (response.json().get("response") or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(f"local model {config.LOCAL_MODEL} unavailable: {exc}") from exc


# --- escalation (hosted) -----------------------------------------------------


def _google_key() -> str | None:
    from services.secrets import secret

    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = secret(name) or os.environ.get(name)
        if value:
            return value
    return None


def generate_escalated(
    prompt: str,
    *,
    system: str | None = None,
    budget: EscalationBudget,
    temperature: float = 0.3,
) -> str:
    """Run a prompt on the hosted model, spending one unit of budget.

    Budget is taken BEFORE the call, so a failed call still counts. Otherwise a
    provider erroring in a retry loop would spend without limit while the counter
    stayed at zero.
    """
    budget.take()
    key = _google_key()
    if not key:
        raise LLMUnavailable("no hosted model key configured")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=config.ESCALATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(f"hosted model {config.ESCALATION_MODEL} failed: {exc}") from exc
    return (getattr(response, "text", "") or "").strip()


def generate(
    prompt: str,
    *,
    system: str | None = None,
    budget: EscalationBudget | None = None,
    prefer_hosted: bool = False,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """Generate text, returning ``(text, which_model)``.

    With ``prefer_hosted`` the hosted model is tried first and the local model is
    the fallback — so a spent budget or an unreachable provider degrades the
    Deep Dive's prose rather than losing the section. Without it, local only.
    """
    if prefer_hosted and budget is not None:
        try:
            return generate_escalated(prompt, system=system, budget=budget,
                                      temperature=temperature), config.ESCALATION_MODEL
        except (EscalationExhausted, LLMUnavailable) as exc:
            logger.info("escalation unavailable (%s); using local model", exc)
    return generate_local(prompt, system=system, temperature=temperature), config.LOCAL_MODEL


def local_available() -> bool:
    try:
        with _client(timeout=5.0) as client:
            return client.get("/api/tags").status_code == 200
    except Exception:  # noqa: BLE001
        return False
