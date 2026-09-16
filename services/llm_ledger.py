"""llm_ledger — record every model-provider attempt as one JSON line.

Canonical source: ~/projects/llm-ledger/python/llm_ledger.py. Services build from
their own directories, so each carries a COPY; `llm-ledger/scripts/check-copies.sh`
fails if a copy drifts. Edit here, then re-copy.

Stdlib only, so it drops into any image (heartbeat is 3.14, REIT 3.12).

The contract (shared with ts/llm-ledger.ts, pinned by fixtures/usage-normalization.json):
  * one line per HTTP ATTEMPT — retries and fallbacks are separate lines
  * tokens only, never dollars — the database prices them from llm_rates
  * no usage => usage_source "missing" and null token fields, never zeros
  * recording must never break the call it records: every failure here is swallowed

Lines go to $LLM_LEDGER_SPOOL/<service>/<UTC date>.jsonl. The host's drain ships them
to public.llm_calls and is idempotent on `id`.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

SPOOL_DIR = os.environ.get("LLM_LEDGER_SPOOL", "/var/spool/llm-ledger")
SERVICE = os.environ.get("LLM_LEDGER_SERVICE", "")
PROVIDERS = ("anthropic", "openai", "gemini", "ollama")
TOKEN_FIELDS = (
    "input_uncached", "input_audio", "cache_write_5m", "cache_write_1h", "cache_read",
    "output_billable", "thinking_tokens", "tool_use_prompt_tokens", "web_search_requests",
)

_warned = False


def _get(obj: Any, *names: str) -> Any:
    """Read a field from a dict or an SDK object, trying each spelling in turn."""
    if obj is None:
        return None
    for n in names:
        if isinstance(obj, Mapping):
            if n in obj and obj[n] is not None:
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return None


def _int(v: Any) -> int | None:
    return None if v is None else int(v)


def _as_plain(obj: Any) -> Any:
    """Best-effort JSON-safe copy of a usage object, for raw_usage."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v) for v in obj]
    for m in ("model_dump", "to_dict", "dict"):
        f = getattr(obj, m, None)
        if callable(f):
            try:
                return _as_plain(f())
            except Exception:
                pass
    return str(obj)


# ------------------------------------------------------------------ normalizers
def normalize_anthropic(u: Any) -> dict:
    # TWO SPELLINGS, one provider. The raw Messages API returns snake_case
    # (input_tokens, cache_creation...); the Vercel AI SDK returns its own camelCase
    # LanguageModelUsage. briefing-agent calls Anthropic through the AI SDK, so a
    # normalizer that only knew the raw shape would silently record zeros there.
    if _get(u, "input_tokens") is None and _get(u, "inputTokens") is not None:
        d = _get(u, "inputTokenDetails")
        od = _get(u, "outputTokenDetails")
        cache_write = _int(_get(d, "cacheWriteTokens"))
        no_cache = _int(_get(d, "noCacheTokens"))
        return {
            # noCacheTokens IS the uncached remainder. Without details, inputTokens is
            # the whole prompt and nothing says how much was cached, so it stands alone
            # and the cache fields stay None (not reported) rather than 0 (none).
            "input_uncached": no_cache if no_cache is not None else _int(_get(u, "inputTokens")),
            "input_audio": None,
            # The SDK reports one cache-write number with no TTL split; 5m is the default.
            "cache_write_5m": cache_write,
            "cache_write_1h": None if cache_write is None else 0,
            "cache_read": _int(_get(d, "cacheReadTokens")),
            "output_billable": _int(_get(u, "outputTokens")),
            "thinking_tokens": _int(_get(od, "reasoningTokens")),
            "tool_use_prompt_tokens": None,
            "web_search_requests": 0,
        }
    # input_tokens is ALREADY the uncached remainder; thinking is inside output_tokens.
    created_total = _int(_get(u, "cache_creation_input_tokens"))
    breakdown = _get(u, "cache_creation")
    w5 = _int(_get(breakdown, "ephemeral_5m_input_tokens"))
    w1 = _int(_get(breakdown, "ephemeral_1h_input_tokens"))
    if breakdown is None and created_total is not None:
        # No TTL breakdown: the default TTL is 5 minutes.
        w5, w1 = created_total, 0
    stu = _get(u, "server_tool_use")
    out = {
        "input_uncached": _int(_get(u, "input_tokens")),
        "input_audio": None,
        "cache_write_5m": w5,
        "cache_write_1h": w1,
        "cache_read": _int(_get(u, "cache_read_input_tokens")),
        "output_billable": _int(_get(u, "output_tokens")),
        "thinking_tokens": None,
        "tool_use_prompt_tokens": None,
        "web_search_requests": _int(_get(stu, "web_search_requests")) or 0,
    }
    for k in ("service_tier", "inference_geo", "speed"):
        v = _get(u, k)
        if v is not None:
            out[k] = str(v)
    return out


def normalize_langchain(u: Any) -> dict:
    """LangChain's provider-agnostic ``usage_metadata`` (a THIRD spelling).

    Semantics differ from every raw provider shape and getting it wrong is silent:
    LangChain documents ``input_tokens`` as the SUM of all input token types, so cache
    reads and cache writes are already inside it (the raw Anthropic ``input_tokens`` is
    the uncached remainder instead), and ``output_tokens`` is the sum of all output
    types, so reasoning is already inside it.
    """
    d = _get(u, "input_token_details") or {}
    od = _get(u, "output_token_details") or {}
    cache_read = _int(_get(d, "cache_read"))
    cache_write = _int(_get(d, "cache_creation"))
    audio = _int(_get(d, "audio"))
    total_in = _int(_get(u, "input_tokens"))
    uncached = None
    if total_in is not None:
        uncached = total_in - (cache_read or 0) - (cache_write or 0) - (audio or 0)
    return {
        "input_uncached": uncached,
        "input_audio": audio,
        # No TTL split is available through LangChain; 5 minutes is the default.
        "cache_write_5m": cache_write,
        "cache_write_1h": None if cache_write is None else 0,
        "cache_read": cache_read,
        "output_billable": _int(_get(u, "output_tokens")),
        "thinking_tokens": _int(_get(od, "reasoning")),
        "tool_use_prompt_tokens": None,
        "web_search_requests": 0,
    }


def normalize_openai(u: Any) -> dict:
    # prompt_tokens INCLUDES cached (and cache-write) tokens; completion includes reasoning.
    prompt = _int(_get(u, "prompt_tokens", "input_tokens")) or 0
    pd = _get(u, "prompt_tokens_details", "input_tokens_details")
    cached = _int(_get(pd, "cached_tokens")) or 0
    cache_write = _int(_get(pd, "cache_write_tokens"))
    cd = _get(u, "completion_tokens_details", "output_tokens_details")
    return {
        "input_uncached": prompt - cached - (cache_write or 0),
        "input_audio": None,
        "cache_write_5m": cache_write,
        "cache_write_1h": None,
        "cache_read": cached,
        "output_billable": _int(_get(u, "completion_tokens", "output_tokens")),
        "thinking_tokens": _int(_get(cd, "reasoning_tokens")),
        "tool_use_prompt_tokens": None,
        "web_search_requests": 0,
    }


def _modality_count(details: Any, modality: str) -> int:
    total = 0
    for d in details or []:
        if str(_get(d, "modality") or "").upper().endswith(modality):
            total += _int(_get(d, "tokenCount", "token_count")) or 0
    return total


def normalize_gemini(u: Any) -> dict:
    # promptTokenCount INCLUDES cached tokens; candidatesTokenCount EXCLUDES thoughts,
    # and thoughts are billed as output.
    prompt = _int(_get(u, "promptTokenCount", "prompt_token_count")) or 0
    cached = _int(_get(u, "cachedContentTokenCount", "cached_content_token_count")) or 0
    audio_in = _modality_count(_get(u, "promptTokensDetails", "prompt_tokens_details"), "AUDIO")
    audio_cached = _modality_count(_get(u, "cacheTokensDetails", "cache_tokens_details"), "AUDIO")
    uncached_audio = audio_in - audio_cached
    candidates = _int(_get(u, "candidatesTokenCount", "candidates_token_count")) or 0
    thoughts = _int(_get(u, "thoughtsTokenCount", "thoughts_token_count"))
    return {
        "input_uncached": prompt - cached - uncached_audio,
        "input_audio": uncached_audio if audio_in else None,
        "cache_write_5m": None,
        "cache_write_1h": None,
        "cache_read": cached,
        "output_billable": candidates + (thoughts or 0),
        "thinking_tokens": thoughts,
        "tool_use_prompt_tokens": _int(_get(u, "toolUsePromptTokenCount", "tool_use_prompt_token_count")),
        "web_search_requests": 0,
    }


def normalize_ollama(u: Any) -> dict:
    return {
        "input_uncached": _int(_get(u, "prompt_eval_count")),
        "input_audio": None,
        "cache_write_5m": None,
        "cache_write_1h": None,
        "cache_read": None,
        "output_billable": _int(_get(u, "eval_count")),
        "thinking_tokens": None,
        "tool_use_prompt_tokens": None,
        "web_search_requests": 0,
    }


_NORMALIZERS = {
    "langchain": normalize_langchain,
    "anthropic": normalize_anthropic,
    "openai": normalize_openai,
    "gemini": normalize_gemini,
    "ollama": normalize_ollama,
}


def normalize(provider: str, usage: Any, shape: str | None = None) -> dict | None:
    """Provider usage -> ledger token fields. None when there is no usage to read.

    ``shape`` names the CLIENT spelling when it is not the provider's own: pass
    ``"langchain"`` for a LangChain ``usage_metadata`` dict, whose field semantics differ
    from the raw provider response.
    """
    if usage is None:
        return None
    return _NORMALIZERS[shape or provider](usage)


# ------------------------------------------------------------------ writing
def _spool_line(row: dict) -> None:
    global _warned
    try:
        service = row["service"]
        day = row["occurred_at"][:10]
        d = os.path.join(SPOOL_DIR, service)
        os.makedirs(d, exist_ok=True)
        line = (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode()
        fd = os.open(os.path.join(d, f"{day}.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        try:
            os.write(fd, line)  # one write per line: O_APPEND keeps concurrent writers whole
        finally:
            os.close(fd)
    except Exception as e:  # never break the call being recorded
        if not _warned:
            print(f"llm_ledger: could not spool a row ({type(e).__name__}: {e})", file=sys.stderr)
            _warned = True


def _classify(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name or "timedout" in name:
        return "timeout"
    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)) or "cancel" in name or "abort" in name:
        return "aborted"
    return "error"


class Attempt:
    """One HTTP attempt. Use as a context manager around exactly one provider request.

        with llm_ledger.attempt("anthropic", "claude-opus-5", "brief.ranking",
                                run_ref=f"brief_run:{run_id}", group=g, attempt=n) as a:
            resp = client.messages.create(...)
            a.ok(resp.usage, model_served=resp.model, request_id=resp.id)

    Leaving the block without calling ok()/refused() records a failure: the exception
    type decides error / timeout / aborted, and usage is "missing".
    """

    def __init__(self, provider: str, model: str, operation: str, *, service: str | None = None,
                 run_ref: str | None = None, group: str | None = None, attempt: int = 1,
                 key_label: str | None = None) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        self.row: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "service": service or SERVICE,
            "operation": operation,
            "run_ref": run_ref,
            "request_group": group,
            "attempt": attempt,
            "provider": provider,
            "key_label": key_label,
            "model_requested": model,
            "model_served": None,
            "provider_request_id": None,
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
            "status": None,
            "http_status": None,
            "error_type": None,
            "usage_source": "missing",
            "grounded_prompts": 0,
            "raw_usage": None,
            **{k: None for k in TOKEN_FIELDS},
        }
        self.row["web_search_requests"] = 0
        self._t0 = time.monotonic()
        self.row["occurred_at"] = datetime.now(timezone.utc).isoformat()

    def ok(self, usage: Any, *, model_served: str | None = None, request_id: str | None = None,
           grounded_prompts: int = 0, status: str = "ok", shape: str | None = None) -> None:
        self.row["status"] = status
        self.row["model_served"] = model_served
        self.row["provider_request_id"] = request_id
        self.row["grounded_prompts"] = grounded_prompts
        tokens = normalize(self.row["provider"], usage, shape)
        if tokens is not None:
            self.row.update(tokens)
            self.row["usage_source"] = "provider"
            self.row["raw_usage"] = _as_plain(usage)

    def refused(self, usage: Any, **kw: Any) -> None:
        self.ok(usage, status="refused", **kw)

    def __enter__(self) -> "Attempt":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.row["status"] is None:
            self.row["status"] = _classify(exc) if exc is not None else "error"
            if exc is not None:
                self.row["error_type"] = type(exc).__name__
                code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                self.row["http_status"] = int(code) if isinstance(code, int) else None
        self.row["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.row["latency_ms"] = int((time.monotonic() - self._t0) * 1000)
        if not self.row["service"]:
            self.row["service"] = "unknown"
        _spool_line(self.row)
        return False  # never swallow the caller's exception


def attempt(provider: str, model: str, operation: str, **kw: Any) -> Attempt:
    return Attempt(provider, model, operation, **kw)


def new_group() -> str:
    """An id shared by every attempt (retry / fallback) of one logical call."""
    return str(uuid.uuid4())
