"""Fetching a public web page safely from inside the gateway.

The security problem is specific and not theoretical. This process sits on
`heartbeat-net` alongside graph-rag (172.19.0.4), the graph-rag sidecar
(172.19.0.2) and core-chat (172.19.0.6), and can reach the host gateway
(172.18.0.1) and from there Ollama and anything else bound on the host. It also
holds Supabase service-role keys.

A URL-fetching tool is driven by *model output*, and model output is influenced by
page content, uploaded documents and chat text — all attacker-reachable in
principle. So "fetch this URL" is a request to make an HTTP call on behalf of
something that can be talked into choosing the URL. Without the checks below,
"summarise http://graph-rag:3000/api/..." or "…http://169.254.169.254/…" is a
straightforward read of internal services.

Defences here:

* scheme restricted to http/https — no file://, gopher://, data:
* every hostname resolved BEFORE connecting, and rejected if ANY resolved address
  is private, loopback, link-local, reserved, multicast or unspecified. Checking
  the resolved address rather than the name is what stops
  `internal.example.com -> 127.0.0.1` and decimal/hex IP encodings
* redirects followed MANUALLY, re-validating each hop, because a public URL is
  free to 302 to `http://localhost` and httpx would follow it happily
* response size and time bounded, so a hostile endpoint cannot exhaust memory
* no cookies, no auth headers, no credentials ever attached

Residual risk, stated plainly: there is a DNS-rebinding window between validation
and connection — the name could resolve to a public IP for our check and a private
one microseconds later. Closing it fully means pinning the socket to the validated
IP, which conflicts with TLS SNI/verification. For a single-user platform the
window is acceptable; it is documented rather than pretended away.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 3
REQUEST_TIMEOUT_S = 15.0
MAX_BYTES = 2_000_000
MAX_CHARS = 12_000
USER_AGENT = "core-chat/1.0 (+personal assistant; respects robots meta)"

# Tags whose text is chrome rather than content. Dropping them is the difference
# between a readable page and 3000 characters of nav links.
_STRIP_TAGS = ("script", "style", "noscript", "nav", "footer", "header",
               "aside", "form", "svg", "iframe", "template")


class WebFetchError(Exception):
    """Raised when a URL is unsafe, unreachable, or unusable."""


def _is_public(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _validate(url: str) -> str:
    """Reject anything that isn't a public http(s) URL. Returns the normalized URL."""
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise WebFetchError(
            f"only http and https URLs can be fetched (got {parsed.scheme or 'no scheme'!r})"
        )
    host = parsed.hostname
    if not host:
        raise WebFetchError("that URL has no hostname")

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise WebFetchError(f"could not resolve {host}") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise WebFetchError(f"could not resolve {host}")
    # EVERY address must be public: a name resolving to both a public and a private
    # address would otherwise be usable to reach the private one.
    private = [a for a in addresses if not _is_public(a)]
    if private:
        logger.warning("blocked SSRF attempt: %s resolved to %s", host, sorted(private))
        raise WebFetchError(
            f"{host} resolves to a private or local address, so it can't be fetched"
        )
    return urlunparse(parsed)


def _extract(html: str) -> tuple[str, str]:
    """(title, readable text) from an HTML document."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    return title, text


def fetch_page(url: str) -> tuple[str, str, str]:
    """Fetch a public page. Returns ``(final_url, title, text)``.

    Redirects are followed by hand so each hop is re-validated; httpx's automatic
    following would happily chase a 302 into the private network.
    """
    current = _validate(url)
    with httpx.Client(
        timeout=REQUEST_TIMEOUT_S,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            response = client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise WebFetchError("redirect without a destination")
                current = _validate(str(response.url.join(location)))
                continue

            if response.status_code >= 400:
                raise WebFetchError(f"the page returned HTTP {response.status_code}")

            content_type = response.headers.get("content-type", "")
            if not any(t in content_type for t in ("text/html", "text/plain", "application/xhtml")):
                raise WebFetchError(
                    f"that URL is {content_type.split(';')[0] or 'not text'}, not a readable page"
                )

            body = response.content[:MAX_BYTES]
            html = body.decode(response.encoding or "utf-8", errors="replace")
            title, text = ("", html.strip()) if "text/plain" in content_type else _extract(html)
            if not text:
                raise WebFetchError("that page has no readable text")
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "\n\n[truncated]"
            return str(response.url), title, text

    raise WebFetchError("too many redirects")
