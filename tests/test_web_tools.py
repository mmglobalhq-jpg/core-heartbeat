"""Web search and page fetching — mostly the SSRF boundary.

`fetch_url` makes an HTTP request chosen by *model output*, and model output is
shaped by page text, uploaded documents and chat messages. So the URL is
attacker-influenceable in principle, and this process is a high-value place to
make requests from: it sits on `heartbeat-net` with graph-rag (172.19.0.4), the
sidecar (172.19.0.2) and core-chat (172.19.0.6), can reach the host gateway, and
holds Supabase service-role keys.

Most of what follows therefore tests refusals, not successes. The happy path is
one test; the rest are the ways in.
"""

import socket

import httpx
import pytest

from services import web
from services.web import WebFetchError, _is_public, _validate, fetch_page
from tools.web_tools import WEB_TOOL_REGISTRY, run_web_tool


@pytest.fixture(autouse=True)
def _no_transport():
    web._transport = None if hasattr(web, "_transport") else None
    yield


def _resolves_to(monkeypatch, ip):
    """Force DNS resolution to a chosen address."""
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))],
    )


# --- address classification -------------------------------------------------


def test_private_and_local_addresses_are_not_public():
    for ip in (
        "127.0.0.1",        # loopback
        "10.0.0.5",         # private
        "172.19.0.4",       # graph-rag, on our own compose network
        "172.18.0.1",       # the host gateway
        "192.168.1.1",      # home LAN
        "169.254.169.254",  # cloud metadata
        "0.0.0.0",          # unspecified
        "::1",              # loopback v6
        "fd00::1",          # unique-local v6
    ):
        assert not _is_public(ip), f"{ip} must not be treated as public"


def test_public_addresses_are_allowed():
    for ip in ("8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"):
        assert _is_public(ip)


# --- URL validation ---------------------------------------------------------


def test_non_http_schemes_are_refused():
    for url in ("file:///etc/passwd", "gopher://x/", "data:text/html,hi", "ftp://x/"):
        with pytest.raises(WebFetchError):
            _validate(url)


def test_a_hostname_resolving_to_loopback_is_refused(monkeypatch):
    """The reason resolution is checked instead of the name: a public-looking
    hostname is free to point at 127.0.0.1."""
    _resolves_to(monkeypatch, "127.0.0.1")
    with pytest.raises(WebFetchError, match="private or local"):
        _validate("https://totally-normal.example.com/")


def test_internal_service_names_are_refused(monkeypatch):
    """`http://graph-rag:3000/...` is the realistic in-house SSRF target."""
    _resolves_to(monkeypatch, "172.19.0.4")
    with pytest.raises(WebFetchError, match="private or local"):
        _validate("http://graph-rag:3000/api/query")


def test_cloud_metadata_is_refused(monkeypatch):
    _resolves_to(monkeypatch, "169.254.169.254")
    with pytest.raises(WebFetchError):
        _validate("http://169.254.169.254/latest/meta-data/")


def test_a_name_resolving_to_both_public_and_private_is_refused(monkeypatch):
    """Allowing it would make the private address reachable by retry."""
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
    ])
    with pytest.raises(WebFetchError, match="private or local"):
        _validate("https://split-horizon.example.com/")


def test_a_public_url_validates(monkeypatch):
    _resolves_to(monkeypatch, "93.184.216.34")
    assert _validate("https://example.com/page").startswith("https://example.com")


# --- redirects --------------------------------------------------------------


def test_a_redirect_into_the_private_network_is_refused(monkeypatch):
    """A public URL may 302 to localhost. httpx would follow it, so redirects are
    followed by hand and each hop re-validated."""
    hosts = {"public.example.com": "93.184.216.34", "internal.example.com": "127.0.0.1"}
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (hosts[host], 443))
    ])

    def handler(request):
        return httpx.Response(302, headers={"location": "https://internal.example.com/secrets"})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{**kw, "transport": httpx.MockTransport(handler)}),
    )
    with pytest.raises(WebFetchError, match="private or local"):
        fetch_page("https://public.example.com/start")


# --- content handling -------------------------------------------------------


def _serve(monkeypatch, response):
    _resolves_to(monkeypatch, "93.184.216.34")
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(**{
            **kw, "transport": httpx.MockTransport(lambda request: response)
        }),
    )


def test_a_normal_page_returns_readable_text(monkeypatch):
    html = ("<html><head><title>Roster</title><style>x{}</style></head>"
            "<body><nav>menu menu</nav><p>Broderick Jones 6-5 310</p>"
            "<script>evil()</script><footer>legal</footer></body></html>")
    _serve(monkeypatch, httpx.Response(200, html=html))
    final_url, title, text = fetch_page("https://example.com/roster")
    assert title == "Roster"
    assert "Broderick Jones 6-5 310" in text
    # Chrome and scripts are stripped, or the useful text drowns in nav links.
    for noise in ("menu menu", "evil()", "legal", "x{}"):
        assert noise not in text


def test_non_text_content_is_refused(monkeypatch):
    _serve(monkeypatch, httpx.Response(200, content=b"%PDF-1.4",
                                       headers={"content-type": "application/pdf"}))
    with pytest.raises(WebFetchError, match="not a readable page"):
        fetch_page("https://example.com/file.pdf")


def test_an_error_page_is_reported(monkeypatch):
    _serve(monkeypatch, httpx.Response(404, html="<html><body>gone</body></html>"))
    with pytest.raises(WebFetchError, match="404"):
        fetch_page("https://example.com/missing")


def test_long_pages_are_truncated(monkeypatch):
    _serve(monkeypatch, httpx.Response(200, html="<html><body>" + ("word " * 60000) + "</body></html>"))
    _, _, text = fetch_page("https://example.com/long")
    assert len(text) <= web.MAX_CHARS + 32
    assert text.endswith("[truncated]")


# --- tool wrappers ----------------------------------------------------------


def test_registry_exposes_the_search_read_loop():
    # find_sources (list pages) + fetch_url (read one) is the two-step that gets a
    # specific detail off a page; search_web answers in prose when that is enough.
    assert WEB_TOOL_REGISTRY == {"search_web", "find_sources", "fetch_url"}


def test_fetch_url_returns_an_error_string_rather_than_raising(monkeypatch):
    """Every run_*_tool converts failure to text so the graph keeps running."""
    _resolves_to(monkeypatch, "127.0.0.1")
    out = run_web_tool("fetch_url", "u1", {"url": "http://localhost:8000/health"})
    assert out.startswith("error:") and "private or local" in out


def test_missing_arguments_are_handled():
    assert run_web_tool("fetch_url", "u1", {}).startswith("error:")
    assert run_web_tool("search_web", "u1", {"query": "  "}).startswith("error:")


def test_search_without_a_key_is_reported_not_crashed(monkeypatch):
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert "not configured" in run_web_tool("search_web", "u1", {"query": "hello"})


def test_unknown_tool_name():
    assert run_web_tool("nope", "u1", {}).startswith("error: unknown tool")


# --- find_sources: the reading list ----------------------------------------
#
# The defect behind this tool: grounding answered a flight question with "check a
# booking site" and the composer relayed the hedge. A summary is not always the
# answer, and the fix is to go and read a page.


def _sources(monkeypatch, results):
    import tools.web_tools as wt
    monkeypatch.setattr(wt, "search_web_raw", lambda q, max_results=10: results[:max_results])


def test_find_sources_lists_pages_with_their_sites(monkeypatch):
    _sources(monkeypatch, [
        {"url": "https://a.example/x", "title": "a.example", "snippet": "", "source": "a.example"},
        {"url": "https://b.example/y", "title": "b.example", "snippet": "", "source": "b.example"},
    ])
    out = run_web_tool("find_sources", "u1", {"query": "memphis weather"})
    assert "1. a.example — https://a.example/x" in out
    assert "2. b.example — https://b.example/y" in out


def test_find_sources_tells_the_model_to_read_and_that_links_expire(monkeypatch):
    _sources(monkeypatch, [{"url": "https://a.example/x", "title": "t", "snippet": "", "source": "a.example"}])
    out = run_web_tool("find_sources", "u1", {"query": "q"})
    assert "fetch_url" in out
    # grounding redirect links 429 once stale, so they must be read this turn
    assert "expire" in out


def test_find_sources_with_no_results_forbids_answering_from_memory(monkeypatch):
    _sources(monkeypatch, [])
    out = run_web_tool("find_sources", "u1", {"query": "q"})
    assert "No sources found" in out
    assert "rather than answering from memory" in out


def test_find_sources_bounds_the_result_count(monkeypatch):
    import tools.web_tools as wt
    seen = {}

    def fake(q, max_results=10):
        seen["n"] = max_results
        return []
    monkeypatch.setattr(wt, "search_web_raw", fake)

    run_web_tool("find_sources", "u1", {"query": "q", "max_results": 500})
    assert seen["n"] == wt.MAX_SOURCE_COUNT
    run_web_tool("find_sources", "u1", {"query": "q", "max_results": 0})
    assert seen["n"] == 1
    run_web_tool("find_sources", "u1", {"query": "q", "max_results": "not a number"})
    assert seen["n"] == wt.DEFAULT_SOURCE_COUNT


def test_find_sources_refuses_an_empty_query():
    assert run_web_tool("find_sources", "u1", {"query": "  "}).startswith("error:")


def test_find_sources_does_not_present_its_order_as_a_ranking(monkeypatch):
    """Observed live 2026-08-25: asked for zoo ticket prices, grounding returned
    facebook.com first and memphiszoo.org fourth, and fetching #1 yielded 81
    characters of nothing. The order is chunk order, not relevance, so the tool
    must not imply otherwise."""
    _sources(monkeypatch, [
        {"url": "https://x/1", "title": "", "snippet": "", "source": "facebook.com"},
        {"url": "https://x/2", "title": "", "snippet": "", "source": "memphiszoo.org"},
    ])
    out = run_web_tool("find_sources", "u1", {"query": "Memphis Zoo ticket prices"})
    assert "NOT A RANKING" in out
    assert "authoritative" in out
    assert "read the next source" in out
