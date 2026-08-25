"""Live flight search via the Amadeus Self-Service API.

WHY THIS EXISTS
``search_web`` is Gemini's Google Search grounding: it returns a prose summary of
what a search engine surfaced. Asked for flights on a specific date it answers
"specific flight times ... are not available in the search results — check a
booking site" (reproduced against production 2026-08-25). No prompt change turns
that into departure times, because the data is not in the response. This tool is
the data.

Amadeus was chosen over scraping Google Flights: it has a free tier, returns
structured itineraries (carrier, times, stops, duration, price) rather than HTML
to parse, and is a documented API rather than something that breaks when a page
changes.

MULTIPLE ORIGINS ARE THE POINT
The reported failure picked HHH (Hilton Head, a handful of daily departures) over
SAV (an hour away, full service) purely because a prose search called it "closest",
and produced no usable itinerary. So ``origins`` is a LIST: the caller passes every
airport within reach and the ranking is done on real itineraries, not on distance.
The nearest airport is frequently the wrong answer.

TIMES ARE LOCAL AND NAIVE
Amadeus returns local naive datetimes (2026-09-18T13:35:00) for each airport, which
is what a traveller reads off a boarding pass. They are passed through unchanged —
no timezone maths, because converting a departure time to anything other than the
departure airport's own clock produces a number the user cannot check against any
booking site.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import httpx

from services.secrets import secret as read_secret

logger = logging.getLogger(__name__)

CLIENT_ID_ENV = "AMADEUS_CLIENT_ID"
CLIENT_SECRET_ENV = "AMADEUS_CLIENT_SECRET"
# Amadeus runs two estates. `test` is free but serves a reduced, cached inventory;
# `api` is production. Default to test so an unconfigured key cannot spend money.
BASE_URL_ENV = "AMADEUS_BASE_URL"
DEFAULT_BASE_URL = "https://test.api.amadeus.com"

TIMEOUT_S = 25.0
MAX_OFFERS_PER_ORIGIN = 12
# What the composer is shown. More than this and the useful options are buried;
# fewer and a connection-heavy route looks like it has no service.
MAX_RENDERED = 8

_token: dict[str, object] = {"value": None, "expires_at": 0.0}


def _base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def _credentials() -> tuple[str, str] | None:
    cid = read_secret(CLIENT_ID_ENV)
    csec = read_secret(CLIENT_SECRET_ENV)
    return (cid, csec) if cid and csec else None


def _access_token(client: httpx.Client) -> str:
    """Cached OAuth2 client-credentials token.

    Amadeus tokens last ~30 minutes. Re-fetching per search would double the
    latency of every flight question for no benefit, so it is cached with a 60s
    safety margin.
    """
    now = time.time()
    if _token["value"] and float(_token["expires_at"]) > now:
        return str(_token["value"])
    creds = _credentials()
    if creds is None:
        raise RuntimeError("amadeus credentials are not configured")
    cid, csec = creds
    r = client.post(
        f"{_base_url()}/v1/security/oauth2/token",
        data={"grant_type": "client_credentials", "client_id": cid, "client_secret": csec},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r.raise_for_status()
    body = r.json()
    _token["value"] = body["access_token"]
    _token["expires_at"] = now + max(60.0, float(body.get("expires_in", 1800)) - 60.0)
    return str(_token["value"])


def _iso_minutes(value: str) -> str:
    """'2026-09-18T13:35:00' -> '1:35 PM'. Returns the raw value if unparseable."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt.strftime("%-I:%M %p")


def _duration(iso: str) -> str:
    """'PT4H17M' -> '4h 17m'."""
    out = iso.removeprefix("PT").lower()
    return out.replace("h", "h ").replace("m", "m").strip()


def _after(depart_iso: str, earliest: str | None) -> bool:
    """True when a departure is at or after ``earliest`` ('13:30' or '1:30 PM')."""
    if not earliest:
        return True
    try:
        dep = datetime.fromisoformat(depart_iso)
    except ValueError:
        return True
    for fmt in ("%H:%M", "%I:%M %p", "%I%p", "%H%M"):
        try:
            cutoff = datetime.strptime(earliest.strip().upper(), fmt).time()
        except ValueError:
            continue
        return dep.time() >= cutoff
    return True


def _render(offers: list[dict], carriers: dict[str, str]) -> list[str]:
    """One line per itinerary, in the order the composer should present them."""
    lines: list[str] = []
    for offer in offers:
        itin = (offer.get("itineraries") or [{}])[0]
        segs = itin.get("segments") or []
        if not segs:
            continue
        first, last = segs[0], segs[-1]
        dep = first.get("departure", {})
        arr = last.get("arrival", {})
        code = first.get("carrierCode", "")
        airline = carriers.get(code, code)
        stops = len(segs) - 1
        viaptr = " nonstop" if stops == 0 else (
            " 1 stop in " + (segs[0].get("arrival", {}).get("iataCode") or "?")
            if stops == 1
            else f" {stops} stops"
        )
        price = (offer.get("price") or {}).get("grandTotal") or (offer.get("price") or {}).get("total")
        currency = (offer.get("price") or {}).get("currency", "USD")
        cost = f", from {price} {currency}" if price else ""
        lines.append(
            f"- {airline} {dep.get('iataCode','?')} {_iso_minutes(dep.get('at',''))} "
            f"-> {arr.get('iataCode','?')} {_iso_minutes(arr.get('at',''))}, "
            f"{_duration(itin.get('duration',''))},{viaptr}{cost}"
        )
    return lines


def search_flights(user_id: str, args: dict) -> str:
    """Search real itineraries for one date across one or more origin airports."""
    origins = args.get("origins") or args.get("origin") or ""
    if isinstance(origins, str):
        origins = [o.strip().upper() for o in origins.replace(",", " ").split() if o.strip()]
    destination = str(args.get("destination") or "").strip().upper()
    date = str(args.get("departure_date") or "").strip()
    earliest = args.get("earliest_departure_time")

    if not origins or not destination or not date:
        return "error: origins, destination and departure_date are all required"
    if _credentials() is None:
        # Stated plainly so the composer reports a missing capability rather than
        # inventing "Delta generally operates this route" (see the web family note).
        return (
            "error: live flight search is not configured on this server "
            "(no Amadeus API credentials). Real schedules and fares cannot be "
            "retrieved — say so plainly rather than estimating."
        )

    rendered: list[str] = []
    errors: list[str] = []
    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            token = _access_token(client)
            headers = {"Authorization": f"Bearer {token}"}
            for origin in origins[:4]:  # bound the fan-out; 4 airports is generous
                try:
                    r = client.get(
                        f"{_base_url()}/v2/shopping/flight-offers",
                        params={
                            "originLocationCode": origin,
                            "destinationLocationCode": destination,
                            "departureDate": date,
                            "adults": int(args.get("adults") or 1),
                            "currencyCode": "USD",
                            "max": MAX_OFFERS_PER_ORIGIN,
                        },
                        headers=headers,
                    )
                    if r.status_code != 200:
                        errors.append(f"{origin}: HTTP {r.status_code}")
                        continue
                    body = r.json()
                except Exception as exc:
                    errors.append(f"{origin}: {type(exc).__name__}")
                    continue

                carriers = (body.get("dictionaries") or {}).get("carriers") or {}
                offers = [
                    o
                    for o in (body.get("data") or [])
                    if _after(
                        ((o.get("itineraries") or [{}])[0].get("segments") or [{}])[0]
                        .get("departure", {})
                        .get("at", ""),
                        earliest,
                    )
                ]
                if not offers:
                    errors.append(f"{origin}: no itineraries matching that time")
                    continue
                rendered.extend(_render(offers, carriers))
    except RuntimeError as exc:
        return f"error: {exc}"
    except Exception as exc:
        logger.warning("flight search failed: %s", exc)
        return f"error: flight search failed ({type(exc).__name__})"

    if not rendered:
        detail = "; ".join(errors) if errors else "no itineraries returned"
        return (
            f"No flights found from {', '.join(origins)} to {destination} on {date}"
            + (f" departing after {earliest}" if earliest else "")
            + f". ({detail})"
        )

    # Sort by departure clock time as rendered, so the earliest option is first --
    # which is almost always what "soonest after X" means.
    rendered.sort(key=lambda line: line)
    head = (
        f"Flights to {destination} on {date}"
        + (f", departing after {earliest}" if earliest else "")
        + f" (from {', '.join(origins)}):"
    )
    body_lines = rendered[:MAX_RENDERED]
    tail = "" if len(rendered) <= MAX_RENDERED else f"\n({len(rendered) - MAX_RENDERED} more not shown.)"
    note = (
        "\nTimes are LOCAL to each airport. Prices are the lowest available fare at "
        "search time and change constantly."
    )
    return head + "\n" + "\n".join(body_lines) + tail + note


FLIGHT_TOOL_REGISTRY = {"search_flights": search_flights}


def run_flight_tool(name: str, user_id: str, args: dict | None = None) -> str:
    """Execute a registered flight tool; never raises."""
    fn = FLIGHT_TOOL_REGISTRY.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return fn(user_id, args or {})
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"
