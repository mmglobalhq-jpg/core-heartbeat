"""Tests for search_flights — the tool that answers what search_web could not.

The bug this feature exists for: asked for flights, search_web returned "specific
flight times ... are not available in the search results" and the composer answered
with drive times instead. So the two behaviours asserted hardest here are (a) real
itineraries render as options, and (b) every failure path says plainly that flights
could not be retrieved, because an unclear failure is what produced the original
misleading answer.
"""
import httpx
import pytest

import tools.flights as fl


@pytest.fixture(autouse=True)
def _reset_token():
    fl._token["value"] = None
    fl._token["expires_at"] = 0.0


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv(fl.CLIENT_ID_ENV, "id")
    monkeypatch.setenv(fl.CLIENT_SECRET_ENV, "secret")


OFFER = {
    "itineraries": [
        {
            "duration": "PT4H17M",
            "segments": [
                {
                    "carrierCode": "DL",
                    "departure": {"iataCode": "SAV", "at": "2026-09-18T13:35:00"},
                    "arrival": {"iataCode": "ATL", "at": "2026-09-18T14:40:00"},
                },
                {
                    "carrierCode": "DL",
                    "departure": {"iataCode": "ATL", "at": "2026-09-18T16:10:00"},
                    "arrival": {"iataCode": "MEM", "at": "2026-09-18T16:52:00"},
                },
            ],
        }
    ],
    "price": {"grandTotal": "312.40", "currency": "USD"},
}


def _transport(offers, *, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1800})
        return httpx.Response(status, json={"data": offers, "dictionaries": {"carriers": {"DL": "DELTA AIR LINES"}}})
    return httpx.MockTransport(handler)


@pytest.fixture
def mock_api(monkeypatch):
    def install(offers, *, status=200):
        real = httpx.Client
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: real(**{**kw, "transport": _transport(offers, status=status)})
        )
    return install


# --- the unconfigured path (true on this host until a key is installed) -----


def test_missing_credentials_says_so_instead_of_failing_vaguely(monkeypatch):
    monkeypatch.delenv(fl.CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(fl.CLIENT_SECRET_ENV, raising=False)
    out = fl.search_flights("u", {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18"})
    assert out.startswith("error:")
    assert "not configured" in out
    # The composer must be told to report it, not to estimate around it.
    assert "say so plainly rather than estimating" in out


# --- happy path -------------------------------------------------------------


def test_renders_a_real_itinerary_as_one_option_line(creds, mock_api):
    mock_api([OFFER])
    out = fl.search_flights("u", {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18"})
    assert "DELTA AIR LINES" in out
    assert "SAV 1:35 PM" in out and "MEM 4:52 PM" in out
    assert "4h 17m" in out
    assert "1 stop in ATL" in out
    assert "312.40 USD" in out


def test_times_are_labelled_local_and_prices_as_volatile(creds, mock_api):
    mock_api([OFFER])
    out = fl.search_flights("u", {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18"})
    assert "LOCAL to each airport" in out
    assert "change constantly" in out


# --- multiple origins: the HHH-vs-SAV failure ------------------------------


def test_several_origins_are_all_searched(creds, mock_api, monkeypatch):
    seen: list[str] = []
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1800})
        seen.append(request.url.params["originLocationCode"])
        return httpx.Response(200, json={"data": [OFFER], "dictionaries": {"carriers": {"DL": "DELTA"}}})

    monkeypatch.setattr(httpx, "Client", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)}))
    fl.search_flights("u", {"origins": "SAV, CHS HHH", "destination": "MEM", "departure_date": "2026-09-18"})
    assert seen == ["SAV", "CHS", "HHH"]


def test_origin_fan_out_is_bounded(creds, monkeypatch):
    seen: list[str] = []
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1800})
        seen.append(request.url.params["originLocationCode"])
        return httpx.Response(200, json={"data": [], "dictionaries": {}})

    monkeypatch.setattr(httpx, "Client", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(handler)}))
    fl.search_flights("u", {"origins": "A B C D E F", "destination": "MEM", "departure_date": "2026-09-18"})
    assert len(seen) == 4


# --- the earliest-departure filter -----------------------------------------


@pytest.mark.parametrize("earliest", ["13:00", "1:00 PM"])
def test_a_flight_after_the_cutoff_is_kept(creds, mock_api, earliest):
    mock_api([OFFER])
    out = fl.search_flights(
        "u",
        {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18",
         "earliest_departure_time": earliest},
    )
    assert "DELTA" in out


def test_a_flight_before_the_cutoff_is_dropped(creds, mock_api):
    mock_api([OFFER])
    out = fl.search_flights(
        "u",
        {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18",
         "earliest_departure_time": "3:00 PM"},
    )
    assert "No flights found" in out
    assert "departing after 3:00 PM" in out


# --- failure paths ----------------------------------------------------------


def test_an_api_error_reports_no_flights_rather_than_pretending(creds, mock_api):
    mock_api([], status=500)
    out = fl.search_flights("u", {"origins": "SAV", "destination": "MEM", "departure_date": "2026-09-18"})
    assert "No flights found" in out
    assert "HTTP 500" in out


def test_missing_arguments_are_refused(creds):
    assert fl.search_flights("u", {"origins": "SAV"}).startswith("error:")


def test_the_dispatcher_never_raises():
    assert fl.run_flight_tool("nope", "u", {}).startswith("error: unknown tool")


# --- formatting helpers -----------------------------------------------------


@pytest.mark.parametrize(
    "iso,expected", [("PT4H17M", "4h 17m"), ("PT55M", "55m"), ("PT2H", "2h")]
)
def test_duration_rendering(iso, expected):
    assert fl._duration(iso) == expected


def test_unparseable_time_is_passed_through_not_crashed():
    assert fl._iso_minutes("not-a-time") == "not-a-time"
