"""Open-Meteo: free weather forecasts, no API key, no account.

Two requests: the geocoder (only when a city is set: sends the city's name) and
the forecast (sends coordinates rounded to two decimals, about 1 km). Each call
goes in the encrypted audit log as its kind and whether it worked, never the
city or coordinates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date

import httpx

log = logging.getLogger(__name__)

GEOCODER = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FAHRENHEIT_COUNTRIES = {"US", "PR", "GU", "VI", "AS", "MP", "UM", "LR", "FM", "MH", "PW", "BS", "BZ", "KY"}


class WeatherError(RuntimeError):
    def __init__(self, message: str, spoken: str = "I can't get the weather right now.") -> None:
        super().__init__(message)
        self.spoken = spoken


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    admin1: str = ""            # state or region
    country: str = ""
    country_code: str = ""

    @property
    def spoken(self) -> str:
        """'Tampa, Florida, United States'."""
        parts = [self.name] + [p for p in (self.admin1, self.country) if p and p != self.name]
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Place":
        return cls(**{k: d[k] for k in ("name", "lat", "lon", "admin1", "country", "country_code") if k in d})


@dataclass
class Day:
    day: date
    code: int
    high: float
    low: float
    rain_chance: int | None


@dataclass
class Forecast:
    place: Place
    fahrenheit: bool
    temp: float
    feels_like: float
    code: int
    wind: float
    days: list[Day] = field(default_factory=list)

    def on(self, d: date) -> Day | None:
        return next((x for x in self.days if x.day == d), None)


def uses_fahrenheit(place: Place, units: str = "auto") -> bool:
    if units in ("fahrenheit", "f"):
        return True
    if units in ("celsius", "c"):
        return False
    return place.country_code.upper() in FAHRENHEIT_COUNTRIES


class OpenMeteo:
    def __init__(self, client: httpx.Client | None = None, timeout_s: float = 10.0, cache_s: float = 600,
                 audit=None, clock=time.monotonic) -> None:
        self._client = client or httpx.Client(timeout=timeout_s)
        self.cache_s = cache_s
        self.audit = audit
        self.clock = clock
        self._cache: dict[tuple, tuple[float, Forecast]] = {}
        self._lock = threading.Lock()
        self.last_ms: float | None = None

    def find(self, name: str, count: int = 5) -> list[Place]:
        """Places called ``name``. "Austin, Texas" and "Austin Texas" narrow by region or country."""
        name = " ".join(name.replace(",", " , ").split()).strip(" ,.")
        if not name:
            return []
        if "," in name:
            head, rest = name.split(",", 1)
            return self._narrow(self._search(head.strip(), max(count, 10)), rest)[:count]
        places = self._search(name, count)
        words = name.split()
        # "Austin Texas": nothing called that, so try "Austin" narrowed by "Texas".
        for cut in range(len(words) - 1, 0, -1):
            if places:
                break
            places = self._narrow(self._search(" ".join(words[:cut]), 10), " ".join(words[cut:]))[:count]
        return places

    def forecast(self, place: Place, fahrenheit: bool) -> Forecast:
        key = (round(place.lat, 2), round(place.lon, 2), fahrenheit)
        with self._lock:
            hit = self._cache.get(key)
            if hit and self.clock() - hit[0] < self.cache_s:
                return hit[1]
        params = {
            "latitude": f"{place.lat:.2f}", "longitude": f"{place.lon:.2f}",
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto", "forecast_days": "7",
            "temperature_unit": "fahrenheit" if fahrenheit else "celsius",
            "wind_speed_unit": "mph" if fahrenheit else "kmh",
        }
        data = self._get(FORECAST, params, "weather_forecast")
        try:
            cur, daily = data["current"], data["daily"]
            days = [Day(date.fromisoformat(t), int(c), float(hi), float(lo), None if p is None else int(p))
                    for t, c, hi, lo, p in zip(daily["time"], daily["weather_code"], daily["temperature_2m_max"],
                                               daily["temperature_2m_min"], daily["precipitation_probability_max"])
                    if hi is not None and lo is not None]
            fc = Forecast(place, fahrenheit, float(cur["temperature_2m"]), float(cur["apparent_temperature"]),
                          int(cur["weather_code"]), float(cur.get("wind_speed_10m") or 0), days)
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherError(f"Unexpected forecast format: {exc}") from exc
        with self._lock:
            self._cache[key] = (self.clock(), fc)
        return fc

    # ---- internals ---------------------------------------------------------------------
    def _search(self, name: str, count: int) -> list[Place]:
        data = self._get(GEOCODER, {"name": name, "count": str(count), "language": "en", "format": "json"},
                         "weather_geocode")
        out = []
        for r in data.get("results") or []:
            try:
                out.append(Place(r["name"], float(r["latitude"]), float(r["longitude"]), r.get("admin1", ""),
                                 r.get("country", ""), r.get("country_code", "")))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _narrow(places: list[Place], qualifier: str) -> list[Place]:
        q = " ".join(qualifier.lower().replace(",", " ").split())
        if not q:
            return places
        words = set(q.split())
        abbrev = {"usa": "us", "uk": "gb", "america": "us", "england": "gb", "britain": "gb"}

        def fits(p: Place) -> bool:
            region = f"{p.admin1} {p.country}".lower()
            return q in region or words <= set(region.split()) or \
                any(abbrev.get(w, w) == p.country_code.lower() for w in words) or \
                _STATES.get(q.upper()) == p.admin1
        return [p for p in places if fits(p)]

    def _get(self, url: str, params: dict, kind: str) -> dict:
        t = time.perf_counter()
        ok, error = False, None
        try:
            r = self._client.get(url, params=params)
            if r.status_code == 429:
                raise WeatherError("Open-Meteo rate limit", "The weather service is busy. Try again in a minute.")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise WeatherError(f"Open-Meteo error: {data.get('reason')}")
            ok = True
            return data
        except httpx.TimeoutException as exc:
            error = "timeout"
            raise WeatherError("Open-Meteo timed out", "The weather service didn't answer in time.") from exc
        except httpx.HTTPError as exc:
            error = type(exc).__name__
            raise WeatherError(f"Open-Meteo request failed: {exc}",
                               "I can't reach the weather service. Is the internet down?") from exc
        except ValueError as exc:
            error = "bad response"
            raise WeatherError("Open-Meteo sent something that isn't JSON") from exc
        except WeatherError as exc:
            error = str(exc)
            raise
        finally:
            self.last_ms = (time.perf_counter() - t) * 1000
            if self.audit is not None:
                try:
                    # Never the city or coordinates.
                    self.audit.record(kind, ok=ok, error=error, ms=round(self.last_ms))
                except Exception:
                    log.exception("Couldn't write the audit log")


# US state abbreviations, so "Austin, TX" works.
_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


# WMO weather interpretation codes, as Open-Meteo documents them.
CONDITIONS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy with frost",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}
WET = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}


def condition(code: int) -> str:
    return CONDITIONS.get(code, "unsettled")
