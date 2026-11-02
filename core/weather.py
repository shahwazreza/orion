"""Weather by voice: "what's the weather?", "will it rain tomorrow?", "set my city to Tampa".

Every number Orion says comes from the forecast, put into words by code. The
model never sees a weather question, so it can't invent a forecast. No city is
built in: each person sets theirs (stored encrypted); guests and anyone without
one get the owner's.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Callable

from core.dialog import Dialog, Heard, yes_no
from core.schedule import Schedule
from core.when import WEEKDAYS, local_now, parse_span
from storage.users import User
from tools.weather import WET, Forecast, OpenMeteo, Place, WeatherError, condition, uses_fahrenheit

log = logging.getLogger(__name__)

SETTING = "weather_place"
NO_CITY = "I don't know your city yet. Say 'set my city to', then the city."

_SET_CITY = re.compile(
    r"\b(?:set|change|update|make|switch|put)\s+(?:my\s+|the\s+)?(?:weather\s+|home\s+)?(?:city|location|town|place)"
    r"(?:\s+for (?:the )?weather)?\s+(?:to|as|is)\s+(?P<city>[a-z][\w .,'-]{1,60}?)\s*[.!?]*$|"
    r"\buse\s+(?P<city2>[a-z][\w .,'-]{1,60}?)\s+for (?:the |my )?weather\b|"
    r"\bmy (?:weather |home )?(?:city|location) (?:is|should be)\s+(?P<city3>[a-z][\w .,'-]{1,60}?)\s*[.!?]*$", re.I)
_WHICH_CITY = re.compile(r"\b(?:what|which) (?:city|location|place|town) (?:are you using|do you use|is (?:set|my "
                         r"(?:weather )?(?:city|location)))\b|\bwhat(?:'s| is) my (?:weather )?(?:city|location)\b|"
                         r"\bwhere (?:do you|are you) (?:get|getting) (?:the|my) weather\b", re.I)
_WEATHER = re.compile(
    r"\bweather\b|\bforecast\b|\b(?:is it|will it|is it going to|it going to|gonna|going to) (?:rain|snow|storm|"
    r"pour|be (?:hot|cold|warm|chilly|sunny|nice|windy|wet|dry|rainy|cloudy))\b|\bumbrella\b|"
    r"\bhow (?:hot|cold|warm|chilly) (?:is it|will it be|is it going to be)\b|\btemperature\b|"
    r"\bchance of (?:rain|snow|showers)\b|\b(?:need|bring|take|wear) a (?:jacket|coat|sweater|hoodie)\b|"
    r"\bany rain\b|\bdegrees (?:outside|out)\b|\bwhat'?s it like outside\b", re.I)
_NOT_FORECAST = re.compile(r"\b(?:why|explain|how does|how do|what causes|claude|history|climate|whenever|"
                           r"remember|remind|forecast(?:ing)? model)\b", re.I)
_RAIN = re.compile(r"\brain|\bumbrella\b|\bwet\b|\bshowers?\b|\bpour\b|\bstorm", re.I)
_SNOW = re.compile(r"\bsnow", re.I)
_TEMP = re.compile(r"\bhot\b|\bcold\b|\bwarm\b|\bchilly\b|\btemperature\b|\bdegrees\b|\bjacket\b|\bcoat\b|"
                   r"\bsweater\b|\bhoodie\b", re.I)
_PLACE = re.compile(r"\bin\s+(?P<place>(?!the (?:morning|afternoon|evening|week)|an? (?:hour|minute|bit))"
                    r"[a-z][a-z .,'-]{1,40}?)(?:\s+(?:today|tomorrow|tonight|this|next|on|right now|now|at|for|over|"
                    r"later)\b.*)?\s*[?.!]*$", re.I)


def is_weather_request(text: str) -> bool:
    return bool(_WEATHER.search(text)) and not _NOT_FORECAST.search(text)


def a_percent(n: int) -> str:
    """'an 83 percent', 'a 40 percent'."""
    return ("an " if str(n).startswith("8") or n in (11, 18) else "a ") + f"{n} percent"


def degrees(x: float) -> str:
    n = round(x)
    return f"minus {abs(n)}" if n < 0 else str(n)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _day_word(d: date, today: date) -> str:
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    return WEEKDAYS[d.weekday()].title()


class Weather:
    """Weather for a person: their city (or the owner's), the forecast, and the words for it."""

    def __init__(self, client: OpenMeteo, schedule: Schedule, owner: Callable[[], User | None],
                 units: str = "auto", now: Callable[[], datetime] = local_now) -> None:
        self.client = client
        self.schedule = schedule
        self.owner = owner
        self.units = units
        self.now = now

    def place_for(self, user: User | None) -> Place | None:
        for who in (user, self.owner()):
            if who is None:
                continue
            raw = self.schedule.setting(who.id, SETTING)
            if raw:
                try:
                    return Place.from_dict(json.loads(raw))
                except (ValueError, TypeError):
                    log.warning("Stored weather city is unreadable")
        return None

    def save_place(self, user: User, place: Place) -> None:
        stored = Place(place.name, round(place.lat, 2), round(place.lon, 2), place.admin1, place.country,
                       place.country_code)
        self.schedule.set_setting(user.id, SETTING, json.dumps(stored.to_dict()))

    def forecast(self, place: Place) -> Forecast:
        return self.client.forecast(place, uses_fahrenheit(place, self.units))

    # ---- words ----------------------------------------------------------------------------
    def today_line(self, fc: Forecast, with_place: bool = True) -> str:
        """'In Tampa it's 72 degrees and overcast. Today, heavy rain, a high of 75 and a low of 67, with an 83
        percent chance of rain.'"""
        where = f"In {fc.place.name} it's" if with_place else "It's"
        line = f"{where} {degrees(fc.temp)} degrees and {condition(fc.code)}"
        if abs(fc.feels_like - fc.temp) >= 5:
            line += f", but it feels like {degrees(fc.feels_like)}"
        line += "."
        today = fc.on(self.now().date())
        if today is not None:
            line += f" Today, {self._day_words(today)}."
        return line

    def _day_words(self, d) -> str:
        words = f"{condition(d.code)}, a high of {degrees(d.high)} and a low of {degrees(d.low)}"
        if d.rain_chance is not None and (d.rain_chance >= 20 or d.code in WET):
            words += f", with {a_percent(d.rain_chance)} chance of rain"
        return words

    def answer(self, text: str, fc: Forecast) -> str:
        now = self.now()
        today = now.date()
        span = parse_span(text, now)
        days = [today]
        if span is not None:
            first, last = span.start.date(), (span.end - timedelta(seconds=1)).date()
            days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        if days and days[0] > today + timedelta(days=6):
            return "I can only see a week ahead."
        found = [fc.on(d) for d in days if fc.on(d) is not None]
        if not found:
            return "I can only see a week ahead."
        where = fc.place.name
        if _SNOW.search(text) or _RAIN.search(text):
            d = found[0]
            snow = bool(_SNOW.search(text))
            wet = d.code in ({71, 73, 75, 77, 85, 86} if snow else WET)
            chance = d.rain_chance or 0
            what = "snow" if snow else "rain"
            day = _day_word(d.day, today)
            if snow:
                return (f"Yes, {condition(d.code)} {day} in {where}." if wet else
                        f"No snow in the forecast for {day} in {where}.")
            umbrella = bool(re.search(r"umbrella", text, re.I))
            if chance >= 50 or wet and chance >= 30:
                lead = "Yes, take an umbrella" if umbrella else "Probably"
            elif chance >= 25:
                lead = "Maybe, so an umbrella wouldn't hurt" if umbrella else "Possibly"
            else:
                lead = "Probably not" if umbrella else "Unlikely"
            return f"{lead}: there's {a_percent(chance)} chance of {what} {day} in {where}."
        if len(found) > 1:
            parts = [f"{_day_word(d.day, today)} {self._day_words(d)}" for d in found[:7]]
            label = span.label if span else "the next few days"
            return f"{_cap(label)} in {where}: " + "; ".join(parts) + "."
        d = found[0]
        if d.day == today:
            if _TEMP.search(text):
                line = f"It's {degrees(fc.temp)} degrees in {where} right now"
                if abs(fc.feels_like - fc.temp) >= 3:
                    line += f", feels like {degrees(fc.feels_like)}"
                return line + f". Today's high is {degrees(d.high)} and the low {degrees(d.low)}."
            return self.today_line(fc)
        return f"{_cap(_day_word(d.day, today))} in {where}: {self._day_words(d)}."


class WeatherCommands:
    def __init__(self, weather: Weather, remember: Callable[[User | None, str, str], None] | None = None) -> None:
        self.weather = weather
        self.remember = remember or (lambda user, q, a: None)

    def handle(self, heard: Heard, dialog: Dialog, user: User | None) -> bool:
        text = heard.text.strip()
        if not text:
            return False
        m = _SET_CITY.search(text)
        if m:
            city = (m.group("city") or m.group("city2") or m.group("city3") or "").strip(" .,!?")
            self._set_city(city, dialog, user)
            return True
        if _WHICH_CITY.search(text):
            place = self.weather.place_for(user)
            dialog.say(f"I'm using {place.spoken}." if place else NO_CITY)
            return True
        if not is_weather_request(text):
            return False
        reply = self._report(text, user)
        dialog.say(reply)
        self.remember(user, text, reply)
        return True

    def _report(self, text: str, user: User | None) -> str:
        try:
            other = self._other_place(text)
            if other == "":
                return "I couldn't find that place."
            place = other or self.weather.place_for(user)
            if place is None:
                return NO_CITY
            return self.weather.answer(text, self.weather.forecast(place))
        except WeatherError as exc:
            log.info("Weather failed: %s", exc)
            return exc.spoken

    def _other_place(self, text: str) -> Place | None | str:
        """A place named in the question ("weather in London"): looked up, not saved. "" if not found."""
        m = _PLACE.search(text)
        if not m:
            return None
        name = m.group("place").strip(" ,.")
        if re.fullmatch(r"(?:the )?(?:morning|afternoon|evening|night|week|weekend|future|area|here|my area|"
                        r"town|my city|the city)", name, re.I):
            return None
        places = self.weather.client.find(name, count=1)
        return places[0] if places else ""

    def _set_city(self, city: str, dialog: Dialog, user: User | None) -> None:
        if user is None:
            dialog.say("I only save a city for people I recognise.")
            return
        if not city:
            dialog.say("Which city?")
            return
        try:
            places = self.weather.client.find(city, count=3)
        except WeatherError as exc:
            dialog.say(exc.spoken)
            return
        if not places:
            dialog.say(f"I couldn't find a place called {city}. Try the city with its state or country.")
            return
        for place in places:
            answer = dialog.ask(f"{place.spoken}. Is that right?")
            ok = yes_no(answer.text) if answer else None
            if ok is True:
                self.weather.save_place(user, place)
                dialog.say(f"Done. I'll use {place.name} for your weather.")
                return
            if ok is None:
                break
        dialog.say("Okay, I haven't changed it. Try the city with its state or country.")
