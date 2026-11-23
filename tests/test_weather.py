"""Phase 4 weather (Open-Meteo, mocked) and the morning briefing."""

import json
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest

from brain.capabilities import unsupported
from core.briefing import LOGGED_REPLY, Briefing, greeting, wants_briefing
from core.dialog import Heard
from core.schedule import ALARM, REMINDER, Schedule
from core.weather import NO_CITY, SETTING, Weather, WeatherCommands, is_weather_request
from storage.audit import AuditLog
from storage.db import Database
from storage.users import User
from tools.weather import OpenMeteo, Place, WeatherError, uses_fahrenheit

TZ = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 2, 7, 5, tzinfo=TZ)             # Monday morning
REZA = User(id="r", name="Reza", role="owner")
SAM = User(id="s", name="Sam")

TAMPA = {"name": "Tampa", "latitude": 27.94752, "longitude": -82.45843, "country_code": "US",
         "country": "United States", "admin1": "Florida"}
TAMPA_KS = {"name": "Tampa", "latitude": 38.547, "longitude": -97.153, "country_code": "US",
            "country": "United States", "admin1": "Kansas"}
AUSTIN = {"name": "Austin", "latitude": 30.267, "longitude": -97.743, "country_code": "US",
          "country": "United States", "admin1": "Texas"}
LONDON = {"name": "London", "latitude": 51.508, "longitude": -0.126, "country_code": "GB",
          "country": "United Kingdom", "admin1": "England"}
PLACES = {"tampa": [TAMPA, TAMPA_KS], "austin": [AUSTIN], "london": [LONDON]}


def forecast_json(start=date(2026, 11, 2), rain=(83, 28, 35, 20, 6, 3, 12), codes=(65, 51, 3, 1, 0, 2, 3)):
    days = [(start + timedelta(days=i)).isoformat() for i in range(7)]
    return {"current": {"temperature_2m": 71.6, "apparent_temperature": 75.0, "weather_code": 3,
                        "wind_speed_10m": 8.8},
            "daily": {"time": days, "weather_code": list(codes), "temperature_2m_max": [75.2, 81.6, 87.7, 86.6,
                                                                                      84.4, 83.6, 81.9],
                      "temperature_2m_min": [66.6, 71.9, 71.3, 71.5, 74.1, 70.8, 71.3],
                      "precipitation_probability_max": list(rain)}}


class Server:
    """A fake Open-Meteo. Records every request."""

    def __init__(self, fail=None, **fc):
        self.requests = []
        self.fail = fail
        self.fc = fc

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail == "down":
            raise httpx.ConnectError("no route")
        if self.fail == "slow":
            raise httpx.ReadTimeout("slow")
        if request.url.host.startswith("geocoding"):
            name = request.url.params["name"].lower()
            return httpx.Response(200, json={"results": PLACES.get(name, [])} if name in PLACES else {})
        return httpx.Response(200, json=forecast_json(**self.fc))


def make(server=None, units="auto", audit=None, now=NOW):
    server = server or Server()
    client = OpenMeteo(client=httpx.Client(transport=httpx.MockTransport(server)), audit=audit)
    db = Database(None, b"k" * 32)
    schedule = Schedule(db, clock=lambda: now.timestamp())
    weather = Weather(client, schedule, owner=lambda: REZA, units=units, now=lambda: now)
    return server, schedule, weather, WeatherCommands(weather)


class Dialog:
    hears_voice = False

    def __init__(self, answers=()):
        self.said, self.asked, self.answers = [], [], list(answers)

    def say(self, text):
        self.said.append(text)

    def ask(self, prompt, timeout_s=None):
        self.asked.append(prompt)
        return Heard(self.answers.pop(0)) if self.answers else None


def ask(cmds, text, user=REZA, answers=()):
    d = Dialog(answers)
    assert cmds.handle(Heard(text), d, user) is True
    return " ".join(d.said), d


def with_city(weather, user=REZA, place=TAMPA):
    weather.save_place(user, Place(place["name"], place["latitude"], place["longitude"], place["admin1"],
                                   place["country"], place["country_code"]))


# ---- the client ---------------------------------------------------------------------------
def test_find_narrows_by_state():
    _, _, weather, _ = make()
    assert [p.spoken for p in weather.client.find("Tampa")] == ["Tampa, Florida, United States",
                                                                "Tampa, Kansas, United States"]
    assert [p.admin1 for p in weather.client.find("Tampa, Kansas")] == ["Kansas"]
    assert [p.admin1 for p in weather.client.find("Austin TX")] == ["Texas"]     # retried as "Austin"
    assert weather.client.find("Atlantis") == []


def test_forecast_sends_rounded_coordinates_and_is_cached():
    server, _, weather, _ = make()
    place = Place("Tampa", 27.94752, -82.45843, "Florida", "United States", "US")
    fc = weather.forecast(place)
    weather.forecast(place)
    [req] = server.requests
    assert req.url.params["latitude"] == "27.95" and req.url.params["longitude"] == "-82.46"
    assert req.url.params["temperature_unit"] == "fahrenheit"
    assert round(fc.temp) == 72 and fc.days[0].rain_chance == 83


def test_saved_city_is_rounded():
    _, schedule, weather, _ = make()
    with_city(weather)
    stored = json.loads(schedule.setting("r", SETTING))
    assert stored["lat"] == 27.95 and stored["lon"] == -82.46


@pytest.mark.parametrize("code,units,f", [("US", "auto", True), ("GB", "auto", False), ("GB", "fahrenheit", True),
                                         ("US", "celsius", False)])
def test_units(code, units, f):
    assert uses_fahrenheit(Place("x", 0, 0, country_code=code), units) is f


def test_audit_never_has_the_city_or_coordinates():
    db = Database(None, b"k" * 32)
    audit = AuditLog(db)
    _, _, weather, cmds = make(audit=audit)
    ask(cmds, "set my city to Tampa", answers=["yes"])
    ask(cmds, "what's the weather")
    rows = audit.recent(10)
    assert {k for _, k, _ in rows} == {"weather_geocode", "weather_forecast"}
    dump = json.dumps([d for _, _, d in rows]).lower()
    for private in ("tampa", "florida", "27.9", "82.4"):
        assert private not in dump


@pytest.mark.parametrize("fail,spoken", [("down", "I can't reach the weather service. Is the internet down?"),
                                         ("slow", "The weather service didn't answer in time.")])
def test_failures_are_said_never_guessed(fail, spoken):
    _, _, weather, cmds = make(Server(fail=fail))
    with_city(weather)
    assert ask(cmds, "what's the weather")[0] == spoken


# ---- answers ---------------------------------------------------------------------------------
@pytest.mark.parametrize("question,answer", [
    ("what's the weather", "In Tampa it's 72 degrees and overcast. Today, heavy rain, a high of 75 and a low of "
                           "67, with an 83 percent chance of rain."),
    ("what's the weather tomorrow", "Tomorrow in Tampa: light drizzle, a high of 82 and a low of 72, with a 28 "
                                    "percent chance of rain."),
    ("do I need an umbrella", "Yes, take an umbrella: there's an 83 percent chance of rain today in Tampa."),
    ("do I need an umbrella on Friday", "Probably not: there's a 6 percent chance of rain Friday in Tampa."),
    ("will it rain on Wednesday", "Possibly: there's a 35 percent chance of rain Wednesday in Tampa."),
    ("is it going to snow", "No snow in the forecast for today in Tampa."),
    ("how hot is it", "It's 72 degrees in Tampa right now, feels like 75. Today's high is 75 and the low 67."),
    ("what's the weather on the 20th of November", "I can only see a week ahead."),
])
def test_answers(question, answer):
    _, _, weather, cmds = make()
    with_city(weather)
    assert ask(cmds, question)[0] == answer


def test_weekend_lists_both_days():
    _, _, weather, cmds = make()
    with_city(weather)
    reply = ask(cmds, "what's the weather this weekend")[0]
    assert reply.startswith("This weekend in Tampa: Saturday") and "; Sunday" in reply


def test_no_city_yet():
    _, _, weather, cmds = make()
    assert ask(cmds, "what's the weather")[0] == NO_CITY


def test_guests_and_others_get_the_owners_city():
    _, _, weather, cmds = make()
    with_city(weather)
    assert ask(cmds, "what's the weather", user=None)[0].startswith("In Tampa")
    assert ask(cmds, "what's the weather", user=SAM)[0].startswith("In Tampa")


def test_a_named_place_is_looked_up_not_saved():
    _, schedule, weather, cmds = make()
    with_city(weather)
    assert ask(cmds, "what's the weather in London")[0].startswith("In London")
    assert json.loads(schedule.setting("r", SETTING))["name"] == "Tampa"
    assert ask(cmds, "what's the weather in Atlantis")[0] == "I couldn't find that place."


def test_setting_a_city_reads_back_each_match():
    _, schedule, weather, cmds = make()
    reply, d = ask(cmds, "set my city to Tampa", answers=["no", "yes"])
    assert d.asked == ["Tampa, Florida, United States. Is that right?", "Tampa, Kansas, United States. Is that right?"]
    assert reply == "Done. I'll use Tampa for your weather."
    assert json.loads(schedule.setting("r", SETTING))["admin1"] == "Kansas"
    assert ask(cmds, "what city are you using")[0] == "I'm using Tampa, Kansas, United States."


def test_setting_a_city_needs_a_known_voice():
    _, schedule, weather, cmds = make()
    assert ask(cmds, "set my city to Tampa", user=None)[0] == "I only save a city for people I recognise."
    assert schedule.setting("guest", SETTING) is None


def test_unclear_answer_changes_nothing():
    _, schedule, weather, cmds = make()
    ask(cmds, "change my weather location to Austin", answers=["hmm"])
    assert schedule.setting("r", SETTING) is None


@pytest.mark.parametrize("text,expected", [
    ("what's the weather", True), ("will it rain tomorrow", True), ("do I need a jacket", True),
    ("what's the temperature outside", True), ("how cold is it", True),
    ("why is the sky blue", False), ("explain how weather forecasting works", False),
    ("ask Claude about the weather on Mars", False), ("remind me to check the weather", False),
    ("whenever it rains, remind me to close the window", False), ("what's on my calendar", False),
])
def test_weather_phrases(text, expected):
    assert is_weather_request(text) is expected


def test_guard_with_weather_wired():
    assert unsupported("what's the weather", weather=True).startswith("I can't answer that about the weather")
    assert unsupported("what's the weather", searchable=True) is None       # before Phase 4: search


# ---- briefing ---------------------------------------------------------------------------------
class FakeOutlook:
    def __init__(self, connected=True, fail=False, slow=0.0):
        from tools.outlook import Email, Event

        self._connected, self.fail, self.slow = connected, fail, slow
        start = NOW.replace(hour=10, minute=0)
        self.graph = self
        self.events_ = [Event("1", "Team standup", start, start + timedelta(minutes=30), location="Room 4"),
                        Event("2", "Dentist", start.replace(hour=15), start.replace(hour=16))]
        self.mail = [Email("m1", "Alex Chen", "alex@example.com", "Lunch?", NOW - timedelta(hours=1), False),
                     Email("m2", "GitHub", "noreply@github.com", "PR merged", NOW - timedelta(hours=2), False)]

    def connected(self, user):
        return self._connected

    def events(self, uid, start, end):
        if self.slow:
            threading.Event().wait(self.slow)
        if self.fail:
            raise RuntimeError("graph down")
        return self.events_

    def unread_count(self, uid):
        return 4

    def unread(self, uid, top=5):
        return self.mail


def briefing(outlook=None, weather=True, sections=("weather", "calendar", "email", "schedule"), timeout_s=5.0,
             city=True):
    _, schedule, w, _ = make()
    if city:
        with_city(w)
    return schedule, Briefing(weather=w if weather else None, outlook=outlook, schedule=schedule,
                              sections=sections, timeout_s=timeout_s, now=lambda: NOW)


def test_full_briefing_in_order():
    schedule, b = briefing(FakeOutlook())
    schedule.add("r", REMINDER, NOW.replace(hour=17, minute=0), "call my mom")
    schedule.add("r", ALARM, (NOW + timedelta(days=1)).replace(hour=7, minute=0))
    said = b.run(Dialog(), REZA)
    assert said[0] == "Good morning, Reza. It's 7:05 AM on Monday, 2 November."
    assert said[1].startswith("In Tampa it's 72 degrees")
    assert said[2] == "On your calendar today: Team standup at 10 AM in Room 4 and Dentist at 3 PM."
    assert said[3] == "You have 4 unread emails, from Alex Chen and GitHub."
    assert said[4] == ("Still to come today: a reminder at 5 PM to call your mom. "
                       "Your next alarm is 7 AM tomorrow.")


def test_guests_get_greeting_and_weather_only():
    schedule, b = briefing(FakeOutlook())
    said = b.run(Dialog(), None)
    assert said[0] == "Good morning. It's 7:05 AM on Monday, 2 November."
    assert len(said) == 2 and said[1].startswith("In Tampa")


def test_parts_without_data_are_skipped():
    schedule, b = briefing(FakeOutlook(connected=False), city=False)
    said = b.run(Dialog(), REZA)
    assert said[1].startswith("I don't know your city yet")
    assert len(said) == 2                     # no Outlook, nothing scheduled


def test_a_slow_part_is_skipped_and_doesnt_hold_up_the_rest():
    schedule, b = briefing(FakeOutlook(slow=2.0), timeout_s=0.5, sections=("weather", "calendar"))
    said = b.run(Dialog(), REZA)
    assert said[-1] == "Outlook is taking too long, so I'll skip your calendar."
    assert said[1].startswith("In Tampa")


def test_a_failed_part_gets_one_sentence():
    schedule, b = briefing(FakeOutlook(fail=True), sections=("calendar",))
    assert b.run(Dialog(), REZA)[1:] == ["I couldn't reach your calendar."]


def test_briefing_logs_a_placeholder_not_the_content():
    logged = []
    schedule, b = briefing(FakeOutlook())
    b.remember = lambda user, q, reply: logged.append((q, reply))
    assert b.handle(Heard("brief me"), Dialog(), REZA)
    assert logged == [("brief me", LOGGED_REPLY)]


@pytest.mark.parametrize("text,hour,expected", [
    ("good morning", 7, True), ("Good morning, Orion.", 9, True), ("good morning", 15, False),
    ("brief me", 15, True), ("morning briefing", 8, True), ("what's my day look like", 8, True),
    ("give me my briefing", 18, True), ("how's my day look", 8, False), ("good morning to you too, what's up", 8,
                                                                         False),
])
def test_briefing_phrases(text, hour, expected):
    assert wants_briefing(text, NOW.replace(hour=hour)) is expected


def test_greeting_follows_the_clock():
    assert greeting("Reza", NOW.replace(hour=19)).startswith("Good evening, Reza. It's 7:05 PM")
