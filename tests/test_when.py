"""Spoken time phrases for the calendar: ranges, times, and how times are said."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from core.when import clock, day_name, parse_span, parse_time_of_day, when_said

TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 10, 30, 10, 0, tzinfo=TZ)         # a Friday morning


def d(month, day, hour=0, minute=0, year=2026):
    return datetime(year, month, day, hour, minute, tzinfo=TZ)


@pytest.mark.parametrize("text,start,end,label", [
    ("what's on my calendar today", NOW, d(10, 31), "today"),
    ("what have I got tomorrow", d(10, 31), d(11, 1), "tomorrow"),
    ("what's on tomorrow afternoon", d(10, 31, 12), d(10, 31, 17), "tomorrow afternoon"),
    ("anything on this evening", d(10, 30, 17), d(10, 30, 22), "this evening"),
    ("what's on tonight", d(10, 30, 17), d(10, 31), "tonight"),
    ("my schedule on Monday", d(11, 2), d(11, 3), "on Monday"),
    ("what do I have on Friday", NOW, d(10, 31), "today"),            # today is Friday
    ("next Friday", d(11, 6), d(11, 7), "on Friday 6 November"),      # Friday of next week, date said
    ("next Monday", d(11, 2), d(11, 3), "on Monday"),
    ("my calendar this week", NOW, d(11, 2), "this week"),
    ("my meetings next week", d(11, 2), d(11, 9), "next week"),
    ("what's on this weekend", d(10, 31), d(11, 2), "this weekend"),
    ("anything on the 3rd", d(11, 3), d(11, 4), "on Tuesday"),   # the 3rd has passed: next month
    ("what's on November 5", d(11, 5), d(11, 6), "on Thursday"),
    ("what's on November 20", d(11, 20), d(11, 21), "on Friday 20 November"),
    ("what's on the 5th of March", d(3, 5, year=2027), d(3, 6, year=2027), "on Friday 5 March"),
])
def test_ranges(text, start, end, label):
    span = parse_span(text, NOW)
    assert (span.start, span.end, span.label) == (start, end, label)


def test_days_are_local_midnights_across_the_clock_change():
    # US daylight saving ends on Sunday 1 November 2026: that day is 25 hours long.
    span = parse_span("what's on Sunday", NOW)
    assert span.start == d(11, 1) and span.end == d(11, 2)
    utc = ZoneInfo("UTC")         # what Graph is sent; same-zone subtraction ignores the change
    assert (span.end.astimezone(utc) - span.start.astimezone(utc)).total_seconds() == 25 * 3600


@pytest.mark.parametrize("text,want", [
    ("am I free at 3", (15, 0)), ("at 9", (9, 0)), ("at 3:30 pm", (15, 30)), ("at 3:30pm", (15, 30)),
    ("at 11 a.m.", (11, 0)), ("at 12", (12, 0)), ("at noon", (12, 0)), ("at midnight", (0, 0)),
    ("at 15:00", (15, 0)), ("at three", (15, 0)), ("at seven o'clock", (19, 0)), ("at 8", (8, 0)),
    ("at 12 am", (0, 0)), ("what's the time", None), ("at 25", None),
])
def test_times(text, want):
    assert parse_time_of_day(text) == want


def test_a_time_alone_means_today_and_with_a_day_means_that_day():
    assert parse_span("am I free at 3", NOW).at == d(10, 30, 15)
    span = parse_span("am I free at 3 tomorrow", NOW)
    assert span.at == d(10, 31, 15) and span.label == "tomorrow"


@pytest.mark.parametrize("text", ["tell me a joke", "what's the capital of France", "how do meetings work"])
def test_no_time_named(text):
    assert parse_span(text, NOW) is None


def test_how_times_and_days_are_said():
    assert [clock(d(1, 1, h, m)) for h, m in [(10, 0), (19, 30), (12, 0), (0, 0), (0, 15), (12, 5)]] == \
        ["10 AM", "7:30 PM", "noon", "midnight", "12:15 AM", "12:05 PM"]
    today = NOW.date()
    assert day_name(today, today) == "today"
    assert day_name(d(10, 31).date(), today) == "tomorrow"
    assert day_name(d(10, 29).date(), today) == "yesterday"
    assert day_name(d(11, 3).date(), today) == "on Tuesday"
    assert day_name(d(11, 20).date(), today) == "on Friday 20 November"


def test_when_an_email_arrived():
    assert when_said(d(10, 30, 9, 14), NOW) == "at 9:14 AM"
    assert when_said(d(10, 29, 16), NOW) == "yesterday at 4 PM"
    assert when_said(d(10, 26, 8), NOW) == "on Monday"
    # Graph sends UTC; it's said in local time.
    assert when_said(datetime(2026, 10, 30, 14, 0, tzinfo=ZoneInfo("UTC")), NOW) == "at 9 AM"
