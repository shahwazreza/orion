"""Spoken time phrases <-> datetimes, for the calendar ("tomorrow afternoon", "at 3").

Deterministic on purpose: a small model asked "what's tomorrow's date?" can be
wrong, and a wrong range means a wrong answer about someone's day. Everything is
in the computer's local time zone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# Parts of the day: (start hour, end hour).
PARTS = {"morning": (6, 12), "afternoon": (12, 17), "evening": (17, 22), "tonight": (17, 24),
         "night": (17, 24)}


@dataclass
class Span:
    start: datetime
    end: datetime
    label: str              # how to say it: "today", "tomorrow afternoon", "on Friday"
    at: datetime | None = None   # a specific time asked about ("at 3")


def local_zone():
    """This computer's time zone with its daylight-saving rules (a fixed offset would put
    'this weekend' an hour out when the clocks change). Falls back to the current offset."""
    import os
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = os.environ.get("TZ", "").lstrip(":")
    if not name:
        real = os.path.realpath("/etc/localtime")
        name = real.split("zoneinfo/", 1)[1] if "zoneinfo/" in real else ""
    try:
        return ZoneInfo(name) if name else datetime.now().astimezone().tzinfo
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now().astimezone().tzinfo


def local_now() -> datetime:
    return datetime.now(local_zone())


def _day_start(d: date, tz) -> datetime:
    return datetime.combine(d, time(0), tzinfo=tz)



def day_name(d: date, today: date) -> str:
    """'today', 'tomorrow', 'yesterday', 'on Friday' (this coming week), or 'on 3 November'."""
    delta = (d - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    if 1 < delta < 7 or -7 < delta < -1:
        return f"on {WEEKDAYS[d.weekday()].title()}"
    return f"on {WEEKDAYS[d.weekday()].title()} {d.day} {MONTHS[d.month - 1].title()}"


def clock(t: datetime) -> str:
    """'10 AM', '7:30 PM', 'noon', 'midnight'."""
    if t.minute == 0 and t.hour in (0, 12):
        return "midnight" if t.hour == 0 else "noon"
    h = t.hour % 12 or 12
    suffix = "AM" if t.hour < 12 else "PM"
    return f"{h} {suffix}" if t.minute == 0 else f"{h}:{t.minute:02d} {suffix}"


def when_said(t: datetime, now: datetime) -> str:
    """When something happened, for mail: 'at 9:14 AM', 'yesterday at 4 PM', 'on Monday'."""
    t = t.astimezone(now.tzinfo)
    day = day_name(t.date(), now.date())
    if day == "today":
        return f"at {clock(t)}"
    if day == "yesterday":
        return f"yesterday at {clock(t)}"
    return day


# ---- parsing ----------------------------------------------------------------------
_TIME = re.compile(
    r"\b(?:at|around|by|from)\s+(?P<h>\d{1,2}|" + "|".join(_NUMBERS) + r")(?::(?P<m>\d{2}))?"
    r"\s*(?P<ap>a\.?\s?m\.?|p\.?\s?m\.?|o'?clock)?(?=\W|$)|\b(?:at\s+)?(?P<word>noon|midday|midnight)\b", re.I)
_DATE = re.compile(r"\b(?:on\s+)?(?:the\s+)?(?P<d1>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<m1>" + "|".join(MONTHS)
                   + r")\b|\b(?P<m2>" + "|".join(MONTHS) + r")\s+(?:the\s+)?(?P<d2>\d{1,2})(?:st|nd|rd|th)?\b|"
                   r"\bon the (?P<d3>\d{1,2})(?:st|nd|rd|th)\b", re.I)
_WEEKDAY = re.compile(r"\b(?P<next>next\s+|this\s+|this coming\s+)?(?P<day>" + "|".join(WEEKDAYS) + r")\b", re.I)
_PART = re.compile(r"\b(?:this\s+|in the\s+)?(?P<part>morning|afternoon|evening|tonight|night)\b", re.I)


def parse_time_of_day(text: str) -> tuple[int, int] | None:
    """(hour, minute) for 'at 3', 'at 3:30pm', 'at 15:00', 'at noon'; None if no time is named."""
    m = _TIME.search(text)
    if not m:
        return None
    if m.group("word"):
        return (0, 0) if m.group("word").lower() == "midnight" else (12, 0)
    raw = m.group("h").lower()
    h = _NUMBERS.get(raw) or int(raw)
    minute = int(m.group("m") or 0)
    ap = (m.group("ap") or "").lower().replace(".", "").replace(" ", "")
    if h > 23 or minute > 59:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif ap not in ("am", "pm") and 1 <= h <= 7:
        h += 12                       # "at 3" means 3 PM; nobody books 3 AM by voice
    return h, minute


def parse_span(text: str, now: datetime | None = None) -> Span | None:
    """The stretch of time a calendar question is about, or None if it names none."""
    now = now or local_now()
    tz = now.tzinfo
    t = text.lower()
    today = now.date()
    day: date | None = None
    label = ""

    if re.search(r"\bnext week\b", t):
        start = today + timedelta(days=7 - today.weekday())
        return Span(_day_start(start, tz), _day_start(start + timedelta(days=7), tz), "next week")
    if re.search(r"\b(?:this|the) week(?:end)?\b|\brest of the week\b", t) and "weekend" not in t:
        end = today + timedelta(days=7 - today.weekday())
        return Span(now, _day_start(end, tz), "this week")
    if re.search(r"\b(?:this |the )?weekend\b", t):
        sat = today + timedelta(days=(5 - today.weekday()) % 7)
        if today.weekday() == 6:
            sat = today - timedelta(days=1)
        if re.search(r"\bnext weekend\b", t):
            sat += timedelta(days=7)
        start = max(now, _day_start(sat, tz))
        return Span(start, _day_start(sat + timedelta(days=2), tz),
                    "next weekend" if "next weekend" in t else "this weekend")

    if re.search(r"\btomorrow\b", t):
        day = today + timedelta(days=1)
    elif re.search(r"\byesterday\b", t):
        day = today - timedelta(days=1)
    elif re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", t):
        day = today
    if day is None:
        m = _WEEKDAY.search(t)
        if m:
            wanted = WEEKDAYS.index(m.group("day").lower())
            if (m.group("next") or "").strip() == "next":
                # "next Friday" = Friday of next week; the reply names the date, so it's never a guess.
                day = today + timedelta(days=7 - today.weekday() + wanted)
            else:
                day = today + timedelta(days=(wanted - today.weekday()) % 7)
    if day is None:
        m = _DATE.search(t)
        if m:
            d = int(m.group("d1") or m.group("d2") or m.group("d3"))
            month_name = m.group("m1") or m.group("m2")
            month = MONTHS.index(month_name.lower()) + 1 if month_name else today.month
            year = today.year
            try:
                day = date(year, month, d)
                if day < today - timedelta(days=1):
                    # "on the 3rd" said on the 20th means next month; "3 March" in November means next year
                    day = date(year + 1, month, d) if month_name else _next_month(day)
            except ValueError:
                return None

    at = parse_time_of_day(t)
    part = _PART.search(t)
    if day is None and (at or part):
        day = today
    if day is None:
        return None
    label = day_name(day, today)
    start, end = _day_start(day, tz), _day_start(day + timedelta(days=1), tz)
    if part:
        name = part.group("part").lower()
        lo, hi = PARTS[name]
        start, end = start + timedelta(hours=lo), start + timedelta(hours=hi)
        if name in ("tonight", "night"):
            label = "tonight" if day == today else f"{label} night"
        else:
            label = f"this {name}" if day == today else f"{label} {name}"
    point = None
    if at:
        point = _day_start(day, tz) + timedelta(hours=at[0], minutes=at[1])
    if day == today and start < now and point is None:
        start = now                      # what's left of today
    return Span(start, end, label, point)


def _next_month(d: date) -> date:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(y, m, d.day)
