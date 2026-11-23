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

    try:
        day = named_day(t, today)
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


def named_day(text: str, today: date) -> date | None:
    """The day a phrase names ("tomorrow", "on Friday", "the 3rd of May"), or None.
    Raises ValueError for a date that doesn't exist ("30 February")."""
    t = text.lower()
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", t):
        return today - timedelta(days=1)
    if re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", t):
        return today
    m = _WEEKDAY.search(t)
    if m:
        wanted = WEEKDAYS.index(m.group("day").lower())
        if (m.group("next") or "").strip() == "next":
            # "next Friday" = Friday of next week; the reply names the date, so it's never a guess.
            return today + timedelta(days=7 - today.weekday() + wanted)
        return today + timedelta(days=(wanted - today.weekday()) % 7)
    m = _DATE.search(t)
    if m:
        d = int(m.group("d1") or m.group("d2") or m.group("d3"))
        month_name = m.group("m1") or m.group("m2")
        month = MONTHS.index(month_name.lower()) + 1 if month_name else today.month
        day = date(today.year, month, d)
        if day < today - timedelta(days=1):
            # "on the 3rd" said on the 20th means next month; "3 March" in November means next year
            day = date(today.year + 1, month, d) if month_name else _next_month(day)
        return day
    return None


def _next_month(d: date) -> date:
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return date(y, m, d.day)


# ---- alarms, timers, reminders (Phase 4) ---------------------------------------------
_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_NUM_WORD = (r"(?:(?:" + "|".join(_TENS) + r")(?:[ -](?:" + "|".join(_UNITS[1:10]) + r"))?|"
             + "|".join(sorted(_UNITS, key=len, reverse=True)) + ")")
_NUM = r"(?:\d+(?:\.\d+)?|" + _NUM_WORD + r"|an?|a couple of|a few)"
_DURATION = re.compile(
    rf"\b(?P<n>{_NUM})(?P<half> and a half)?[ -](?P<unit>hours?|hrs?|minutes?|mins?|seconds?|secs?)\b"
    rf"(?P<half2> and a half)?", re.I)
_HALF_HOUR = re.compile(r"\bhalf an? hour\b|\b(?:a )?half[ -]hour\b", re.I)
_QUARTER_HOUR = re.compile(r"\ba quarter (?:of an |an )?hour\b", re.I)


def number(word: str) -> float | None:
    """'7', '2.5', 'seven', 'twenty five', 'an' -> a number; None if it isn't one."""
    w = word.lower().strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", w):
        return float(w)
    if w in ("a", "an"):
        return 1.0
    if w == "a couple of":
        return 2.0
    if w == "a few":
        return 3.0
    if w in _UNITS:
        return float(_UNITS.index(w))
    parts = re.split(r"[ -]", w)
    if parts[0] in _TENS:
        rest = _UNITS.index(parts[1]) if len(parts) == 2 and parts[1] in _UNITS[1:10] else 0
        return float(_TENS[parts[0]] + rest)
    return None


def parse_duration(text: str) -> timedelta | None:
    """'10 minutes', 'an hour and a half', 'two and a half hours', '1 hour 20 minutes', 'half an hour'."""
    total = 0.0
    found = False
    for special, seconds in ((_HALF_HOUR, 1800), (_QUARTER_HOUR, 900)):
        if special.search(text):
            total, found = total + seconds, True
            text = special.sub(" ", text)
    for m in _DURATION.finditer(text):
        n = number(m.group("n"))
        if n is None:
            continue
        if m.group("half") or m.group("half2"):
            n += 0.5
        unit = m.group("unit").lower()
        total += n * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)
        found = True
    return timedelta(seconds=round(total)) if found and total > 0 else None


def say_duration(d: timedelta) -> str:
    """'10 minutes', '1 hour and 30 minutes', '45 seconds'."""
    s = int(round(d.total_seconds()))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hour" + ("s" if h != 1 else ""))
    if m:
        parts.append(f"{m} minute" + ("s" if m != 1 else ""))
    if sec and not h:
        parts.append(f"{sec} second" + ("s" if sec != 1 else ""))
    if not parts:
        return "less than a second"
    return " and ".join(parts)


# A clock time for an alarm or reminder: "at 7", "for 6:30 am", "7.15", "at seven", "at noon".
_CLOCK = re.compile(
    r"\b(?:(?P<pre>at|for|to|by|around)\s+)?(?P<h>\d{1,2}|" + "|".join(_NUMBERS) + r")"
    r"(?:(?::|\.)(?P<m>\d{2})|\s+(?P<mw>oh[ -]?(?:" + "|".join(_UNITS[1:10]) + r")|fifteen|thirty|forty[ -]five|"
    r"(?:twenty|thirty|forty|fifty)(?:[ -](?:" + "|".join(_UNITS[1:10]) + r"))?|ten|eleven|twelve|thirteen|"
    r"fourteen|sixteen|seventeen|eighteen|nineteen))?"
    r"\s*(?P<ap>a\.?\s?m\.?|p\.?\s?m\.?|o'?clock|in the (?:morning|afternoon|evening)|at night|tonight)?(?=\W|$)|"
    r"\b(?:at\s+)?(?P<word>noon|midday|midnight)\b", re.I)


@dataclass
class ClockTime:
    hour: int               # as said: 1-12 when no AM/PM was given, else 0-23
    minute: int
    meridiem: str | None    # "am" | "pm" | None (not said)


def parse_clock(text: str) -> ClockTime | None:
    """The time in an alarm or reminder request, without deciding AM or PM yet.

    A bare number counts only after "at", "for", "to", "by" or "around", or with AM/PM or
    minutes, so "remind me in 2 hours" or "my 3 kids" don't read as clock times.
    """
    for m in _CLOCK.finditer(text):
        if m.group("word"):
            return ClockTime(0 if m.group("word").lower() == "midnight" else 12, 0, "am" if
                             m.group("word").lower() == "midnight" else "pm")
        ap = (m.group("ap") or "").lower().replace(".", "").replace(" ", "")
        if not (m.group("pre") or m.group("m") or m.group("mw") or ap):
            continue
        raw = m.group("h").lower()
        h = int(raw) if raw.isdigit() else _NUMBERS[raw]
        # "at 10 minutes" is a duration, not a time
        after = text[m.end():m.end() + 12].lower()
        if re.match(r"\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?)\b", after) and not ap:
            continue
        minute = int(m.group("m")) if m.group("m") else _minute_words(m.group("mw")) if m.group("mw") else 0
        if h > 23 or minute is None or minute > 59:
            continue
        meridiem = None
        if ap in ("am", "inthemorning"):
            meridiem = "am"
        elif ap in ("pm", "intheafternoon", "intheevening", "atnight", "tonight"):
            meridiem = "pm"
        if h > 12 or h == 0:
            meridiem = None if h > 12 else "am"       # 24-hour clock
        return ClockTime(h, minute, meridiem)
    return None


def _minute_words(words: str) -> int | None:
    w = words.lower().replace("-", " ")
    if w.startswith("oh "):
        n = number(w[3:])
        return int(n) if n is not None else None
    if w == "forty five":
        return 45
    n = number(w)
    return int(n) if n is not None else None


def _at(day: date, hour: int, minute: int, tz) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=tz)


def resolve_clock(ct: ClockTime, now: datetime, day: date | None = None, morning: bool = False,
                  part: str | None = None) -> datetime:
    """When an alarm or reminder for ``ct`` should go off.

    ``morning``: "wake me up at 7" means 7 AM. ``part``: a part of the day that was said
    ("tomorrow morning"). Otherwise a bare hour is the next time that hour comes round
    (or, on a named day other than today, 8 to 11 is the morning and 12 to 7 the afternoon).
    """
    tz = now.tzinfo
    h = ct.hour
    if ct.meridiem == "am":
        hours = [0 if h == 12 else h]
    elif ct.meridiem == "pm":
        hours = [h if h == 12 else h + 12]
    elif h > 12 or h == 0:
        hours = [h]
    elif morning or part == "morning":
        hours = [0 if h == 12 else h]
    elif part in ("afternoon", "evening", "tonight", "night"):
        hours = [h if h == 12 else h + 12]
    elif day is not None and day != now.date():
        hours = [h + 12 if 1 <= h <= 7 else h]
    else:
        hours = sorted({h % 12, h % 12 + 12})
    if day is not None:
        return _at(day, hours[0], ct.minute, tz)
    for d in (now.date(), now.date() + timedelta(days=1)):
        for hh in sorted(hours):
            t = _at(d, hh, ct.minute, tz)
            if t > now:
                return t
    return _at(now.date() + timedelta(days=1), hours[0], ct.minute, tz)


# Repeats: weekday numbers, Monday = 0.
_EVERY_DAY = re.compile(r"\b(?:every ?day|daily|every (?:morning|night|evening))\b", re.I)
_WEEKDAYS_RE = re.compile(r"\b(?:every weekday|on weekdays|weekdays|monday to friday|monday through friday|"
                          r"workdays|every work ?day)\b", re.I)
_WEEKENDS_RE = re.compile(r"\b(?:every weekend|on weekends|weekends)\b", re.I)
_EVERY_NAMED = re.compile(r"\b(?:every|on|each)\s+(?P<days>(?:(?:" + "|".join(WEEKDAYS) + r")s?(?:,?\s*(?:and\s+)?)?)+)",
                          re.I)


def parse_repeat(text: str) -> list[int] | None:
    """Days an alarm repeats on, or None if it's a one-off."""
    if _WEEKDAYS_RE.search(text):
        return [0, 1, 2, 3, 4]
    if _WEEKENDS_RE.search(text):
        return [5, 6]
    if _EVERY_DAY.search(text):
        return list(range(7))
    m = _EVERY_NAMED.search(text)
    if m and (m.group(0).lower().startswith(("every", "each")) or re.search(r"(?:mondays|tuesdays|wednesdays|"
                                                                          r"thursdays|fridays|saturdays|sundays)",
                                                                          m.group("days"), re.I)):
        days = sorted({WEEKDAYS.index(d.lower().rstrip("s") if d.lower() != "tuesdays" else "tuesday")
                       for d in re.findall("|".join(WEEKDAYS) + "|" + "|".join(w + "s" for w in WEEKDAYS),
                                           m.group("days"), re.I)})
        return days or None
    return None


def next_on_days(hour: int, minute: int, days: list[int], after: datetime) -> datetime:
    """The first ``hour:minute`` on one of ``days`` strictly after ``after`` (DST-safe: local wall time)."""
    tz = after.tzinfo
    for i in range(8):
        d = after.date() + timedelta(days=i)
        if d.weekday() in days:
            t = _at(d, hour, minute, tz)
            if t > after:
                return t
    raise ValueError("no repeat days")


def say_days(days: list[int]) -> str:
    """'every day', 'on weekdays', 'on weekends', 'every Monday and Wednesday'."""
    if sorted(days) == list(range(7)):
        return "every day"
    if sorted(days) == [0, 1, 2, 3, 4]:
        return "on weekdays"
    if sorted(days) == [5, 6]:
        return "on weekends"
    names = [WEEKDAYS[d].title() for d in sorted(days)]
    return "every " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])


def say_when(t: datetime, now: datetime) -> str:
    """'7 AM tomorrow', '5 PM today', '9 AM on Friday', '9 AM on Friday 13 November'."""
    t = t.astimezone(now.tzinfo)
    day = day_name(t.date(), now.date())
    return f"{clock(t)} {day}"


def part_of_day(text: str) -> str | None:
    m = _PART.search(text)
    return m.group("part").lower() if m else None
