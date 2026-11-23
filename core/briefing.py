"""The morning briefing: greeting, weather, today's calendar, unread email, what's still to come.

Runs after a morning alarm is stopped, or on "good morning" / "brief me". The parts
are fetched at the same time, each with its own timeout, and said in order as they
arrive: a slow Outlook doesn't hold up the greeting or the weather. Every part is
built by code from the data (like Outlook and weather), never by the model.
People Orion doesn't recognise get the greeting and the weather only.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timedelta
from typing import Callable

from core.alarms import describe as describe_item, reminder_text
from core.dialog import Dialog, Heard
from core.schedule import ALARM, REMINDER, TIMER, Schedule
from core.when import MONTHS, WEEKDAYS, clock, local_now, say_when
from storage.users import User

log = logging.getLogger(__name__)

SECTIONS = ("weather", "calendar", "email", "schedule")
LOGGED_REPLY = "(morning briefing)"

_ASK = re.compile(r"^(?:(?:hey |ok |okay )?orion,? )?(?:can you |could you |please )?(?:give me |run |do )?(?:my |the |a )?"
                  r"(?:morning |daily )?brief(?:ing)?(?: me)?(?: for today)?(?:,? please)?[.!?]*$|"
                  r"^(?:(?:hey |ok |okay )?orion,? )?brief me\b|\bmorning briefing\b|\bdaily briefing\b|"
                  r"\bwhat(?:'s| does| is) my day look(?:ing)? like\b|\b(?:run|go) (?:me )?through my day\b", re.I)
_GOOD_MORNING = re.compile(r"^(?:(?:hey |ok |okay )?orion,? )?(?:good )?morning(?:,? orion)?[.!]*$", re.I)


def wants_briefing(text: str, now: datetime) -> bool:
    t = text.strip()
    return bool(_ASK.search(t)) or (bool(_GOOD_MORNING.match(t)) and now.hour < 12)


def greeting(name: str | None, now: datetime) -> str:
    part = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 17 else "Good evening"
    who = f", {name}" if name else ""
    return (f"{part}{who}. It's {clock(now)} on {WEEKDAYS[now.weekday()].title()}, "
            f"{now.day} {MONTHS[now.month - 1].title()}.")


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class Briefing:
    def __init__(self, *, weather=None, outlook=None, schedule: Schedule | None = None,
                 sections: list[str] | tuple[str, ...] = SECTIONS, timeout_s: float = 10.0,
                 now: Callable[[], datetime] = local_now,
                 clean_for: Callable[[User | None], Callable[[str], str]] | None = None,
                 remember: Callable[[User | None, str, str], None] | None = None) -> None:
        self.weather = weather          # core.weather.Weather
        self.outlook = outlook          # core.outlook.Outlook
        self.schedule = schedule
        self.sections = [s for s in sections if s in SECTIONS]
        self.timeout_s = timeout_s
        self.now = now
        self.clean_for = clean_for or (lambda user: (lambda s: s))
        self.remember = remember or (lambda user, q, a: None)
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="briefing")

    def handle(self, heard: Heard, dialog: Dialog, user: User | None) -> bool:
        if not wants_briefing(heard.text, self.now()):
            return False
        self.run(dialog, user)
        self.remember(user, heard.text.strip(), LOGGED_REPLY)
        return True

    def run(self, dialog: Dialog, user: User | None) -> list[str]:
        """Say the briefing. Returns what was said (for tests; never logged)."""
        clean = self.clean_for(user)
        said: list[str] = []

        def say(line: str | None) -> None:
            if line:
                line = clean(line)
                said.append(line)
                dialog.say(line)

        sections = [s for s in self.sections if user is not None or s == "weather"]
        futures = [(s, self._pool.submit(getattr(self, f"_{s}"), user)) for s in sections]
        say(greeting(user.name if user else None, self.now()))
        deadline = datetime.now().timestamp() + self.timeout_s
        for name, future in futures:
            try:
                say(future.result(timeout=max(0.1, deadline - datetime.now().timestamp())))
            except FutureTimeout:
                log.info("Briefing: %s took too long", name)
                say(_SLOW.get(name))
            except Exception as exc:
                log.info("Briefing: %s failed (%s)", name, type(exc).__name__)
                say(_FAILED.get(name))
        return said

    # ---- parts: each returns a sentence, or None to skip -----------------------------
    def _weather(self, user: User | None) -> str | None:
        if self.weather is None:
            return None
        place = self.weather.place_for(user)
        if place is None:
            return "I don't know your city yet, so no weather. Say 'set my city to', then the city." \
                if user is not None else None
        return self.weather.today_line(self.weather.forecast(place))

    def _calendar(self, user: User) -> str | None:
        from core.outlook import describe_event

        if self.outlook is None or not self.outlook.connected(user):
            return None
        now = self.now()
        end = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=now.tzinfo)
        events = [e for e in self.outlook.graph.events(user.id, now, end) if e.end > now]
        if not events:
            return "Nothing on your calendar today."
        shown = [describe_event(e, now) for e in events[:6]]
        more = f" And {len(events) - 6} more." if len(events) > 6 else ""
        return ("On your calendar today: " if len(events) > 1 else "On your calendar today, ") + _and(shown) + "." + more

    def _email(self, user: User) -> str | None:
        from core.outlook import spoken_name

        if self.outlook is None or not self.outlook.connected(user):
            return None
        g = self.outlook.graph
        count = g.unread_count(user.id)
        if count == 0:
            return "No unread email."
        senders = list(dict.fromkeys(spoken_name(e.sender, e.address) for e in g.unread(user.id, top=5)))[:3]
        line = f"You have {count} unread email" + ("s" if count != 1 else "")
        return line + (f", from {_and(senders)}." if senders else ".")

    def _schedule(self, user: User) -> str | None:
        if self.schedule is None:
            return None
        now = self.now()
        end = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=now.tzinfo)
        items = [i for i in self.schedule.for_user(user.id) if i.due <= end.timestamp()]
        parts = []
        for i in items[:5]:
            due = i.due_at(now.tzinfo)
            if i.kind == REMINDER:
                parts.append(f"a reminder at {clock(due)} to {reminder_text(i.label)}")
            elif i.kind == TIMER:
                parts.append(describe_item(i, now))
            elif i.kind == ALARM:
                parts.append(f"an alarm at {clock(due)}")
        nxt = [i for i in self.schedule.for_user(user.id, (ALARM,)) if i.due > end.timestamp()]
        line = ("Still to come today: " + _and(parts) + ".") if parts else ""
        if nxt and nxt[0].due - now.timestamp() < 36 * 3600:
            line += f" Your next alarm is {say_when(nxt[0].due_at(now.tzinfo), now)}."
        return line.strip() or None


_SLOW = {"weather": "The weather is taking too long, so I'll skip it.",
         "calendar": "Outlook is taking too long, so I'll skip your calendar.",
         "email": "Outlook is taking too long, so I'll skip your email.",
         "schedule": None}
_FAILED = {"weather": "I couldn't get the weather.", "calendar": "I couldn't reach your calendar.",
           "email": "I couldn't reach your email.", "schedule": None}
