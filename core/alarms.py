"""Alarms, timers, and reminders by voice, and what happens when one goes off.

Times are parsed by code (core/when.py) and read back with the day, so a
mishearing is caught at once ("Alarm set for 7 PM today"). Reminder text is the
person's own words, never anything from an email, a web page, or the model.

Alarms and reminders belong to the person who set them. Timers are for anyone:
a guest's timer is shared by everyone. Anyone in the room can stop a ringing
alarm.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Callable, Protocol

import numpy as np

from core.dialog import Dialog, Heard
from core.memory import to_second_person
from core.schedule import ALARM, GUEST, REMINDER, TIMER, Item, Schedule
from core.when import (_WEEKDAY, clock, local_zone, named_day, next_on_days, parse_clock, parse_duration,
                       parse_repeat, part_of_day, resolve_clock, say_days, say_duration, say_when)
from storage.users import User

log = logging.getLogger(__name__)

MAX_TIMER = timedelta(hours=24)
REMINDER_SNOOZE_MIN = 10
MORNING = (4, 11)        # alarms set between these hours start the morning briefing

# ---- phrases ------------------------------------------------------------------------
_KIND_WORDS = {ALARM: r"alarms?", TIMER: r"timers?", REMINDER: r"reminders?"}
_ANY_KIND = r"(?:alarms?|timers?|reminders?)"
_CANCEL = re.compile(r"\b(?:cancel|delete|remove|clear|turn off|switch off|disable|stop|get rid of|kill|"
                     r"forget|drop|scrap)\b.*\b" + _ANY_KIND + r"\b", re.I)
_LIST = re.compile(r"\bwhat\s+(?:alarms|timers|reminders)\b|\b(?:what|which) (?:alarms|timers|reminders) "
                   r"(?:do i have|are (?:set|running|on))\b|"
                   r"\b(?:any|my|list (?:my|the)|read (?:me )?my|tell me my)\s+(?:alarms|timers|reminders)\b"
                   r"(?:\s+(?:set|running|on|for (?:today|tomorrow)))?\s*[?.!]*$|"
                   r"\bdo i have (?:any )?(?:alarms|timers|reminders)\b|"
                   r"\bwhat are my (?:alarms|timers|reminders)\b|"
                   r"\bhow (?:long|much time)(?: is| 's)? (?:left|remaining)\b|\bhow long (?:until|till|before) "
                   r"(?:the|my) (?:timer|alarm)\b|"
                   r"\bwhen(?:'s| is| does) my (?:next )?alarm\b|\bwhat time is my alarm\b|"
                   r"\bis (?:my|an|the) alarm set\b|\bis there (?:an|a) (?:alarm|timer)\b", re.I)
_SET_ALARM = re.compile(r"\b(?:set|make|create|add|put on|schedule)\b.{0,30}?\balarm\b|\bwake me(?: up)?\b|"
                        r"^(?:an? )?alarm (?:for|at)\b|\balarm (?:for|at) \d", re.I)
_SET_TIMER = re.compile(r"\b(?:set|start|make|create|put on|run)\b.{0,30}?\btimer\b|^(?:an? )?(?:\S+ ){0,3}timer\b|"
                        r"\btimer for\b|\bcount ?down\b", re.I)
_REMIND = re.compile(r"\bremind me\b|\bset (?:a|an) reminder\b|\b(?:add|create|make) (?:a|an) reminder\b", re.I)
_WAKE = re.compile(r"\bwake me\b|\bwake(?: |-)?up\b", re.I)
_ALL = re.compile(r"\ball\b|\bevery (?:one|single)\b|\beverything\b", re.I)

# While something rings (no wake word needed).
_SNOOZE = re.compile(r"\bsnooze\b|\b(?:\d+|five|ten|fifteen|a few|couple) more minutes\b|\bremind me (?:again|later)\b|"
                     r"\b(?:again )?later\b|\bnot yet\b|\bgive me\b|\bin a (?:bit|minute|while)\b", re.I)
_STOP = re.compile(r"\b(?:stop|off|i'?m up|i am up|i'?m awake|awake|okay|ok|got it|enough|dismiss|cancel|"
                   r"shut up|quiet|thanks?|thank you|done|alright|all right|yes|yeah|fine|good morning|"
                   r"silence|that'?s enough)\b", re.I)


def ring_reply(text: str) -> tuple[str, float | None] | None:
    """('snooze', minutes or None) / ('stop', None) for what was said while ringing, else None."""
    if not text:
        return None
    if _SNOOZE.search(text):
        d = parse_duration(re.sub(r"\bmore minutes\b", "minutes", text, flags=re.I))
        return "snooze", (d.total_seconds() / 60 if d else None)
    if _STOP.search(text):
        return "stop", None
    return None


# ---- ring tone --------------------------------------------------------------------------
def ring_tone(sample_rate: int, volume: float) -> np.ndarray:
    """About 2.4 s: two bursts of four beeps. Generated, so no audio file ships."""
    note = int(0.11 * sample_rate)
    t = np.arange(note) / sample_rate
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.008)
    beep = (np.sin(2 * np.pi * 1046 * t) + 0.35 * np.sin(2 * np.pi * 2093 * t)) * fade * volume / 1.35
    gap = np.zeros(int(0.07 * sample_rate))
    burst = np.concatenate([np.concatenate([beep, gap]) for _ in range(4)])
    rest = np.zeros(int(0.45 * sample_rate))
    return np.concatenate([burst, rest, burst, rest]).astype(np.float32)


# ---- describing things --------------------------------------------------------------
def reminder_text(label: str) -> str:
    """How a reminder is said back: 'call my mom' -> 'call your mom'."""
    return to_second_person(label).rstrip(" .!?")


def timer_name(item: Item) -> str:
    """'pasta timer', '10 minute timer'."""
    if item.label:
        return f"{item.label} timer"
    if item.seconds:
        return f"{_length_word(item.seconds)} timer"
    return "timer"


def _length_word(seconds: int) -> str:
    """'10 minute', '1 hour', '90 second', '1 hour 30 minute' (as in "a 10 minute timer")."""
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    parts = []
    if h:
        parts.append(f"{h} hour")
    if m:
        parts.append(f"{m} minute")
    if s and not h:
        parts.append(f"{s} second")
    return " ".join(parts) or "short"


def describe(item: Item, now: datetime) -> str:
    """One item for a list: '7 AM on weekdays', 'pasta timer, 4 minutes left', '5 PM today: call your mom'."""
    due = item.due_at(now.tzinfo)
    if item.kind == TIMER:
        left = max(timedelta(0), due - now)
        return f"{_a(timer_name(item))} with {say_duration(_round_left(left))} left"
    if item.kind == ALARM:
        if item.repeats and not item.snoozed:
            return f"{clock(due)} {say_days(item.days)}"
        return say_when(due, now) + (" (snoozed)" if item.snoozed else "")
    return f"{say_when(due, now)}: {reminder_text(item.label)}"


def _round_left(d: timedelta) -> timedelta:
    s = d.total_seconds()
    return timedelta(seconds=round(s) if s < 120 else round(s / 60) * 60)


def _a(phrase: str) -> str:
    return ("an " if re.match(r"(?:[aeio]|8|11\b|18\b)", phrase) else "a ") + phrase


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---- reminder text -----------------------------------------------------------------------
_DAY_WORDS = re.compile(
    r"\b(?:(?:on |this |next |this coming )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?: (?:morning|afternoon|evening|night))?|today|tonight|tomorrow(?: (?:morning|afternoon|evening|night))?|"
    r"this (?:morning|afternoon|evening)|in the (?:morning|afternoon|evening)|at night|later today|"
    r"every ?day|every (?:weekday|morning|night|evening)|on weekdays|on weekends)\b", re.I)
_TIME_BITS = re.compile(
    r"\b(?:at|for|by|around)\s+(?:\d{1,2}(?:[:.]\d{2})?|noon|midday|midnight|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|eleven|twelve)(?:\s*(?:a\.?\s?m\.?|p\.?\s?m\.?|o'?clock))?\b|"
    r"\b\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?\s?m\.?|p\.?\s?m\.?)|\b(?:at\s+)?(?:noon|midday|midnight)\b", re.I)


def reminder_label(text: str) -> str:
    """'remind me at 5 to call my mom' -> 'call my mom'. Empty if no message was given."""
    t = re.sub(r"^.*?\bremind me\b", "", text, flags=re.I) if _REMIND.search(text) else text
    t = re.sub(r"^.*?\bset (?:a|an) reminder\b", "", t, flags=re.I)
    t = re.sub(r"\bin\s+(?:a|an|\d+(?:\.\d+)?|[a-z]+(?:[ -][a-z]+)?)(?: and a half)?\s+(?:hours?|hrs?|minutes?|mins?|"
               r"seconds?|secs?)(?: and a half)?(?:\s+and\s+(?:\d+|[a-z]+)\s+(?:minutes?|mins?))?\b", " ", t, flags=re.I)
    t = re.sub(r"\bin\s+half an hour\b|\bin a quarter of an hour\b", " ", t, flags=re.I)
    t = _TIME_BITS.sub(" ", t)
    t = _DAY_WORDS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.!?")
    t = re.sub(r"^(?:please\s+)?(?:to|about|that|of|for)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(?:please|thanks|thank you|okay)$", "", t, flags=re.I)
    return t.strip(" ,.!?")


def timer_label(text: str) -> str:
    """'set a pasta timer for 8 minutes' -> 'pasta'; '10 minute timer' -> ''."""
    m = re.search(r"\b(?:a|an|the|my)\s+(?P<label>[a-z][a-z' ]{1,24}?)\s+timer\b", text, re.I) or \
        re.search(r"^(?P<label>[a-z][a-z' ]{1,24}?)\s+timer\b", text, re.I)
    if not m:
        m = re.search(r"\btimer for (?:the |my )?(?P<label>[a-z][a-z' ]{1,24}?)(?: for | of |$)", text, re.I)
    if not m:
        return ""
    label = m.group("label").strip()
    if parse_duration(label + " ") or re.search(r"\b(?:minute|hour|second|min|hr|sec|new|quick|kitchen timer|"
                                                r"countdown)s?\b|\d", label, re.I) or parse_duration(f"1 {label}"):
        return ""
    if re.fullmatch(r"(?:set|start|make|create|put|run|another|one|new|quick)(?: (?:a|an|the|my))?", label, re.I):
        return ""
    return label.lower()


# ---- the commands ----------------------------------------------------------------------
class AlarmCommands:
    """"Set an alarm for 7", "10 minute timer", "remind me at 5 to call Mom", list, cancel."""

    def __init__(self, schedule: Schedule, now: Callable[[], datetime] | None = None,
                 remember: Callable[[User | None, str, str], None] | None = None) -> None:
        self.schedule = schedule
        self.now = now or (lambda: datetime.now(local_zone()))
        self.remember = remember or (lambda user, q, a: None)

    def handle(self, heard: Heard, dialog: Dialog, user: User | None) -> bool:
        text = heard.text.strip()
        if not text:
            return False
        kind = self._intent(text)
        if kind is None:
            return False
        said: list[str] = []

        def say(line: str) -> None:
            said.append(line)
            dialog.say(line)

        if kind == "cancel":
            self._cancel(text, say, dialog, user)
        elif kind == "list":
            self._list(text, say, user)
        elif kind == TIMER:
            self._set_timer(text, say, dialog, user)
        elif user is None:
            say("I only set alarms and reminders for people I recognise. Timers work for anyone.")
        elif kind == ALARM:
            self._set_alarm(text, say, dialog, user)
        else:
            self._set_reminder(text, say, dialog, user)
        self.remember(user, text, " ".join(said))
        return True

    @staticmethod
    def _intent(text: str) -> str | None:
        if _CANCEL.search(text) and not re.search(r"\b(?:wake me|remind me)\b", text, re.I):
            return "cancel"
        if _LIST.search(text):
            return "list"
        if _REMIND.search(text) and not re.search(r"\b(?:whenever|every time|each time)\b|^(?:if|when)\b", text, re.I):
            return REMINDER
        if _SET_ALARM.search(text):
            return ALARM
        if _SET_TIMER.search(text) and (parse_duration(text) or re.search(r"\b(?:set|start)\b", text, re.I)):
            return TIMER
        if re.match(r"^(?:a |an )?(?:\S+ ){0,3}(?:minutes?|hours?|seconds?)\s+timer\b", text, re.I):
            return TIMER
        return None

    # ---- setting ---------------------------------------------------------------------------
    def _set_timer(self, text: str, say, dialog: Dialog, user: User | None) -> None:
        length = parse_duration(text)
        if length is None:
            answer = dialog.ask("For how long?")
            length = parse_duration(answer.text) if answer else None
            if length is None:
                say("I didn't catch how long, so no timer.")
                return
        if length > MAX_TIMER:
            say("Timers go up to 24 hours. For longer, set an alarm or a reminder.")
            return
        label = timer_label(text)
        now = self.now()
        item = self.schedule.add(user.id if user else GUEST, TIMER, now + length, label,
                                 seconds=int(length.total_seconds()))
        say(f"{_a(timer_name(item)).capitalize()}, starting now." if label else
            f"Timer set for {say_duration(length)}.")

    def _set_alarm(self, text: str, say, dialog: Dialog, user: User) -> None:
        now = self.now()
        days = parse_repeat(text)
        ct = parse_clock(text)
        # "wake me up in 20 minutes" is a nap: an alarm after a duration.
        nap = parse_duration(text) if ct is None and re.search(r"\bin\s", text, re.I) else None
        if ct is None and nap is None:
            answer = dialog.ask("For what time?")
            if answer is None:
                say("No alarm, then.")
                return
            text = f"{text} {answer.text}"
            days = days or parse_repeat(answer.text)
            ct = parse_clock(answer.text) or parse_clock(f"at {answer.text}")
            nap = parse_duration(answer.text) if ct is None else None
            if ct is None and nap is None:
                say("I didn't catch a time, so no alarm.")
                return
        morning = bool(_WAKE.search(text))
        if nap is not None:
            due = now + nap
            self.schedule.add(user.id, ALARM, due)
            say(f"Alarm set for {say_when(due, now)}, {say_duration(nap)} from now.")
            return
        if days:
            due0 = resolve_clock(ct, now, day=None, morning=morning or ct.meridiem is None and 4 <= ct.hour <= 11,
                                 part=part_of_day(text))
            nxt = next_on_days(due0.hour, due0.minute, days, now)
            self.schedule.add(user.id, ALARM, nxt, hour=due0.hour, minute=due0.minute, days=days)
            say(f"Alarm set for {clock(nxt)} {say_days(days)}. The first one is {say_when(nxt, now)}.")
            return
        try:
            day = named_day(_without_every(text), now.date())
        except ValueError:
            say("That date doesn't exist, so no alarm.")
            return
        # On another day, an alarm for a bare 4 to 11 is a morning alarm ("alarm for 7 tomorrow").
        if day is not None and ct.meridiem is None and 4 <= ct.hour <= 11:
            morning = True
        due = resolve_clock(ct, now, day=day, morning=morning, part=part_of_day(text))
        if due <= now:
            say(f"{_cap(say_when(due, now))} has already passed, so no alarm.")
            return
        clash = [i for i in self.schedule.for_user(user.id, (ALARM,)) if abs(i.due - due.timestamp()) < 30]
        if clash:
            say(f"You already have an alarm for {say_when(due, now)}.")
            return
        self.schedule.add(user.id, ALARM, due)
        say(f"Alarm set for {say_when(due, now)}." + _from_now(due, now))

    def _set_reminder(self, text: str, say, dialog: Dialog, user: User) -> None:
        now = self.now()
        label = reminder_label(text)
        due = self._reminder_time(text, now)
        if due is None:
            answer = dialog.ask("When should I remind you?")
            if answer is None:
                say("No reminder, then.")
                return
            due = self._reminder_time(answer.text, now) or self._reminder_time(f"at {answer.text}", now)
            if due is None:
                say("I didn't catch a time, so no reminder.")
                return
            if not label:
                label = reminder_label(answer.text)
        if due == "invalid":
            say("That date doesn't exist, so no reminder.")
            return
        if due <= now:
            say(f"{_cap(say_when(due, now))} has already passed, so no reminder.")
            return
        if not label:
            answer = dialog.ask("What should I remind you about?")
            label = reminder_label(answer.text) if answer else ""
            if not label:
                say("No reminder, then.")
                return
        self.schedule.add(user.id, REMINDER, due, label)
        say(f"Okay, I'll remind you to {reminder_text(label)} at {say_when(due, now)}."
            if not re.match(r"^(?:about|that)\b", label) else
            f"Okay, I'll remind you at {say_when(due, now)}.")

    @staticmethod
    def _reminder_time(text: str, now: datetime):
        d = parse_duration(text) if re.search(r"\bin\s", text, re.I) else None
        ct = parse_clock(text)
        if ct is None and d is not None:
            return now + d
        if ct is None:
            # "remind me tomorrow morning to ..." with no hour: 9 AM, or a sensible time for the part of day.
            part = part_of_day(text)
            try:
                day = named_day(text, now.date())
            except ValueError:
                return "invalid"
            if day is None and part is None:
                return None
            hour = {"morning": 9, "afternoon": 14, "evening": 18, "tonight": 20, "night": 20}.get(part, 9)
            due = datetime.combine(day or now.date(), datetime.min.time(), tzinfo=now.tzinfo).replace(hour=hour)
            return due
        try:
            day = named_day(text, now.date())
        except ValueError:
            return "invalid"
        return resolve_clock(ct, now, day=day, part=part_of_day(text))

    # ---- listing -------------------------------------------------------------------------
    def _list(self, text: str, say, user: User | None) -> None:
        kinds = _kinds_in(text)
        if re.search(r"\bhow (?:long|much time)\b", text, re.I) and not kinds:
            kinds = (TIMER,)
        if re.search(r"\b(?:when|what time)\b.*\balarm\b", text, re.I):
            kinds = (ALARM,)
        if user is None:
            kinds = (TIMER,)
        items = self.schedule.for_user(user.id if user else GUEST, kinds)
        now = self.now()
        if not items:
            what = _and([_KIND_NAMES[k] for k in (kinds or (ALARM, TIMER, REMINDER))]).replace(" and ", " or ")
            say(f"You don't have any {what} set." if user else "There aren't any timers running.")
            return
        groups = []
        for kind in (TIMER, ALARM, REMINDER):
            group = [i for i in items if i.kind == kind]
            if not group:
                continue
            parts = [describe(i, now) for i in group[:6]]
            if kind == TIMER:
                groups.append(_cap(_and(parts)) if len(group) == 1 else
                              f"{len(group)} timers: " + _and(parts))
            elif kind == ALARM:
                groups.append(f"{_cap(_plural(len(group), 'alarm'))}: " + _and(parts) if len(group) > 1 else
                              f"An alarm for {parts[0]}")
            else:
                groups.append(f"{_cap(_plural(len(group), 'reminder'))}: " + "; ".join(parts) if len(group) > 1
                              else f"A reminder at {parts[0]}")
        say(". ".join(groups) + ".")

    # ---- cancelling ---------------------------------------------------------------------
    def _cancel(self, text: str, say, dialog: Dialog, user: User | None) -> None:
        kinds = _kinds_in(text) or (ALARM, TIMER, REMINDER)
        if user is None:
            if kinds != (TIMER,) and TIMER not in kinds:
                say("I only change alarms and reminders for the person who set them.")
                return
            kinds = (TIMER,)
        items = self.schedule.for_user(user.id if user else GUEST, kinds)
        what = _and([_KIND_NAMES[k] for k in kinds]).replace(" and ", " or ")
        if not items:
            say(f"You don't have any {what} set.")
            return
        now = self.now()
        if not _ALL.search(text):
            items = _matching(items, text, now)
        if not items:
            say(f"I can't find that one. " + self._options(self.schedule.for_user(user.id if user else GUEST,
                                                                                 kinds), now))
            return
        if len(items) > 1 and not _ALL.search(text):
            answer = dialog.ask("Which one? " + self._options(items, now))
            chosen = _matching(items, answer.text, now) if answer else []
            if answer is not None and _ALL.search(answer.text):
                chosen = items
            if not chosen or len(chosen) > 1 and not (answer and _ALL.search(answer.text)):
                say("Okay, I've left them all.")
                return
            items = chosen
        for i in items:
            self.schedule.delete(i.id)
        if len(items) == 1:
            i = items[0]
            say(f"Cancelled {_the(i, now)}.")
        else:
            say(f"Cancelled {_plural(len(items), 'item' if len({i.kind for i in items}) > 1 else items[0].kind)}.")

    @staticmethod
    def _options(items: list[Item], now: datetime) -> str:
        return f"You have {_and([_the(i, now) for i in items[:5]])}."


_KIND_NAMES = {ALARM: "alarms", TIMER: "timers", REMINDER: "reminders"}


def _kinds_in(text: str) -> tuple[str, ...]:
    return tuple(k for k in (ALARM, TIMER, REMINDER) if re.search(rf"\b{_KIND_WORDS[k]}\b", text, re.I))


def _the(item: Item, now: datetime) -> str:
    due = item.due_at(now.tzinfo)
    if item.kind == TIMER:
        return f"the {timer_name(item)}"
    if item.kind == ALARM:
        return f"the {clock(due)} alarm" + (f" {say_days(item.days)}" if item.repeats else
                                            "" if due.date() == now.date() else f" {say_when(due, now).split(' ', 2)[-1]}")
    return f"the reminder to {reminder_text(item.label)}"


def _matching(items: list[Item], text: str, now: datetime) -> list[Item]:
    """Items a cancel request points at: by time ("the 7 AM one"), name, or words of a reminder."""
    if len(items) == 1 and not parse_clock(text) and not re.search(r"\babout\b|\bto\s+\w", text, re.I):
        return items
    ct = parse_clock(text) or (parse_clock(f"at {text}") if re.search(r"\d|noon|midnight", text) else None)
    if ct is not None:
        if ct.meridiem == "am":
            hours = {ct.hour % 12}
        elif ct.meridiem == "pm":
            hours = {ct.hour % 12 + 12}
        elif ct.hour > 12:
            hours = {ct.hour}
        else:
            hours = {ct.hour % 12, ct.hour % 12 + 12}
        hits = [i for i in items if (i.due_at(now.tzinfo).hour, i.due_at(now.tzinfo).minute) in
                {(h, ct.minute) for h in hours} or (i.hour in hours and i.minute == ct.minute)]
        try:
            day = named_day(text, now.date())
        except ValueError:
            day = None
        if day is not None and len(hits) > 1:
            hits = [i for i in hits if i.due_at(now.tzinfo).date() == day] or hits
        return hits
    words = {w for w in re.findall(r"[a-z']{3,}", text.lower())} - _STOPWORDS
    if words:
        hits = [i for i in items if words & set(re.findall(r"[a-z']{3,}", (i.label or "").lower()))]
        if hits:
            return hits
    for name in ("first", "last", "next"):
        if re.search(rf"\b(?:the )?{name}(?: one)?\b", text, re.I):
            return [items[-1] if name == "last" else items[0]]
    if len(items) == 1:
        return items
    if re.search(r"\b(?:the|my) (?:alarm|timer|reminder)\b", text, re.I) and not words - _KINDLESS:
        return items         # ambiguous: the caller asks which
    return [] if words - _KINDLESS else items


_KINDLESS = {"alarm", "alarms", "timer", "timers", "reminder", "reminders", "one", "please", "now", "orion"}
_STOPWORDS = {"cancel", "delete", "remove", "clear", "turn", "off", "switch", "disable", "stop", "get", "rid",
              "kill", "forget", "drop", "scrap", "the", "my", "about", "that", "this", "for", "and", "you", "can",
              "could", "please", "set", "all", "orion", "hey", "today", "tomorrow", "tonight", "morning",
              "afternoon", "evening", "one", "with", "every", "weekday", "weekdays", "weekend", "weekends",
              "alarm", "alarms", "timer", "timers", "reminder", "reminders", "now", "just", "want", "need",
              "don't", "dont", "anymore", "any", "more", "remind", "o'clock"}


def _without_every(text: str) -> str:
    return re.sub(r"\bevery\s+\w+", " ", text, flags=re.I)


def _from_now(due: datetime, now: datetime) -> str:
    left = due - now
    if left < timedelta(hours=12):
        return f" That's {say_duration(_round_left(left))} from now."
    return ""


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ---- going off ----------------------------------------------------------------------------
class RingIO(Protocol):
    """What the ringer needs from the voice loop or text mode."""

    def say(self, text: str) -> None: ...                  # speak and wait

    def ring(self, volume: float) -> None: ...             # one ring (about 2.4 s), then wait

    def chime(self) -> None: ...

    def hear(self, timeout_s: float) -> str | None: ...    # a short reply, no wake word


class Ringer:
    """Rings alarms and timers until stopped or snoozed; says reminders; reports missed ones."""

    def __init__(self, schedule: Schedule, people: Callable[[str], User | None], *, snooze_minutes: float = 9,
                 ring_minutes: float = 5, volume: float = 0.6, listen_s: float = 3.0,
                 clock: Callable[[], float] | None = None) -> None:
        self.schedule = schedule
        self.people = people
        self.snooze_minutes = snooze_minutes
        self.ring_minutes = ring_minutes
        self.volume = volume
        self.listen_s = listen_s
        self.clock = clock or schedule.clock
        # Called after a morning alarm is stopped: (io, person) -> None (the briefing).
        self.after_morning_alarm: Callable[[RingIO, User], None] | None = None
        self.ringing = False

    def go_off(self, io: RingIO) -> None:
        """Everything due now: reminders are said, alarms and timers ring together."""
        due = self.schedule.due_now(self.clock())
        if not due:
            return
        for item in [i for i in due if i.kind == REMINDER]:
            self._remind(io, item)
        ringing = [i for i in due if i.kind != REMINDER]
        if ringing:
            self._ring(io, ringing)

    def catch_up(self, grace_s: float | None = None) -> list[str]:
        """At start-up: what came due while Orion was off. Recent alarms and timers are left to
        ring; older ones are reported once (repeating alarms move on); reminders are said late."""
        grace = self.ring_minutes * 60 if grace_s is None else grace_s
        now = self.clock()
        tz = local_zone()
        lines = []
        for item in self.schedule.due_now(now):
            if item.kind != REMINDER and now - item.due <= grace:
                continue
            user = self.people(item.user_id)
            name = f", {user.name}" if user else ""
            when = clock(item.due_at(tz))
            if item.kind == ALARM:
                lines.append(f"You missed an alarm at {when}{name}.")
            elif item.kind == TIMER:
                lines.append(f"{_cap(_a(timer_name(item)))} finished at {when} while I was off.")
            else:
                lines.append(f"While I was off{name}, you had a reminder at {when}: {reminder_text(item.label)}.")
            self.schedule.finish(item, now)
        return lines

    def _remind(self, io: RingIO, item: Item) -> None:
        user = self.people(item.user_id)
        io.chime()
        io.say(f"{user.name + ', a' if user else 'A'} reminder: {reminder_text(item.label)}.")
        reply = ring_reply(io.hear(6.0) or "")
        if reply is not None and reply[0] == "snooze":
            minutes = reply[1] or REMINDER_SNOOZE_MIN
            again = self.schedule.snooze(item, minutes, self.clock())
            io.say(f"Okay, I'll remind you again at {clock(again.due_at())}.")
            return
        self.schedule.finish(item, self.clock())

    def _announce(self, items: list[Item]) -> str:
        tz = local_zone()
        now = datetime.fromtimestamp(self.clock(), tz)
        parts = []
        alarms = [i for i in items if i.kind == ALARM]
        if alarms:
            names = [u.name for u in (self.people(i.user_id) for i in alarms) if u]
            names = list(dict.fromkeys(names))
            parts.append(f"It's {clock(now)}" + (f", {_and(names)}." if names else "."))
        for t in (i for i in items if i.kind == TIMER):
            parts.append(f"Your {timer_name(t)} is done." if t.label or t.seconds else "Your timer is done.")
        return " ".join(parts)

    def _ring(self, io: RingIO, items: list[Item]) -> None:
        self.ringing = True
        try:
            io.say(self._announce(items))
            start = self.clock()
            step = 0
            while self.clock() - start < self.ring_minutes * 60:
                io.ring(min(self.volume, 0.25 + step * 0.06))
                step += 1
                reply = ring_reply(io.hear(self.listen_s) or "")
                if reply is None:
                    continue
                if reply[0] == "snooze":
                    minutes = reply[1] or self.snooze_minutes
                    again = None
                    for i in items:
                        again = self.schedule.snooze(i, minutes, self.clock())
                    io.say(f"Snoozing until {clock(again.due_at())}." if again else "Snoozing.")
                    return
                self._stopped(io, items)
                return
            log.info("Nobody stopped the %s; gave up after %s minutes", "/".join(i.kind for i in items),
                     self.ring_minutes)
            for i in items:
                self.schedule.finish(i, self.clock())
        finally:
            self.ringing = False

    def _stopped(self, io: RingIO, items: list[Item]) -> None:
        for i in items:
            self.schedule.finish(i, self.clock())
        morning = [i for i in items if i.kind == ALARM and MORNING[0] <= _set_hour(i) < MORNING[1]]
        if morning and self.after_morning_alarm is not None:
            user = self.people(morning[0].user_id)
            if user is not None:
                self.after_morning_alarm(io, user)
                return
        io.say("Okay." if any(i.kind == ALARM for i in items) else "Okay, timer off.")


def _set_hour(item: Item) -> int:
    return item.hour if item.hour is not None else item.due_at().hour
