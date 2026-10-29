"""Requests Orion can't serve yet, answered honestly before the model sees them.

A 3B model told "you have no weather access" will still happily invent a
forecast (seen live: "a high of twenty-five degrees with scattered showers").
For anything that needs live or personal data Orion doesn't have yet, a fixed
answer is the only reliable way to never make things up. Each entry is removed
when the phase that adds the real tool lands; the Phase 2 router takes over.
"""

from __future__ import annotations

import re

# (pattern, what it needs, when it arrives)
_UNSUPPORTED: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(what'?s|how'?s|what is|how is) the weather\b|\bweather (today|tomorrow|tonight|"
                r"this (week|weekend|morning|afternoon|evening)|like|outside|forecast|in [a-z]+)\b|"
                r"\bforecast (for|today|tomorrow)\b|\b(is it|will it) (rain|snow)|"
                r"\b(do i|should i) (need|take|bring) an umbrella\b|\bhow (hot|cold|warm) is it\b"),
     "check the weather", "with the morning briefing"),
    (re.compile(r"\b(my|read|check|any|new|unread|latest) (e-?mails?|inbox|messages?)\b|"
                r"\b(did i get|have i got|do i have) (any )?(e-?mails?|messages?)\b"),
     "read your email", "once Outlook is connected"),
    (re.compile(r"\bmy (calendar|schedule|meetings?|appointments?|agenda)\b|"
                r"\bwhat'?s on (today|tomorrow|this week)\b|\bam i (free|busy) (today|tomorrow|at)\b"),
     "see your calendar", "once Outlook is connected"),
    (re.compile(r"\b(the|today'?s|latest|any) (news|headlines)\b|\bwhat'?s (in the news|happening in the world)\b"),
     "get the news", "when I can go online"),
    (re.compile(r"\b(stock|share) price\b|\bprice of (bitcoin|gold|oil)\b|"
                r"\b(score|result) of (the|last night'?s|today'?s) (game|match)\b|\bwho won (the|last night)\b"),
     "look up live prices or scores", "when I can go online"),
    (re.compile(r"\b(set|start) (an? )?(alarm|timer|reminder)\b|\bwake me (up )?at\b|"
                r"\bremind me (to|at|in \d|tomorrow|tonight|later)\b"),
     "set alarms, timers or reminders", "soon"),
]


# With web search on, these become searchable instead of refused.
_SEARCHABLE = {"check the weather", "get the news", "look up live prices or scores"}


def unsupported(text: str, searchable: bool = False) -> str | None:
    """A spoken, honest refusal if ``text`` needs a capability Orion lacks, else None."""
    t = text.lower()
    for pattern, ability, when in _UNSUPPORTED:
        if searchable and ability in _SEARCHABLE:
            continue
        if pattern.search(t):
            return f"I can't {ability} yet. That's coming {when}, and I won't guess."
    return None
