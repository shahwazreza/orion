"""Casual language: when Orion may swear, and keeping it clean when he may not.

Profanity is on by default: Orion mirrors the user's register and never
lectures about language. Clean mode is on whenever *any* reason holds:

- ``config``  profanity: off in config.yaml
- ``user``    "Orion, keep it clean" (until "back to normal")
- ``demo``    --demo, for IT demos and interviews (can't be talked out of)
- ``meeting`` while taking meeting notes (Phase 5)

Speech is the only place swearing is ever allowed. Anything written (emails,
push notifications, commit messages, files) must go through :func:`scrub`.
"""

from __future__ import annotations

import re
import threading

CONFIG, USER, DEMO, MEETING = "config", "user", "demo", "meeting"

# Word stems, matched case-insensitively with any suffix ("fucking", "shitty").
_STEMS = ["fuck", "motherfuck", "shit", "bullshit", "bitch", "bastard", "asshole", "arsehole",
          "dickhead", "cunt", "twat", "wank", "bollock", "piss", "goddamn", "damn", "crap"]
_STANDALONE = ["ass", "arse", "dick", "dicks", "cock", "prick", "pricks", "bloody", "hell",
               "frigging", "effing"]
# Intensifiers can just be dropped ("that's fucking brilliant" -> "that's brilliant").
_INTENSIFIERS = re.compile(r"\b(?:fucking|fuckin|effing|frigging|bloody|goddamn(?:ed)?|damn(?:ed)?|"
                           r"bleeding|sodding)\s+", re.I)
_WORDS = re.compile(
    r"\b(?:" + "|".join(_STEMS) + r")[a-z]*\b|\b(?:" + "|".join(_STANDALONE) + r")\b",
    re.I,
)
# Masked forms models sometimes write: f*ck, f**k, f***, sh*t, b*tch. Run this only
# after markdown emphasis is removed, or "**the**" would look masked.
MASKED = re.compile(r"\b[a-z]{1,3}\*+[a-z]{0,4}(?![\w*])", re.I)


# Whole phrases first, so "holy shit, well done" becomes "wow, well done", not "holy, well done".
_PHRASES = [
    (re.compile(r"\bholy (?:shit|fuck|crap|hell)\b", re.I), "wow"),
    (re.compile(r"\bwhat the (?:fuck|hell|shit|heck)\b", re.I), "what on earth"),
    (re.compile(r"\bbloody hell\b", re.I), "blimey"),
    (re.compile(r"\bno shit\b", re.I), "no kidding"),
    (re.compile(r"\b(?:fuck|screw|sod) (?:it|that)\b", re.I), "forget it"),
    (re.compile(r"\b(?:oh )?(?:fuck|shit|damn|crap)(?= *[,.!?])", re.I), "oh no"),
    (re.compile(r"\b(?:fuck|piss) off\b", re.I), "go away"),
    (re.compile(r"\bshit(?:ty)? (?=(?:day|week|time|luck|weather)\b)", re.I), "rough "),
    # Insult nouns: "being a dick" -> "being a pain", rather than leaving a gap.
    (re.compile(r"\b(?:an?|such an?)(?: total| complete| absolute)? (?:dick(?:head)?|prick|bastard|twat|"
                r"wanker|cunt|ass(?:hole)?|arse(?:hole)?|bitch|knob(?:head)?)\b", re.I), "a pain"),
]


def contains_profanity(text: str) -> bool:
    return bool(_WORDS.search(text) or MASKED.search(text))


def scrub(text: str) -> str:
    """Remove swearing from text. Use on everything Orion writes, and on speech in clean mode."""
    if not contains_profanity(text):
        return text
    for pattern, replacement in _PHRASES:
        text = pattern.sub(lambda m, r=replacement: _match_case(m.group(0), r), text)
    text = _INTENSIFIERS.sub("", text)
    text = _WORDS.sub("", text)
    text = MASKED.sub("", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # tidy gaps left behind
    text = re.sub(r"([,;:])(?=[,.!?;:])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"^[,;:]\s*", "", text)
    return text[:1].upper() + text[1:] if text else text


def _match_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def _parse_setting(value: object) -> bool:
    # YAML 1.1 reads bare on/off as booleans; accept either form.
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("on", "true", "yes", "1")


class ProfanityPolicy:
    def __init__(self, setting: object = "on") -> None:
        self._lock = threading.Lock()
        self._clean_reasons: set[str] = set() if _parse_setting(setting) else {CONFIG}

    def keep_clean(self, reason: str) -> None:
        with self._lock:
            self._clean_reasons.add(reason)

    def release(self, reason: str) -> None:
        with self._lock:
            self._clean_reasons.discard(reason)

    @property
    def reasons(self) -> frozenset[str]:
        return frozenset(self._clean_reasons)

    def allowed(self) -> bool:
        return not self._clean_reasons


_KEEP_CLEAN = re.compile(r"\b(keep it clean|clean (it )?up your language|no (more )?swearing|"
                         r"stop swearing|clean mode|watch your language|language,? please)\b", re.I)
_BACK_TO_NORMAL = re.compile(r"\b(back to normal|normal mode|you can swear( again)?|"
                             r"swear(ing)? (is )?(fine|ok|okay)( again)?)\b", re.I)


def handle_command(text: str, policy: ProfanityPolicy) -> str | None:
    """Mode switches by voice or text. Returns Orion's reply, or None if not a command."""
    if _KEEP_CLEAN.search(text):
        policy.keep_clean(USER)
        return "Understood. Keeping it clean until you say: back to normal."
    if _BACK_TO_NORMAL.search(text):
        policy.release(USER)
        blockers = policy.reasons - {USER}
        if DEMO in blockers:
            return "Demo mode is on, so I'm keeping it clean regardless."
        if MEETING in blockers:
            return "I'll stay clean while I'm taking meeting notes."
        if CONFIG in blockers:
            return "Swearing is switched off in my settings, so I'll stay clean."
        return "Back to normal."
    return None
