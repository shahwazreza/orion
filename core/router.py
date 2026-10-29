"""Local vs cloud: Claude mode, one-off "ask Claude", and the optional auto-router.

LOCAL is the default and the answer whenever anything is unclear. The cloud is
used only when the person explicitly asks ("ask Claude ...", "switch to Claude
mode"), or when the auto-router is switched on AND the request is plainly
general knowledge with nothing personal in it. Every decision is logged.

Privacy guard: what's sent to Claude is built from scratch: a persona with no
names, the request itself, and (in Claude mode) that mode's own history. No
memories, past conversations, rules, or names, unless the person says so for
that request ("ask Claude, and include what you know about ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LOCAL, CLOUD = "local", "cloud"

# Only connecting words are dropped ("ask Claude to ...", "about ..."); question words stay,
# or Claude gets "my sister's name is." (seen live).
_ASK = re.compile(r"^(?:hey,? |okay,? |so,? )?(?:can you |could you |please )?(?:ask|check with|run (?:this|that|it) "
                  r"(?:by|past)) claude(?: (?:to|about|whether|if|for))?[,:]?\s*(?P<q>.*)$", re.I)
_MODE_ON = re.compile(r"\b(?:switch|change|go) (?:over )?to claude(?: mode)?\b|\b(?:use|turn on|enable|start|enter) "
                      r"claude mode\b|\bclaude mode on\b|\blet me talk to claude\b", re.I)
_MODE_OFF = re.compile(r"\bback to local\b|\blocal mode\b|\bswitch (?:back )?to local\b|\b(?:turn off|disable|exit|"
                       r"leave|stop) claude mode\b|\bclaude mode off\b|\bstop using claude\b", re.I)
# Explicit, per-request consent to send personal context to Claude.
_INCLUDE = re.compile(r"\b(?:and |, )?(?:include|use|share|send|tell (?:it|claude))\s+(?:what you know|my memories|"
                      r"what i(?:'ve| have) told you|my notes|my (?:name|details|info))(?: about (?P<topic>[^,.?]+))?",
                      re.I)

# Auto-router: only plain general-knowledge questions with nothing personal.
_PERSONAL = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself|we|our|us|you|your|orion|remember|"
                       r"remind|forget|rule|today|tomorrow|tonight|yesterday)\b", re.I)
_GENERAL = re.compile(r"^(?:what|who|when|where|why|how|which|explain|define|describe|summari[sz]e|compare|"
                      r"tell me about|give me an? (?:overview|summary|explanation))\b", re.I)


@dataclass
class Decision:
    route: str              # LOCAL or CLOUD
    reason: str
    text: str               # what to answer (an "ask Claude ..." prefix removed)
    include_topic: str | None = None    # personal context the person allowed for this request
    include_personal: bool = False
    announce: str | None = None         # spoken before answering (mode switches)
    handled: bool = False               # a mode command with nothing further to answer


class Router:
    def __init__(self, auto: bool = False, cloud_available: bool = True) -> None:
        self.auto = auto
        self.cloud_available = cloud_available
        self.mode: dict[str, str] = {}       # per person: LOCAL or CLOUD (Claude mode)

    def in_claude_mode(self, key: str) -> bool:
        return self.mode.get(key) == CLOUD

    def decide(self, text: str, person_key: str) -> Decision:
        t = text.strip()
        if _MODE_OFF.search(t):
            was = self.in_claude_mode(person_key)
            self.mode[person_key] = LOCAL
            return Decision(LOCAL, "mode off", "", handled=True,
                            announce="Back to local. Everything stays on this computer." if was
                            else "I'm already answering locally.")
        if _MODE_ON.search(t):
            if not self.cloud_available:
                return Decision(LOCAL, "cloud unavailable", "", handled=True,
                                announce="Claude mode isn't available: I have no API key or no audit log.")
            self.mode[person_key] = CLOUD
            return Decision(CLOUD, "mode on", "", handled=True,
                            announce="Switching to Claude mode. Your requests now go to the cloud until you "
                                     "say back to local.")

        include = _INCLUDE.search(t)
        bare = (t[:include.start()] + t[include.end():]).strip(" ,") if include else t
        ask = _ASK.match(bare)
        if ask:
            q = ask.group("q").strip(" ,")
            if not q:
                return Decision(LOCAL, "empty ask", "", handled=True, announce="What should I ask Claude?")
            if not self.cloud_available:
                return Decision(LOCAL, "cloud unavailable", q,
                                announce="I can't reach Claude, so I'll answer myself.")
            return Decision(CLOUD, "asked for Claude", q, include_topic=include and include.group("topic"),
                            include_personal=bool(include))

        if self.in_claude_mode(person_key):
            if not self.cloud_available:
                return Decision(LOCAL, "cloud unavailable", t)
            return Decision(CLOUD, "claude mode", bare, include_topic=include and include.group("topic"),
                            include_personal=bool(include))

        if self.auto and self.cloud_available and _GENERAL.search(t) and not _PERSONAL.search(t):
            return Decision(CLOUD, "auto: general knowledge", t)
        return Decision(LOCAL, "default", t)
