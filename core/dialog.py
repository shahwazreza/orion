"""A tiny ask/answer abstraction so onboarding works the same by voice or by keyboard."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class Heard:
    text: str
    audio: np.ndarray | None = None     # in memory only; None in text mode
    sample_rate: int = 16000

    @property
    def seconds(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / self.sample_rate


class Dialog(Protocol):
    hears_voice: bool

    def say(self, text: str) -> None:
        """Speak and wait until finished."""
        ...

    def ask(self, prompt: str, timeout_s: float | None = None) -> Heard | None:
        """Say ``prompt`` and return the reply, or None if nobody answered."""
        ...


class TextDialog:
    hears_voice = False

    def __init__(self, speaker, read: Callable[[str], str] = input) -> None:
        self.speaker = speaker
        self.read = read

    def say(self, text: str) -> None:
        self.speaker.say(text)
        self.speaker.wait()

    def ask(self, prompt: str, timeout_s: float | None = None) -> Heard | None:
        self.say(prompt)
        try:
            line = self.read("You: ").strip()
        except EOFError:
            return None
        return Heard(line) if line else None


_YES = {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "ok", "okay", "please",
        "absolutely", "definitely", "affirmative", "go", "do", "that's", "thats"}
_NO = {"no", "nope", "nah", "not", "don't", "dont", "wrong", "incorrect", "never", "cancel", "stop"}


def yes_no(text: str) -> bool | None:
    """True for a yes, False for a no, None if unclear. 'No' wins over 'yes' ("yes, no wait")."""
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    if not words or len(words) > 8:
        return None
    if any(w in _NO for w in words):
        return False
    if words[0] in _YES or any(w in {"yes", "yeah", "yep", "yup", "correct"} for w in words):
        return True
    return None
