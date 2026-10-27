"""Make LLM output fit for speech: strip markdown, split into sentences as it streams."""

from __future__ import annotations

import re
from typing import Iterator

from brain.profanity import MASKED

_CODE_FENCE = re.compile(r"```.*?(```|$)", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_EMPHASIS = re.compile(r"(\*{1,3}|_{2,3})(\S.*?\S|\S)\1")
_STRAY = re.compile(r"[*#>|~]+")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]+"
)
_SPACES = re.compile(r"[ \t]+")


def clean_for_speech(text: str) -> str:
    """Remove formatting that would otherwise be read aloud."""
    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = MASKED.sub("", text)        # "f**k" would otherwise be read out as "f k"
    text = _STRAY.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


# Sentence end: terminal punctuation (optionally followed by a closing quote or
# bracket) then whitespace, or a newline. Common abbreviations are protected.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]?\s+|\n+")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e|approx|No)\.$", re.I)


_CLAUSE = re.compile(r"[,;:]\s+|\s+[—–-]\s+")


class SentenceChunker:
    """Accumulates streamed tokens and yields speakable chunks as soon as possible.

    Speaking the first words while the model is still generating the rest is the
    single biggest latency win in the pipeline. Small models like to write one
    long sentence, so when a sentence runs long it is also split at clause
    boundaries (commas, semicolons, dashes). The first chunk splits early to
    start audio fast; later ones only when long, to keep prosody natural.
    """

    def __init__(self, min_chars: int = 2, first_clause_chars: int = 24,
                 clause_chars: int = 110) -> None:
        self.min_chars = min_chars
        self.first_clause_chars = first_clause_chars
        self.clause_chars = clause_chars
        self._buf = ""
        self._emitted = 0

    def _emit(self, text: str) -> Iterator[str]:
        chunk = clean_for_speech(text)
        if chunk:
            self._emitted += 1
            yield chunk

    def feed(self, token: str) -> Iterator[str]:
        self._buf += token
        start = 0
        for m in _BOUNDARY.finditer(self._buf):
            candidate = self._buf[start:m.start()].strip()
            if _ABBREV.search(candidate) or len(candidate) < self.min_chars:
                continue
            start = m.end()
            yield from self._emit(candidate)
        self._buf = self._buf[start:]

        first = self._emitted == 0
        if len(self._buf) < (self.first_clause_chars if first else self.clause_chars):
            return
        cuts = [m for m in _CLAUSE.finditer(self._buf) if m.start() >= self.first_clause_chars]
        if cuts:
            # First chunk: earliest clause (fast start). Later: the longest run (natural flow).
            m = cuts[0] if first else cuts[-1]
            keep_punct = m.group().strip() in ",;:"
            head = self._buf[:m.start() + (1 if keep_punct else 0)]
            self._buf = self._buf[m.end():]
            yield from self._emit(head)

    def flush(self) -> Iterator[str]:
        rest = clean_for_speech(self._buf)
        self._buf = ""
        if rest:
            yield rest
