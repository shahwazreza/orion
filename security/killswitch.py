"""The wipe passphrase: set by the owner, stored only as a salted Argon2id hash.

The passphrase itself is never stored, logged, printed, or committed. The hash
lives inside an encrypted file (data/killswitch.enc), so reading it needs the
data key as well.

Two ways to use it:
- spoken: every utterance is checked (``matches``) before it is shown, logged,
  or saved anywhere. A non-match gives no reaction at all, so there's nothing
  to learn from guessing, and there's no "close" answer.
- typed: ``python main.py wipe`` asks with hidden input (``verify_typed``).
  Three wrong tries lock it for 5 minutes, and each further lockout doubles,
  up to a day. The count is kept in the same encrypted file.

Spoken checks aren't counted: ordinary speech can't be told apart from a
guess, and since a spoken non-match says nothing, guessing aloud teaches
nothing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from storage.encrypted_file import EncryptedJSONFile

# OWASP's recommended Argon2id settings: ~35 ms per check on the laptop, so
# checking every utterance costs nothing you'd notice.
_HASHER = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)

MIN_WORDS = 3
MIN_LETTERS = 12
MAX_TRIES = 3
LOCKOUT_S = 300
MAX_LOCKOUT_S = 86400
_LEADING_NAME = re.compile(r"^(?:hey |ok |okay )?orion ")
_TRAILING_NAME = re.compile(r" orion$")
# Phrases with another job: a passphrase that is one of them would fire both.
_RESERVED = ("confirm wipe", "shut down", "stop listening", "start listening")


def normalize(text: str) -> str:
    """How speech and typing are compared: lowercase words, no punctuation.

    "Purple otter, jumps quietly!" and "purple otter jumps quietly" match, so
    Whisper's punctuation and capitals don't matter. Apostrophes are dropped
    ("don't" == "dont")."""
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def problems_with(passphrase: str) -> list[str]:
    """Why a new passphrase won't work well (empty if it's fine). Never echoes it."""
    norm = normalize(passphrase)
    issues = []
    if len(norm.split()) < MIN_WORDS:
        issues.append(f"use at least {MIN_WORDS} words")
    if sum(c.isalpha() for c in norm) < MIN_LETTERS:
        issues.append(f"use at least {MIN_LETTERS} letters")
    if re.search(r"\d", passphrase):
        issues.append("use words only, no digits (speech recognition may write '4' or 'four')")
    if any(r in norm for r in _RESERVED):
        issues.append("don't include Orion's command phrases (confirm wipe, shut down, stop/start listening)")
    if _LEADING_NAME.match(norm + " ") or norm.endswith(" orion") or norm == "orion":
        issues.append("don't start or end it with 'Orion' (it's stripped from what you say)")
    return issues


@dataclass
class _State:
    hash: str | None = None
    fails: int = 0
    lockouts: int = 0
    locked_until: float = 0.0


class KillSwitch:
    def __init__(self, file: EncryptedJSONFile | None, clock=time.time) -> None:
        self.file = file
        self.clock = clock
        self._state = _State()
        if file is not None and file.exists():
            self._state = _State(**file.load())

    @property
    def is_set(self) -> bool:
        return bool(self._state.hash)

    def set(self, passphrase: str) -> None:
        issues = problems_with(passphrase)
        if issues:
            raise ValueError("; ".join(issues))
        self._state = _State(hash=_HASHER.hash(normalize(passphrase)))
        self._save()

    def matches(self, heard: str) -> bool:
        """True if a spoken utterance is the passphrase (with or without 'Orion' around it)."""
        if not self.is_set:
            return False
        norm = normalize(heard)
        if not norm or len(norm) > 200:
            return False
        candidates = dict.fromkeys([norm, _LEADING_NAME.sub("", norm), _TRAILING_NAME.sub("", norm),
                                    _TRAILING_NAME.sub("", _LEADING_NAME.sub("", norm))])
        return any(self._verify(c) for c in candidates if c)

    def locked_for(self) -> float:
        """Seconds until typed attempts are allowed again (0 if not locked)."""
        return max(0.0, self._state.locked_until - self.clock())

    def verify_typed(self, passphrase: str) -> bool:
        """Rate-limited check for the terminal. A locked switch refuses without checking."""
        if not self.is_set or self.locked_for() > 0:
            return False
        if self._verify(normalize(passphrase)):
            self._state.fails = self._state.lockouts = 0
            self._state.locked_until = 0.0
            self._save()
            return True
        self._state.fails += 1
        if self._state.fails >= MAX_TRIES:
            self._state.locked_until = self.clock() + min(MAX_LOCKOUT_S, LOCKOUT_S * 2 ** self._state.lockouts)
            self._state.lockouts += 1
            self._state.fails = 0
        self._save()
        return False

    def _verify(self, candidate: str) -> bool:
        try:
            return _HASHER.verify(self._state.hash, candidate)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False

    def _save(self) -> None:
        if self.file is not None:
            self.file.save(self._state.__dict__)
