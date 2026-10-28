"""Opt-in training notes: what went wrong, never what was said.

When Orion misroutes, picks the wrong tool, or the user says "that was wrong",
a note records the failure type, an optional short description the user chose
to give, and a few whitelisted, non-content facts about the turn (latency,
reply length, model...). Questions, replies, emails, and memories are never
stored. Phase 10 reads these notes to decide which *synthetic* training
examples to write; the notes themselves are never training data.

Stored encrypted under the data directory, so the nuclear wipe removes them
(unlike ~/orion-training/, which holds no personal data).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from storage.encrypted_file import EncryptedJSONFile

WRONG_ANSWER = "wrong_answer"   # user said "that was wrong"
MISROUTE = "misroute"           # router sent a request to the wrong place (Phase 2)
WRONG_TOOL = "wrong_tool"       # wrong tool or invalid arguments (Phase 2+)
KINDS = {WRONG_ANSWER, MISROUTE, WRONG_TOOL}

MAX_NOTE_CHARS = 200
MAX_NOTES = 2000

# Only these context keys are accepted, and only with these value shapes.
# Free text is impossible here by construction.
_CONTEXT: dict[str, tuple[type, ...] | frozenset[str]] = {
    "mode": frozenset({"voice", "text"}),
    "profile": (str,),            # hardware profile name from config.yaml
    "llm_model": (str,),          # e.g. "llama3.2:3b"
    "route": frozenset({"local", "cloud"}),
    "tool": (str,),               # tool *name*, never its arguments
    "speaker_role": frozenset({"owner", "member"}),
    "language": frozenset({"casual", "clean"}),
    "follow_up": (bool,),
    "reply_chars": (int,),
    "reply_sentences": (int,),
    "first_audio_s": (int, float),
    "llm_total_s": (int, float),
}
_SAFE_STR = 64


class TrainingNoteError(ValueError):
    pass


def _check_context(context: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        rule = _CONTEXT.get(key)
        if rule is None:
            raise TrainingNoteError(f"context key {key!r} is not allowed in training notes")
        if isinstance(rule, frozenset):
            if value not in rule:
                raise TrainingNoteError(f"{key}={value!r} is not an allowed value")
        elif not isinstance(value, rule) or (isinstance(value, str) and len(value) > _SAFE_STR):
            raise TrainingNoteError(f"{key} has an invalid value")
        clean[key] = value
    return clean


class TrainingNotes:
    """``file=None`` means notes can't be stored (disabled or no keyring)."""

    def __init__(self, file: EncryptedJSONFile | None, enabled: bool = False) -> None:
        self.file = file
        self.enabled = enabled and file is not None
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if self.file is None or not self.file.exists():
            return []
        return list(self.file.load().get("notes", []))

    def record(self, kind: str, note: str | None = None, **context: Any) -> bool:
        """Append a note. Returns False (and stores nothing) when disabled."""
        if kind not in KINDS:
            raise TrainingNoteError(f"unknown note kind {kind!r}")
        entry = {"ts": round(time.time()), "kind": kind, "context": _check_context(context)}
        if note:
            entry["note"] = " ".join(note.split())[:MAX_NOTE_CHARS]
        if not self.enabled:
            return False
        with self._lock:
            notes = self._load()
            notes.append(entry)
            self.file.save({"version": 1, "notes": notes[-MAX_NOTES:]})
        return True

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._load()

    def clear(self) -> int:
        with self._lock:
            n = len(self._load())
            if self.file is not None and self.file.exists():
                self.file.save({"version": 1, "notes": []})
            return n
