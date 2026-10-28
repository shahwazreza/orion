""""That was wrong": turn user feedback into an opt-in training note."""

from __future__ import annotations

import re

from core.dialog import Dialog, Heard
from storage.training_notes import WRONG_ANSWER, TrainingNotes
from storage.users import User

_WRONG = re.compile(
    r"^(?:orion,? |no,? |nope,? )*(?:that was wrong|that'?s wrong|that is wrong|that'?s not right|"
    r"that is not right|that'?s incorrect|wrong answer|you got that wrong|that was a bad answer|"
    r"bad answer)\b", re.I)
_SKIP = re.compile(r"^\W*(skip|no|nope|nothing|never ?mind|no thanks|that's all)\W*$", re.I)


def handle(heard: Heard, dialog: Dialog, user: User | None, notes: TrainingNotes,
           last_turn: dict | None) -> bool:
    """Returns True if the utterance was feedback (whether or not it was logged)."""
    if not _WRONG.search(heard.text.strip()):
        return False
    if last_turn is None:
        dialog.say("Sorry. I haven't answered anything yet, though.")
        return True
    if not notes.enabled or user is None:
        # Strangers can't write notes; with notes off there's nothing to store.
        dialog.say("Sorry about that." + (" Want to try asking another way?" if user else ""))
        return True

    answer = dialog.ask("Sorry about that. I'll note it. In a few words, what went wrong? "
                        "Or say skip.")
    note = None if answer is None or _SKIP.match(answer.text) else answer.text
    notes.record(WRONG_ANSWER, note, speaker_role=user.role, **last_turn)
    dialog.say("Noted. Thanks.")
    return True
