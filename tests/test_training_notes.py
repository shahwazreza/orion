"""Training notes: opt-in, encrypted, no conversation content, feedback flow."""

import secrets

import pytest

from core import feedback
from core.dialog import Heard
from storage.encrypted_file import EncryptedJSONFile
from storage.training_notes import (MAX_NOTE_CHARS, MISROUTE, WRONG_ANSWER, TrainingNoteError,
                                    TrainingNotes)
from storage.users import User

LAST = {"mode": "voice", "profile": "laptop", "llm_model": "llama3.2:3b", "language": "casual",
        "follow_up": False, "reply_chars": 42, "reply_sentences": 2, "first_audio_s": 3.1,
        "llm_total_s": 2.2}
REZA = User(id="r", name="Reza", role="owner")


def notes_at(tmp_path, enabled=True, key=None):
    key = key or secrets.token_bytes(32)
    return TrainingNotes(EncryptedJSONFile(tmp_path / "tn.enc", key, "training-notes"), enabled)


class Dialog:
    hears_voice = True

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.said = []

    def say(self, t):
        self.said.append(t)

    def ask(self, t, timeout_s=None):
        self.said.append(t)
        return Heard(self.replies.pop(0)) if self.replies else None


def test_disabled_by_default_stores_nothing(tmp_path):
    n = notes_at(tmp_path, enabled=False)
    assert n.record(WRONG_ANSWER, "too long") is False
    assert not (tmp_path / "tn.enc").exists()


def test_record_is_encrypted_and_truncated(tmp_path):
    n = notes_at(tmp_path)
    assert n.record(WRONG_ANSWER, "way too long   and rambly " * 20, **LAST)
    raw = (tmp_path / "tn.enc").read_bytes()
    assert b"rambly" not in raw and b"llama" not in raw
    entry = n.all()[0]
    assert entry["kind"] == WRONG_ANSWER and len(entry["note"]) == MAX_NOTE_CHARS
    assert entry["context"]["reply_chars"] == 42


@pytest.mark.parametrize("ctx", [
    {"question": "what's in my inbox"},          # unknown key: content can't sneak in
    {"mode": "email"},                           # not an allowed value
    {"reply_chars": "lots"},                     # wrong type
    {"tool": "x" * 500},                         # suspiciously long string
])
def test_context_whitelist_rejects_content(tmp_path, ctx):
    with pytest.raises(TrainingNoteError):
        notes_at(tmp_path).record(MISROUTE, **ctx)


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(TrainingNoteError):
        notes_at(tmp_path).record("vibes")


def test_clear(tmp_path):
    n = notes_at(tmp_path)
    n.record(WRONG_ANSWER)
    n.record(MISROUTE, route="cloud")
    assert n.clear() == 2 and n.all() == []


def test_feedback_flow_records_note_without_content(tmp_path):
    n = notes_at(tmp_path)
    d = Dialog(["it was way too long"])
    assert feedback.handle(Heard("No, that was wrong."), d, REZA, n, LAST)
    e = n.all()[0]
    assert e["note"] == "it was way too long" and e["context"]["speaker_role"] == "owner"
    assert set(e["context"]) <= set(LAST) | {"speaker_role"}
    assert d.said[-1] == "Noted. Thanks."


def test_feedback_skip_records_without_note(tmp_path):
    n = notes_at(tmp_path)
    feedback.handle(Heard("that's wrong"), Dialog(["skip"]), REZA, n, LAST)
    assert "note" not in n.all()[0]


def test_strangers_and_disabled_notes_store_nothing(tmp_path):
    n = notes_at(tmp_path)
    d = Dialog()
    assert feedback.handle(Heard("that was wrong"), d, None, n, LAST)
    off = notes_at(tmp_path / "off", enabled=False)
    assert feedback.handle(Heard("that was wrong"), d, REZA, off, LAST)
    assert n.all() == [] and off.all() == []


@pytest.mark.parametrize("text", ["what's wrong with my code", "is that wrong?", "tell me what was wrong"])
def test_ordinary_questions_are_not_feedback(tmp_path, text):
    assert not feedback.handle(Heard(text), Dialog(), REZA, notes_at(tmp_path), LAST)


def test_assistant_last_turn_fits_the_whitelist(tmp_path):
    """The assistant's real last_turn must always be accepted by the whitelist."""
    from audio.tts import Speaker
    from core.assistant import Assistant
    from core.metrics import MetricsSink, TurnMetrics

    class LLM:
        model = "llama3.2:3b"

        def stream_chat(self, m):
            yield "Paris. Lovely city."

    spk = Speaker(None, None)
    a = Assistant(LLM(), spk, MetricsSink(None, False), "laptop")
    a.respond("capital of France?", TurnMetrics("text"), REZA)
    spk.close()
    assert a.last_turn["reply_sentences"] == 2 and "Paris" not in str(a.last_turn)
    assert notes_at(tmp_path).record(WRONG_ANSWER, None, speaker_role="owner", **a.last_turn)
