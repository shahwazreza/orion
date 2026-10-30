"""Conversation flow of the voice loop, with scripted fakes for every component."""

import numpy as np
import pytest

from audio.vad import Status
from brain.profanity import ProfanityPolicy
from core.voice_loop import VoiceLoop, is_dismissal
from storage.users import User


class Mic:
    sample_rate = 16000

    def __init__(self, n=10_000):
        self.n = n

    def start(self): pass
    def drain(self): pass
    def close(self): pass

    def read(self, timeout=1.0):
        self.n -= 1
        if self.n < 0:
            raise KeyboardInterrupt
        return np.zeros(1280, dtype=np.int16)


class Wake:
    """Fires once on the first frame, then never again."""
    last_score = 1.0

    def __init__(self):
        self.fired = 0

    def detect(self, frame):
        self.fired += 1
        return self.fired == 1

    def reset(self): pass


class ScriptedEndpointer:
    """Each reset() starts the next scripted utterance: 'speech' or 'silence'."""
    chunk_ms, end_frames = 80, 10

    def __init__(self, script):
        self.script = list(script)
        self.timeouts = []

    def reset(self, start_timeout_s=None):
        self.timeouts.append(start_timeout_s)
        self.current = self.script.pop(0) if self.script else "silence"

    def push(self, frame):
        return Status.DONE if self.current == "speech" else Status.TIMEOUT

    def audio(self):
        return np.ones(16000, dtype=np.int16)


class STT:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, sample_rate=16000):
        return self.texts.pop(0)


class Assistant:
    def __init__(self):
        self.heard = []
        self.profanity = ProfanityPolicy("on")
        self.last_turn = None
        self.web = None

    def respond(self, text, turn, user=None):
        self.heard.append(text)

    def forget_guest(self):
        pass


class FakeIdentity:
    """Everyone is Reza. Records how much audio it was given to judge."""
    current = None

    def __init__(self):
        self.seen_seconds = []

    def start_conversation(self):
        self.current = None

    def observe(self, audio, sample_rate=16000, follow_up=False, unreliable=False):
        self.seen_seconds.append(len(audio) / sample_rate)
        self.current = User(id="r", name="Reza")
        return self.current


class Speaker:
    def __init__(self):
        self.said, self.chimes = [], []

    def say(self, t): self.said.append(t)
    def play(self, a): pass
    def wait(self, timeout=None): return True


class Chimes:
    listening, follow_up, done = "listening", "follow_up", "done"


def run(script, texts, follow_up_s=6.0, identity=None):
    ep, assistant, speaker = ScriptedEndpointer(script), Assistant(), Speaker()
    loop = VoiceLoop(Mic(n=50), Wake(), ep, STT(texts), assistant, speaker, None,
                     identity or FakeIdentity(), None, follow_up_s=follow_up_s, show=lambda s: None)
    with pytest.raises(KeyboardInterrupt):
        loop.run("hey jarvis")
    assert speaker.said[0] == "Orion online."
    speaker.said.pop(0)
    return ep, assistant, speaker


def test_follow_ups_need_no_wake_word():
    ep, assistant, _ = run(["speech", "speech", "speech", "silence"], ["one", "two", "three"])
    assert assistant.heard == ["one", "two", "three"]
    # first turn uses the default timeout, follow-ups use the follow-up window
    assert ep.timeouts == [None, 6.0, 6.0, 6.0]


def test_silence_ends_conversation_quietly():
    ep, assistant, speaker = run(["speech", "silence"], ["hello"])
    assert assistant.heard == ["hello"]
    assert speaker.said == []
    assert len(ep.timeouts) == 2          # no third attempt without a new wake word


def test_dismissal_ends_conversation_without_llm():
    ep, assistant, speaker = run(["speech", "speech", "speech"], ["what time is it", "Thanks, that's all.", "x"])
    assert assistant.heard == ["what time is it"]
    assert speaker.said == ["Any time, Reza."]
    assert len(ep.timeouts) == 2


def test_follow_up_disabled():
    ep, assistant, _ = run(["speech", "speech"], ["one", "two"], follow_up_s=0)
    assert assistant.heard == ["one"]
    assert ep.timeouts == [None]


def test_noise_in_follow_up_window_closes_it_silently():
    ep, assistant, speaker = run(["speech", "speech"], ["hi", ""])
    assert assistant.heard == ["hi"]
    assert speaker.said == []


@pytest.mark.parametrize("text,expected", [
    ("Thanks!", True), ("thank you, Orion.", True), ("That's all", True), ("Never mind.", True),
    ("Thanks, and what's the capital of France?", False), ("stop the timer at five", False),
    ("Thanks, Orion, that's all for now.", True), ("Okay.", True), ("No, I meant Paris.", False),
])
def test_is_dismissal(text, expected):
    assert is_dismissal(text) is expected


def test_keep_it_clean_is_a_command_not_a_question():
    ep, assistant, speaker = run(["speech", "speech", "speech", "silence"],
                                 ["Orion, keep it clean", "what's up", "back to normal"])
    assert assistant.heard == ["what's up"]
    assert speaker.said[0].startswith("Understood. Keeping it clean")
    assert speaker.said[1] == "Back to normal."


def test_wake_word_audio_helps_identify_the_first_utterance_only():
    ident = FakeIdentity()
    run(["speech", "speech", "silence"], ["who am I", "and tomorrow"], identity=ident)
    first, follow = ident.seen_seconds
    assert first == pytest.approx(1.0 + 1280 / 16000)   # utterance + the wake-word frame
    assert follow == pytest.approx(1.0)                 # follow-ups: utterance only
