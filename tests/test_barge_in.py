"""Barge-in: interrupt triggers, echo rejection, and stopping speech mid-reply."""

import threading
import time

import numpy as np

from audio.tts import Speaker
from core.barge_in import VOICE, WAKE_WORD, BargeIn, OwnVoice
from core.assistant import Assistant
from core.metrics import MetricsSink, TurnMetrics

ORION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
PERSON = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class Mic:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self, timeout=0.1):
        if not self.frames:
            time.sleep(0.01)
            return None
        return self.frames.pop(0)


class Wake:
    def __init__(self, fire_at=None):
        self.n, self.fire_at = 0, fire_at

    def detect(self, frame):
        self.n += 1
        return self.n == self.fire_at


class Encoder:
    """Embeds a frame window as Orion or a person, by the frames' fill value."""

    def embed(self, audio, sample_rate=16000):
        return PERSON if audio.mean() > 0 else ORION


class FakeSpeaker:
    def __init__(self):
        self.stopped = threading.Event()

    def interrupt(self):
        self.stopped.set()


def frames(n, value):
    return [np.full(1280, value, dtype=np.int16) for _ in range(n)]


def watch(mic_frames, wake=None, mode=VOICE, own=None):
    own = own if own is not None else OwnVoice()
    if not own.prints:
        own.add(ORION)
    spk = FakeSpeaker()
    bi = BargeIn(Mic(mic_frames), wake or Wake(), vad=lambda f: 0.9 if f.any() else 0.0,
                 speaker=spk, mode=mode, encoder=Encoder(), own_voice=own, speech_ms=400)
    bi.start()
    spk.stopped.wait(1.0)
    return bi.stop(), bi, spk


def test_person_talking_over_orion_interrupts_and_keeps_their_words():
    reason, bi, spk = watch(frames(40, 5))            # positive audio = a person
    assert reason == VOICE and spk.stopped.is_set()
    assert len(bi.prefill) >= 5                        # their speech so far is handed back


def test_a_single_low_echo_reading_does_not_interrupt():
    """Live: echo scored 0.57, 0.67, 0.66, 0.52, then dipped to 0.31 once."""
    class DippingEncoder:
        scores = iter([0.57, 0.67, 0.66, 0.52, 0.31, 0.6, 0.65, 0.62] + [0.6] * 20)

        def embed(self, audio, sample_rate=16000):
            s = next(self.scores)
            return np.array([s, np.sqrt(1 - s * s), 0], dtype=np.float32)

    own = OwnVoice()
    own.add(ORION)
    spk = FakeSpeaker()
    bi = BargeIn(Mic(frames(60, 5)), Wake(), vad=lambda f: 0.9, speaker=spk, mode=VOICE,
                 encoder=DippingEncoder(), own_voice=own)
    bi.start()
    time.sleep(2.5)
    assert bi.stop() is None and not spk.stopped.is_set()


def test_orions_own_echo_does_not_interrupt():
    reason, bi, spk = watch(frames(20, -5))           # negative audio = Orion's echo
    assert reason is None and not spk.stopped.is_set()
    assert bi.echo_scores and min(bi.echo_scores) > 0.9


def test_silence_does_not_interrupt():
    reason, _, spk = watch(frames(20, 0))
    assert reason is None and not spk.stopped.is_set()


def test_wake_word_always_interrupts_even_over_echo():
    reason, bi, _ = watch(frames(20, -5), wake=Wake(fire_at=3))
    assert reason == WAKE_WORD and bi.prefill == []


def test_wake_word_mode_ignores_plain_speech():
    reason, _, spk = watch(frames(20, 5), mode=WAKE_WORD)
    assert reason is None and not spk.stopped.is_set()


def test_voice_mode_without_voiceprints_falls_back_to_wake_word():
    bi = BargeIn(Mic([]), Wake(), vad=lambda f: 0.0, speaker=FakeSpeaker(), mode=VOICE,
                 encoder=None, own_voice=None)
    assert bi.mode == WAKE_WORD


class SlowPlayer:
    """Plays in 50 ms blocks, like the real AudioPlayer, honouring stop()."""

    def __init__(self):
        self.blocks = 0

    def play(self, audio, stop=None):
        for _ in range(0, len(audio), 50):
            if stop and stop():
                return
            time.sleep(0.005)
            self.blocks += 1

    def close(self):
        pass


class TTS:
    sample_rate = 1000

    def synthesize(self, text):
        yield np.zeros(1000, dtype=np.float32)          # 20 blocks per sentence


def test_interrupt_stops_speech_quickly_and_drops_the_queue():
    player = SlowPlayer()
    spk = Speaker(TTS(), player)
    spk.begin()
    for i in range(5):
        spk.say(f"sentence {i}")
    time.sleep(0.03)
    t = time.monotonic()
    spk.interrupt()
    assert spk.wait(timeout=1.0)
    assert time.monotonic() - t < 0.3
    assert player.blocks < 40                           # nowhere near 5 x 20 blocks
    spk.resume()
    spk.close()


def test_assistant_stops_generating_when_interrupted():
    class LLM:
        model = "fake"
        produced = 0

        def stream_chat(self, messages):
            for i in range(100):
                LLM.produced += 1
                yield f"Sentence number {i}. "

    spk = Speaker(None, None)
    a = Assistant(LLM(), spk, MetricsSink(None, False), "test")

    def interrupt_soon():
        time.sleep(0.01)
        spk.interrupt()

    spk.begin()
    LLM.produced = 0
    turn = TurnMetrics("voice")
    threading.Thread(target=interrupt_soon).start()
    orig = spk.begin
    spk.begin = lambda on_first_audio=None: None        # keep the flag that the thread sets

    class Slow(LLM):
        def stream_chat(self, messages):
            for piece in super().stream_chat(messages):
                time.sleep(0.002)
                yield piece

    a.llm = Slow()
    a.respond("talk forever", turn)
    spk.begin = orig
    assert turn.extra.get("interrupted") is True
    assert LLM.produced < 100
    spk.close()


def test_echo_transcript_is_recognised():
    from core.voice_loop import is_own_echo
    said = ["I'm sorry to hear you're feeling frustrated.",
            "Can I help you with something specific?"]
    assert is_own_echo("your feeling", said)
    assert is_own_echo("help you with something", said)
    assert not is_own_echo("no, tell me about Paris instead", said)
    assert not is_own_echo("delete Joe", said)
