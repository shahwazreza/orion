"""The two-stage (synthesize / play) speaker keeps order and reports idleness correctly."""

import threading
import time

import numpy as np

from audio.tts import Speaker


class SlowTTS:
    sample_rate = 1000

    def synthesize(self, text):
        time.sleep(0.02)
        yield np.full(3, len(text), dtype=np.float32)


class RecordingPlayer:
    def __init__(self):
        self.played = []
        self.lock = threading.Lock()

    def play(self, audio, stop=None):
        time.sleep(0.02)
        with self.lock:
            self.played.append(float(audio[0]))

    def close(self):
        pass


def test_order_preserved_and_wait_blocks_until_done():
    player = RecordingPlayer()
    spk = Speaker(SlowTTS(), player)
    first = []
    spk.begin(on_first_audio=lambda: first.append(len(player.played)))
    spk.play(np.full(3, -1.0, dtype=np.float32))   # chime
    for word in ("a", "bb", "ccc"):
        spk.say(word)
    assert spk.wait(timeout=5)
    assert player.played == [-1.0, 1.0, 2.0, 3.0]
    assert first == [1]   # fired once, when speech (not the chime) started
    spk.close()


def test_failing_tts_does_not_hang_wait():
    class Broken:
        sample_rate = 1000

        def synthesize(self, text):
            raise RuntimeError("boom")
            yield

    spk = Speaker(Broken(), RecordingPlayer())
    spk.say("hello")
    assert spk.wait(timeout=2)
    spk.close()
