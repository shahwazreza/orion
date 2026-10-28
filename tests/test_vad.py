import numpy as np

from audio.vad import Endpointer, Status


class FakeScore:
    def __init__(self, scores):
        self.scores = list(scores)
        self.resets = 0

    def __call__(self, frame):
        return self.scores.pop(0)

    def reset(self):
        self.resets += 1


def frames(n, value=0):
    return [np.full(4, value, dtype=np.int16) for _ in range(n)]


def make(scores, **kw):
    params = dict(chunk_ms=80, start_timeout_s=0.8, end_silence_ms=240, max_utterance_s=2.0, pre_roll_ms=160)
    params.update(kw)
    return Endpointer(FakeScore(scores), **params)


def test_times_out_without_speech():
    ep = make([0.0] * 20)
    statuses = [ep.push(f) for f in frames(10)]
    assert statuses[-1] is Status.TIMEOUT
    assert statuses[-2] is Status.WAITING


def test_ends_after_trailing_silence_and_keeps_pre_roll():
    ep = make([0.1, 0.1, 0.1, 0.9, 0.9, 0.0, 0.0, 0.0])
    data = [np.full(4, i, dtype=np.int16) for i in range(8)]
    statuses = [ep.push(f) for f in data]
    assert statuses[3] is Status.SPEAKING
    assert statuses[-1] is Status.DONE
    audio = ep.audio()
    # pre-roll holds the last two silent frames before onset (1, 2), not frame 0
    assert audio[0] == 1
    # one trailing silent frame trimmed of three
    assert len(audio) == 4 * (2 + 2 + 2)


def test_max_length_cuts_off_monologue():
    ep = make([0.9] * 40)
    statuses = [ep.push(f) for f in frames(40)]
    assert Status.DONE in statuses
    assert statuses.index(Status.DONE) <= 2000 // 80


def test_reset_clears_state_and_vad():
    ep = make([0.9, 0.0, 0.0, 0.0, 0.9])
    for f in frames(4):
        ep.push(f)
    assert ep.status is Status.DONE
    ep.reset()
    assert ep.status is Status.WAITING and ep.audio().size == 0
    assert ep.score.resets == 2  # once in __init__, once now


def test_soft_speech_after_onset_does_not_end_the_utterance():
    # 0.4 is below the 0.5 start threshold but above the 0.35 "still speaking" one.
    ep = make([0.9, 0.4, 0.4, 0.4, 0.4, 0.9, 0.0, 0.0, 0.0])
    statuses = [ep.push(f) for f in frames(9)]
    assert Status.DONE not in statuses[:6]
    old = make([0.9, 0.4, 0.4, 0.4, 0.4, 0.9], neg_threshold=0.5)   # without hysteresis
    assert Status.DONE in [old.push(f) for f in frames(6)]
