"""Speaker matching rules and conversation continuity, with a fake encoder."""

import numpy as np

from audio.speaker_id import identify
from core.identity import Identity
from storage.users import UserStore


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """Returns whatever embedding the test queued, ignoring the audio."""

    def __init__(self):
        self.next = None

    def embed(self, audio, sample_rate=16000):
        return self.next


def two_people():
    s = UserStore(None)
    a = s.add("Ana", [unit(1, 0, 0)])
    b = s.add("Ben", [unit(0, 1, 0)])
    return s, a, b


def test_identify_threshold_and_margin():
    s, a, b = two_people()
    assert identify(unit(1, 0.1, 0), s.all()).user.id == a.id
    assert identify(unit(0, 0, 1), s.all()).user is None                 # nobody close
    assert identify(unit(1, 1, 0), s.all(), threshold=0.5).user is None   # a tie: refuse to guess
    assert identify(unit(1, 0, 0), []).user is None


def make(store):
    enc = FakeEncoder()
    return Identity(store, enc, threshold=0.6, margin=0.05, min_audio_s=0.8,
                    adapt_threshold=0.9, adapt_min_audio_s=2.0), enc


SECOND = np.zeros(16000, dtype=np.int16)
THREE = np.zeros(48000, dtype=np.int16)


def test_short_clip_keeps_speaker_only_within_a_conversation():
    s, a, _ = two_people()
    ident, enc = make(s)
    enc.next = unit(1, 0, 0)
    assert ident.observe(SECOND).id == a.id
    short = np.zeros(8000, dtype=np.int16)            # 0.5 s: too short to judge
    assert ident.observe(short, follow_up=True).id == a.id
    assert ident.observe(short, follow_up=False) is None


def test_borderline_follow_up_keeps_speaker_but_a_stranger_breaks_it():
    s, a, _ = two_people()
    ident, enc = make(s)
    enc.next = unit(1, 0, 0)
    ident.observe(SECOND)
    enc.next = unit(1, 0.9, 0.3)                       # below threshold but near Ana
    enc.next = enc.next / np.linalg.norm(enc.next)
    assert ident.observe(SECOND, follow_up=True).id == a.id
    enc.next = unit(0, 0, 1)                           # clearly someone else
    assert ident.observe(SECOND, follow_up=True) is None


def test_confident_long_matches_adapt_the_voiceprint():
    s, a, _ = two_people()
    ident, enc = make(s)
    enc.next = unit(1, 0.05, 0)
    ident.observe(SECOND)
    assert s.get(a.id).adaptive == []                  # too short to adapt
    ident.observe(THREE)
    assert len(s.get(a.id).adaptive) == 1


def test_talking_over_orion_does_not_demote_the_current_speaker():
    s, a, b = two_people()
    ident, enc = make(s)
    enc.next = unit(1, 0, 0)
    ident.observe(SECOND)                                   # Ana, recognised
    enc.next = unit(0.1, 0.1, 1)                            # her voice mixed with Orion's echo
    enc.next = enc.next / np.linalg.norm(enc.next)
    assert ident.observe(SECOND, follow_up=True, unreliable=True).id == a.id
    enc.next = unit(0, 1, 0)                                # but a clear match still switches
    assert ident.observe(SECOND, follow_up=True, unreliable=True).id == b.id
