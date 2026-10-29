"""Wake phrase handling: stripping it from requests, model lists, per-model thresholds."""

import numpy as np
import pytest

from audio.wakeword import WakeWordDetector, model_specs, strip_wake_phrase

PHRASES = ["hey jarvis", "hey orion"]


@pytest.mark.parametrize("heard,expected", [
    ("Hey Jarvis, who am I?", "Who am I?"),
    ("Hey Jarvis who am I", "Who am I"),
    ("Hey Orion, tell me a joke.", "Tell me a joke."),
    ("Hey Ryan, my name is Reza", "My name is Reza"),        # live: "Orion" heard as "Ryan"
    ("hey, javis. What's the time?", "What's the time?"),
    ("Hey Jarvis.", ""),
    ("Hey Sam, look at this", "Hey Sam, look at this"),       # not a wake phrase: untouched
    ("Hey, how does weather form?", "Hey, how does weather form?"),
    ("Tell me about Jarvis Cocker", "Tell me about Jarvis Cocker"),
])
def test_strip_wake_phrase(heard, expected):
    assert strip_wake_phrase(heard, PHRASES) == expected


def test_model_specs_accepts_list_or_old_single_key():
    assert model_specs({"models": ["a.onnx", "hey_jarvis"]}) == ["a.onnx", "hey_jarvis"]
    assert model_specs({"model": "hey_jarvis"}) == ["hey_jarvis"]
    assert model_specs({}) == ["hey_jarvis"]


def test_any_model_can_wake_with_its_own_threshold():
    det = WakeWordDetector.__new__(WakeWordDetector)
    det.threshold, det.thresholds, det.cooldown_s = 0.5, {"hey_orion": 0.7}, 0.0
    det._last_hit, det.last_score, det.last_model = 0.0, 0.0, None

    class FakeModel:
        def __init__(self, scores):
            self.scores = scores

        def predict(self, frame):
            return self.scores

    frame = np.zeros(1280, dtype=np.int16)
    det._model = FakeModel({"hey_jarvis_v0.1": 0.2, "hey_orion": 0.65})
    assert not det.detect(frame)                      # 0.65 < hey_orion's own 0.7
    det._model = FakeModel({"hey_jarvis_v0.1": 0.2, "hey_orion": 0.75})
    assert det.detect(frame) and det.last_model == "hey_orion"
    det._model = FakeModel({"hey_jarvis_v0.1": 0.6, "hey_orion": 0.1})
    assert det.detect(frame) and det.last_model == "hey_jarvis_v0.1"
