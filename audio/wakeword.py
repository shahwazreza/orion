"""Wake word detection with openWakeWord (ONNX backend, works on x86_64 and aarch64).

Several models can listen at once (e.g. the pretrained "hey jarvis" plus a newly
trained "hey orion" while you try it out); whichever fires wakes Orion.
"""

from __future__ import annotations

import difflib
import re
import time
from pathlib import Path

import numpy as np

BUILTIN_VERSION = "v0.1"


def wakeword_model_path(spec: str, models_dir: Path, root: Path) -> Path:
    """A built-in name like ``hey_jarvis`` or a path to a custom ``.onnx`` model.

    Custom relative paths are resolved against ``root`` (the config directory).
    """
    p = Path(spec).expanduser()
    if p.suffix == ".onnx":
        return p if p.is_absolute() else (root / p)
    return models_dir / "openwakeword" / f"{spec}_{BUILTIN_VERSION}.onnx"


def model_specs(ww_cfg: dict) -> list[str]:
    """``wakeword.models`` (list), falling back to the older single ``wakeword.model``."""
    specs = ww_cfg.get("models") or ww_cfg.get("model") or ["hey_jarvis"]
    return [specs] if isinstance(specs, str) else list(specs)


class WakeWordDetector:
    def __init__(self, model_paths: list[Path] | Path, models_dir: Path, threshold: float = 0.5,
                 cooldown_s: float = 1.5, thresholds: dict[str, float] | None = None) -> None:
        from openwakeword.model import Model

        paths = [model_paths] if isinstance(model_paths, Path) else list(model_paths)
        oww = models_dir / "openwakeword"
        for f in (*paths, oww / "melspectrogram.onnx", oww / "embedding_model.onnx"):
            if not f.exists():
                hint = ("Run: python scripts/download_models.py" if f.parent == oww
                        else "Train it first: see docs/wake-word.md")
                raise FileNotFoundError(f"{f} missing. {hint}")
        self._model = Model(
            wakeword_models=[str(p) for p in paths],
            inference_framework="onnx",
            melspec_model_path=str(oww / "melspectrogram.onnx"),
            embedding_model_path=str(oww / "embedding_model.onnx"),
        )
        self.threshold = threshold
        # Per-model overrides, keyed by file stem (e.g. "hey_orion": 0.6).
        self.thresholds = dict(thresholds or {})
        self.cooldown_s = cooldown_s
        self._last_hit = 0.0
        self.last_score = 0.0
        self.last_model: str | None = None

    def _threshold_for(self, name: str) -> float:
        for key, value in self.thresholds.items():
            if name == key or name.startswith(f"{key}_v") or name.startswith(key):
                return value
        return self.threshold

    def detect(self, frame: np.ndarray) -> bool:
        scores = self._model.predict(frame)
        self.last_score = max(scores.values()) if scores else 0.0
        now = time.monotonic()
        if now - self._last_hit <= self.cooldown_s:
            return False
        for name, score in scores.items():
            if score >= self._threshold_for(name):
                self._last_hit = now
                self.last_model = name
                return True
        return False

    def reset(self) -> None:
        """Clear internal buffers so stale audio can't re-trigger."""
        self._model.reset()


# How Whisper has transcribed the wake phrase at the start of a request.
_WAKE_LEAD = re.compile(r"^\W*(?:hey|hi|hay|a|okay|ok)\W+([a-z'\-]+)\W*", re.I)
_MISHEARD = {"jarvis": {"javis", "jervis", "travis", "jarvi"},
             "orion": {"ryan", "o'ryan", "oryan", "orien", "arian", "o'brien", "orian"}}


def strip_wake_phrase(text: str, phrases: list[str]) -> str:
    """Drop a leading "Hey Jarvis," / "Hey Orion," from a transcript.

    The wake word is how you get Orion's attention, not part of the request.
    Left in, the model sees "Hey Jarvis, ..." and corrects you ("It's Orion,
    not Jarvis").
    """
    m = _WAKE_LEAD.match(text)
    if not m:
        return text
    word = m.group(1).lower()
    for phrase in phrases:
        name = phrase.lower().split()[-1]
        if word == name or word in _MISHEARD.get(name, set()) or \
                difflib.SequenceMatcher(None, word, name).ratio() >= 0.75:
            rest = text[m.end():].strip()
            return rest[:1].upper() + rest[1:] if rest else ""
    return text
