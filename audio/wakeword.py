"""Wake word detection with openWakeWord (ONNX backend, works on x86_64 and aarch64)."""

from __future__ import annotations

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


class WakeWordDetector:
    def __init__(self, model_path: Path, models_dir: Path, threshold: float = 0.5,
                 cooldown_s: float = 1.5) -> None:
        from openwakeword.model import Model

        oww = models_dir / "openwakeword"
        for f in (model_path, oww / "melspectrogram.onnx", oww / "embedding_model.onnx"):
            if not f.exists():
                raise FileNotFoundError(f"{f} missing. Run: python scripts/download_models.py")
        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(oww / "melspectrogram.onnx"),
            embedding_model_path=str(oww / "embedding_model.onnx"),
        )
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._last_hit = 0.0
        self.last_score = 0.0

    def detect(self, frame: np.ndarray) -> bool:
        scores = self._model.predict(frame)
        self.last_score = max(scores.values()) if scores else 0.0
        now = time.monotonic()
        if self.last_score >= self.threshold and now - self._last_hit > self.cooldown_s:
            self._last_hit = now
            return True
        return False

    def reset(self) -> None:
        """Clear internal buffers so stale audio can't re-trigger."""
        self._model.reset()
