"""Voice activity detection and end-of-utterance detection."""

from __future__ import annotations

from collections import deque
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np


class SileroVAD:
    """Speech probability per frame, using the Silero model shipped with openWakeWord."""

    def __init__(self, model_path: Path) -> None:
        from openwakeword.vad import VAD

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing. Run: python scripts/download_models.py")
        self._vad = VAD(model_path=str(model_path))

    def __call__(self, frame: np.ndarray) -> float:
        # 1280-sample frames are scored as two 40 ms windows.
        return float(self._vad.predict(frame, frame_size=640))

    def reset(self) -> None:
        self._vad.reset_states()


class Status(Enum):
    WAITING = "waiting"    # no speech yet
    SPEAKING = "speaking"
    DONE = "done"          # speech followed by enough silence (or max length)
    TIMEOUT = "timeout"    # nobody spoke


class Endpointer:
    """Collects one utterance: waits for speech, then stops after trailing silence.

    Speech must score ``threshold`` to start, but once someone is talking a frame
    only needs ``neg_threshold`` to count as speech (hysteresis, as in Silero's own
    segmentation). Without it, soft word endings and quiet syllables score just
    under the start threshold, count as silence, and cut people off mid-sentence.
    """

    def __init__(
        self,
        score: Callable[[np.ndarray], float],
        chunk_ms: int = 80,
        threshold: float = 0.5,
        start_timeout_s: float = 6.0,
        end_silence_ms: int = 800,
        max_utterance_s: float = 20.0,
        pre_roll_ms: int = 240,
        neg_threshold: float | None = None,
    ) -> None:
        self.score = score
        self.chunk_ms = chunk_ms
        self.threshold = threshold
        self.neg_threshold = max(0.05, threshold - 0.15) if neg_threshold is None else neg_threshold
        self.default_start_timeout_s = start_timeout_s
        self.end_frames = max(1, end_silence_ms // chunk_ms)
        self.max_frames = int(max_utterance_s * 1000 / chunk_ms)
        self.pre_roll: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_ms // chunk_ms))
        self.reset()

    def reset(self, start_timeout_s: float | None = None) -> None:
        """Start a new utterance. ``start_timeout_s`` overrides the wait for speech once."""
        timeout = self.default_start_timeout_s if start_timeout_s is None else start_timeout_s
        self.start_frames = max(1, int(timeout * 1000 / self.chunk_ms))
        self.status = Status.WAITING
        self._frames: list[np.ndarray] = []
        self._waited = 0
        self._silence = 0
        self.pre_roll.clear()
        reset = getattr(self.score, "reset", None)
        if callable(reset):
            reset()

    def push(self, frame: np.ndarray) -> Status:
        if self.status in (Status.DONE, Status.TIMEOUT):
            return self.status
        p = self.score(frame)

        if self.status is Status.WAITING:
            if p >= self.threshold:
                self.status = Status.SPEAKING
                self._frames = [*self.pre_roll, frame]
            else:
                self.pre_roll.append(frame)
                self._waited += 1
                if self._waited >= self.start_frames:
                    self.status = Status.TIMEOUT
            return self.status

        is_speech = p >= self.neg_threshold
        self._frames.append(frame)
        self._silence = 0 if is_speech else self._silence + 1
        if self._silence >= self.end_frames or len(self._frames) >= self.max_frames:
            self.status = Status.DONE
        return self.status

    def audio(self) -> np.ndarray:
        """The utterance as int16, with most of the trailing silence trimmed."""
        if not self._frames:
            return np.zeros(0, dtype=np.int16)
        keep = len(self._frames) - max(0, self._silence - 2)
        return np.concatenate(self._frames[:keep])
