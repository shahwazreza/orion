"""Barge-in: stop talking when the user talks over Orion.

While Orion speaks, a watcher keeps reading the microphone. Two triggers:

- the wake word ("hey jarvis"): always reliable, works in any room;
- in ``voice`` mode, any speech that isn't Orion's own voice coming back
  through the speakers. Speech is detected with the VAD, then a speaker
  embedding of the last second is compared with Orion's voiceprint (his clean
  TTS voice plus his echo as heard in this room, sampled at startup). Close to
  Orion = echo, ignored; anything else = a person, so he stops.

On a voice interrupt the buffered audio is handed back so the user's words
("no, I meant Paris") become the next request without repeating them.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

OFF, WAKE_WORD, VOICE = "off", "wake_word", "voice"


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


class OwnVoice:
    """Orion's voiceprint: embeddings of his TTS output and of his room echo."""

    def __init__(self) -> None:
        self.prints: list[np.ndarray] = []

    def add(self, embedding: np.ndarray | None) -> None:
        if embedding is not None:
            self.prints.append(unit(np.asarray(embedding, dtype=np.float32)))

    def similarity(self, embedding: np.ndarray) -> float:
        return max((float(embedding @ p) for p in self.prints), default=-1.0)


class BargeIn:
    def __init__(self, mic, wake, vad: Callable[[np.ndarray], float], speaker, mode: str = VOICE,
                 encoder=None, own_voice: OwnVoice | None = None, chunk_ms: int = 80,
                 vad_threshold: float = 0.6, speech_ms: int = 400, echo_threshold: float = 0.5,
                 window_s: float = 1.0, confirm_checks: int = 2) -> None:
        self.mic, self.wake, self.vad, self.speaker = mic, wake, vad, speaker
        self.mode = mode if (mode != VOICE or (encoder is not None and own_voice)) else WAKE_WORD
        self.encoder = encoder
        self.own_voice = own_voice or OwnVoice()
        self.vad_threshold = vad_threshold
        self.speech_frames = max(1, speech_ms // chunk_ms)
        self.echo_threshold = echo_threshold
        self.window = max(1, int(window_s * 1000 / chunk_ms))
        # Orion's echo through laptop speakers occasionally dips low for one check
        # (measured: 0.52 -> 0.31 -> back up); a person talking stays low (0.16, 0.17).
        self.confirm_checks = max(1, confirm_checks)
        self.check_every = max(1, 240 // chunk_ms)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reason: str | None = None          # "wake_word" | "voice" | None
        self.prefill: list[np.ndarray] = []      # user audio to continue listening from
        self.echo_scores: list[float] = []       # for -v diagnostics and calibration

    @property
    def enabled(self) -> bool:
        return self.mode != OFF

    def start(self) -> None:
        if not self.enabled:
            return
        self.reason, self.prefill = None, []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="barge-in", daemon=True)
        self._thread.start()

    def stop(self) -> str | None:
        """Stop watching. Returns why Orion was interrupted, or None."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        return self.reason

    def _trigger(self, reason: str, frames: list[np.ndarray]) -> None:
        self.reason = reason
        self.prefill = frames if reason == VOICE else []
        self.speaker.interrupt()
        log.info("Barge-in (%s): stopped speaking", reason)

    def _run(self) -> None:
        recent: deque[np.ndarray] = deque(maxlen=self.window)
        speech: deque[bool] = deque(maxlen=self.speech_frames + 2)
        since_check = 0
        low_in_a_row = 0
        while not self._stop.is_set():
            frame = self.mic.read(timeout=0.1)
            if frame is None:
                continue
            recent.append(frame)
            if self.wake.detect(frame):
                self._trigger(WAKE_WORD, [])
                return
            if self.mode != VOICE:
                continue
            speech.append(self.vad(frame) >= self.vad_threshold)
            if sum(speech) < self.speech_frames or len(recent) < self.speech_frames:
                continue
            since_check += 1
            if since_check < self.check_every:   # embed every ~240 ms of audio
                continue
            since_check = 0
            emb = self.encoder.embed(np.concatenate(recent))
            score = self.own_voice.similarity(emb)
            self.echo_scores.append(score)
            log.debug("Barge-in check: similarity to Orion's voice %.2f", score)
            low_in_a_row = low_in_a_row + 1 if score < self.echo_threshold else 0
            if low_in_a_row >= self.confirm_checks:
                self._trigger(VOICE, list(recent))
                return


def learn_own_voice(tts, encoder, sentences: list[str]) -> OwnVoice:
    """Orion's clean TTS voiceprint (room echo is added later from the mic)."""
    from scipy.signal import resample_poly

    own = OwnVoice()
    g = np.gcd(16000, tts.sample_rate)
    for text in sentences:
        audio = np.concatenate(list(tts.synthesize(text)))
        own.add(encoder.embed(resample_poly(audio, 16000 // g, tts.sample_rate // g)
                              .astype(np.float32)))
    return own
