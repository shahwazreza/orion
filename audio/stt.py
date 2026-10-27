"""Speech-to-text behind a swappable interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.hardware import resolve_stt_device

log = logging.getLogger(__name__)


class SpeechToText(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """``audio`` is mono int16 or float32 in [-1, 1]."""
        ...

    def warm_up(self) -> None:
        ...


def to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


class FasterWhisperSTT:
    name = "faster_whisper"

    # Segments Whisper itself thinks are probably not speech are dropped; this
    # catches the classic "Thank you." hallucination on noise.
    NO_SPEECH_CUTOFF = 0.6

    def __init__(self, model: str, device: str, compute_type: str, download_root: Path,
                 language: str | None = "en", beam_size: int = 1, cpu_threads: int = 0) -> None:
        from faster_whisper import WhisperModel

        self.device, self.compute_type = resolve_stt_device(device, compute_type)
        log.info("Loading Whisper '%s' on %s (%s)", model, self.device, self.compute_type)
        kwargs = dict(device=self.device, compute_type=self.compute_type,
                      cpu_threads=cpu_threads, download_root=str(download_root))
        try:
            # Stay offline once the model is cached; otherwise the hub is
            # queried for a newer revision on every start.
            self._model = WhisperModel(model, local_files_only=True, **kwargs)
        except Exception:
            log.info("Whisper '%s' not cached; downloading once from Hugging Face", model)
            self._model = WhisperModel(model, **kwargs)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        segments, _info = self._model.transcribe(
            to_float32(audio),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,               # we already endpointed with Silero
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        parts = [s.text.strip() for s in segments if s.no_speech_prob < self.NO_SPEECH_CUTOFF]
        return " ".join(p for p in parts if p).strip()

    def warm_up(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))


def make_stt(stt_cfg: dict[str, Any], models_dir: Path) -> SpeechToText:
    backend = stt_cfg.get("backend", "faster_whisper")
    if backend == "faster_whisper":
        return FasterWhisperSTT(
            model=stt_cfg.get("model", "small"),
            device=stt_cfg.get("device", "auto"),
            compute_type=stt_cfg.get("compute_type", "auto"),
            download_root=models_dir / "whisper",
            language=stt_cfg.get("language", "en"),
            beam_size=stt_cfg.get("beam_size", 1),
            cpu_threads=stt_cfg.get("cpu_threads", 0),
        )
    raise ValueError(f"Unknown STT backend '{backend}'")
