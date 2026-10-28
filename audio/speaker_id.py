"""Speaker recognition: who is talking, from a few seconds of their voice.

Each utterance is turned into a speaker embedding (NVIDIA TitaNet-small via
sherpa-onnx, about 30 ms on a laptop CPU) and compared by cosine similarity to
every enrolled user's voiceprint.

This is for personalisation, not security: a recording of someone's voice can
fool it. Nothing sensitive should ever be unlocked by voice match alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from storage.users import User


class SpeakerEncoder(Protocol):
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Unit-length embedding for mono int16 or float32 audio."""
        ...


class SherpaSpeakerEncoder:
    def __init__(self, model_path: Path, threads: int = 2, use_cuda: bool = False) -> None:
        import sherpa_onnx

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing. Run: python scripts/download_models.py")
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=threads, provider="cuda" if use_cuda else "cpu",
        )
        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        stream = self._ext.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()
        e = np.asarray(self._ext.compute(stream), dtype=np.float32)
        return e / (np.linalg.norm(e) or 1.0)


@dataclass(frozen=True)
class Match:
    user: User | None      # None = unknown voice
    score: float           # best cosine similarity
    runner_up: float       # second-best, for the ambiguity check


def identify(embedding: np.ndarray, users: Iterable[User], threshold: float = 0.40,
             margin: float = 0.05) -> Match:
    """Best-matching enrolled user, or None if nobody is close enough.

    ``margin`` rejects near-ties between two enrolled people (e.g. siblings):
    better to say "I'm not sure who that is" than to use the wrong name.
    """
    scored = [(float(embedding @ c), u) for u in users if (c := u.centroid()) is not None]
    if not scored:
        return Match(None, 0.0, 0.0)
    scored.sort(key=lambda s: s[0], reverse=True)
    best, user = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    if best < threshold or best - second < margin:
        return Match(None, best, second)
    return Match(user, best, second)
