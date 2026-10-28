"""Who is Orion talking to right now?

Every voice utterance is matched against enrolled voiceprints. Within a
conversation (follow-ups), a short or borderline utterance keeps the current
speaker rather than flipping to "unknown", because short clips ("yes", "and
tomorrow?") carry too little voice to judge.
"""

from __future__ import annotations

import logging

import numpy as np

from audio.speaker_id import Match, SpeakerEncoder, identify
from storage.users import User, UserStore

log = logging.getLogger(__name__)


class Identity:
    def __init__(self, store: UserStore, encoder: SpeakerEncoder | None, threshold: float = 0.40,
                 margin: float = 0.05, min_audio_s: float = 0.8, continuity_slack: float = 0.1,
                 adapt_threshold: float = 0.72, adapt_min_audio_s: float = 2.0) -> None:
        self.store = store
        self.encoder = encoder
        self.threshold = threshold
        self.margin = margin
        self.min_audio_s = min_audio_s
        self.continuity_slack = continuity_slack
        self.adapt_threshold = adapt_threshold
        self.adapt_min_audio_s = adapt_min_audio_s
        self.current: User | None = None
        self.last_match: Match | None = None

    def start_conversation(self) -> None:
        self.current = None

    def set_current(self, user: User | None) -> None:
        self.current = user

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        if self.encoder is None or len(audio) < self.min_audio_s * sample_rate:
            return None
        return self.encoder.embed(audio, sample_rate)

    def observe(self, audio: np.ndarray, sample_rate: int = 16000, follow_up: bool = False) -> User | None:
        """Update and return the current speaker from one utterance."""
        duration = len(audio) / sample_rate
        emb = self.embed(audio, sample_rate)
        if emb is None:
            # Too short to judge: trust the ongoing conversation, otherwise unknown.
            if not follow_up:
                self.current = None
            return self.current

        match = identify(emb, self.store.all(), self.threshold, self.margin)
        self.last_match = match
        log.debug("Speaker match: %s score=%.2f runner_up=%.2f",
                  match.user.id if match.user else "unknown", match.score, match.runner_up)

        if match.user is not None:
            self.current = match.user
            if match.score >= self.adapt_threshold and duration >= self.adapt_min_audio_s:
                self.store.add_voice_sample(match.user.id, emb)
        elif follow_up and self.current is not None:
            # Keep the current speaker only if this voice is still reasonably close to theirs.
            c = self.current.centroid()
            if c is None or float(emb @ c) < self.threshold - self.continuity_slack:
                self.current = None
        else:
            self.current = None
        return self.current
