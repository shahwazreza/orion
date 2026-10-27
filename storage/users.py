"""People Orion knows: names, roles, and voiceprints, stored encrypted.

A voiceprint is a set of speaker embeddings (fixed-length vectors), never audio
or transcripts. The first person enrolled is the owner; later phases use the
role to decide whose email, calendar, and memory can be touched.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from storage.encrypted_file import EncryptedJSONFile

log = logging.getLogger(__name__)

OWNER, MEMBER = "owner", "member"
SCHEMA_VERSION = 1


@dataclass
class User:
    id: str
    name: str
    role: str = MEMBER
    created: float = field(default_factory=time.time)
    enrollment: list[list[float]] = field(default_factory=list)  # from the enrollment session
    adaptive: list[list[float]] = field(default_factory=list)    # confident matches since
    voice_opt_out: bool = False   # declined voice learning; don't ask again unprompted

    @property
    def is_owner(self) -> bool:
        return self.role == OWNER

    @property
    def has_voice(self) -> bool:
        return bool(self.enrollment)

    def centroid(self) -> np.ndarray | None:
        vecs = self.enrollment + self.adaptive
        if not vecs:
            return None
        c = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        return c / (np.linalg.norm(c) or 1.0)


class UserStore:
    """Thread-safe collection of users. ``file=None`` keeps everything in memory only."""

    def __init__(self, file: EncryptedJSONFile | None, max_adaptive: int = 30) -> None:
        self.file = file
        self.max_adaptive = max_adaptive
        self._lock = threading.Lock()
        self._users: dict[str, User] = {}
        if file is not None and file.exists():
            data = file.load()
            for raw in data.get("users", []):
                u = User(**raw)
                self._users[u.id] = u

    @property
    def persistent(self) -> bool:
        return self.file is not None

    def _save(self) -> None:
        if self.file is not None:
            self.file.save({"version": SCHEMA_VERSION,
                            "users": [asdict(u) for u in self._users.values()]})

    # queries -------------------------------------------------------------
    def all(self) -> list[User]:
        with self._lock:
            return sorted(self._users.values(), key=lambda u: u.created)

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def find(self, name: str) -> User | None:
        needle = name.strip().lower()
        return next((u for u in self.all() if u.name.lower() == needle), None)

    def owner(self) -> User | None:
        return next((u for u in self.all() if u.is_owner), None)

    def is_empty(self) -> bool:
        return not self._users

    # changes -------------------------------------------------------------
    def add(self, name: str, enrollment: list[np.ndarray] | None = None) -> User:
        with self._lock:
            role = OWNER if not any(u.is_owner for u in self._users.values()) else MEMBER
            user = User(id=uuid.uuid4().hex[:12], name=name, role=role,
                        enrollment=[_vec(e) for e in enrollment or []])
            self._users[user.id] = user
            self._save()
        log.info("Added user %s (%s)", user.id, role)
        return user

    def rename(self, user_id: str, name: str) -> None:
        with self._lock:
            self._users[user_id].name = name
            self._save()

    def set_voice(self, user_id: str, enrollment: list[np.ndarray]) -> None:
        with self._lock:
            u = self._users[user_id]
            u.enrollment = [_vec(e) for e in enrollment]
            u.adaptive = []
            self._save()

    def set_opt_out(self, user_id: str, value: bool) -> None:
        with self._lock:
            self._users[user_id].voice_opt_out = value
            self._save()

    def add_voice_sample(self, user_id: str, embedding: np.ndarray) -> None:
        """Remember a confidently-matched utterance so recognition adapts over time
        (colds, new mic, room). Oldest adaptive samples are dropped first."""
        with self._lock:
            u = self._users.get(user_id)
            if u is None:
                return
            u.adaptive.append(_vec(embedding))
            del u.adaptive[:-self.max_adaptive]
            self._save()

    def delete(self, user_id: str) -> User | None:
        with self._lock:
            user = self._users.pop(user_id, None)
            if user and user.is_owner and self._users:
                heir = min(self._users.values(), key=lambda u: u.created)
                heir.role = OWNER
                log.info("Owner removed; %s is now the owner", heir.id)
            self._save()
        return user


def _vec(e: Any) -> list[float]:
    return [round(float(x), 6) for x in np.asarray(e, dtype=np.float32).ravel()]
