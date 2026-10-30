"""Small encrypted JSON documents (AES-256-GCM).

Format: ``ORN1`` magic | 12-byte random nonce | ciphertext+tag. The file's
purpose is bound in as associated data, so a ciphertext copied over a
different Orion file fails to decrypt instead of being misread.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"ORN1"
NONCE_LEN = 12


class DecryptionError(RuntimeError):
    pass


class EncryptedJSONFile:
    def __init__(self, path: Path, key: bytes, purpose: str) -> None:
        self.path = path
        self._aead = AESGCM(key)
        self._aad = f"orion:{purpose}".encode()

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Any:
        blob = self.path.read_bytes()
        if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + NONCE_LEN + 16:
            raise DecryptionError(f"{self.path.name} is not an Orion encrypted file")
        nonce = blob[len(MAGIC):len(MAGIC) + NONCE_LEN]
        try:
            plain = self._aead.decrypt(nonce, blob[len(MAGIC) + NONCE_LEN:], self._aad)
        except InvalidTag as exc:
            raise DecryptionError(f"{self.path.name}: wrong key or tampered file") from exc
        return json.loads(plain)

    def save(self, data: Any) -> None:
        """Atomic write: readers see either the old file or the new one, never half."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        nonce = secrets.token_bytes(NONCE_LEN)
        blob = MAGIC + nonce + self._aead.encrypt(nonce, json.dumps(data).encode(), self._aad)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


class EncryptedLines:
    """An append-only encrypted log: one AES-GCM record per line (base64), for metrics.

    Appending never rewrites earlier lines, so a crash loses at most the line
    being written. Each record is bound to the file's purpose like EncryptedJSONFile.
    """

    def __init__(self, path: Path, key: bytes, purpose: str) -> None:
        self.path = path
        self._aead = AESGCM(key)
        self._aad = f"orion:{purpose}".encode()

    def append(self, record: Any) -> None:
        import base64

        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_bytes(NONCE_LEN)
        line = base64.b64encode(nonce + self._aead.encrypt(nonce, json.dumps(record).encode(), self._aad))
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as fh:
            fh.write(line + b"\n")

    def read(self) -> list[Any]:
        """All records; lines that fail to decrypt (torn write, wrong key) are skipped."""
        import base64

        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_bytes().splitlines():
            try:
                blob = base64.b64decode(line, validate=True)
                out.append(json.loads(self._aead.decrypt(blob[:NONCE_LEN], blob[NONCE_LEN:], self._aad)))
            except (ValueError, InvalidTag):
                continue
        return out
