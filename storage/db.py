"""Orion's encrypted database (SQLCipher): memories, conversations, rules, audit log.

One file, ``data/orion.db``, encrypted with a key derived from the data key in
the system keyring (HKDF with its own label, so the same key material is never
used for two ciphers). Without the keyring entry the file is unreadable; the
Phase 2.5 wipe deletes that entry (crypto-shredding).

Backups are copies of the encrypted file made with SQLite's online backup API,
one per day, the newest ``keep`` kept.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created REAL NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS memories_user ON memories(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    ts REAL NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text TEXT NOT NULL,
    route TEXT NOT NULL DEFAULT 'local',
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS messages_user_ts ON messages(user_id, ts);

CREATE TABLE IF NOT EXISTS summaries (
    user_id TEXT NOT NULL,
    day TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    PRIMARY KEY (user_id, day)
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    action TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS rules_user ON rules(user_id);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class DatabaseError(RuntimeError):
    pass


def derive_db_key(data_key: bytes) -> bytes:
    """A 256-bit SQLCipher key from the Orion data key, labelled for this use only."""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"orion:sqlcipher:v1").derive(data_key)


class Database:
    """Thread-safe SQLCipher connection. ``path=None`` is an in-memory database (tests)."""

    def __init__(self, path: Path | None, key: bytes) -> None:
        import sqlcipher3

        self.path = path
        self._key = key
        self._lock = threading.RLock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.chmod(0o700)
        self._conn = self._open(str(path) if path else ":memory:")
        if path is not None:
            path.chmod(0o600)
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                               (str(SCHEMA_VERSION),))
            self._conn.commit()

    def _open(self, target: str):
        import sqlcipher3

        conn = sqlcipher3.connect(target, check_same_thread=False)
        conn.execute(f"PRAGMA key = \"x'{self._key.hex()}'\"")
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlcipher3.DatabaseError as exc:
            conn.close()
            raise DatabaseError("Can't open the Orion database: wrong key or corrupt file") from exc
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---- queries ----------------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a write; returns lastrowid."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- backups ------------------------------------------------------------
    def backup(self, backup_dir: Path, keep: int = 30, today: date | None = None) -> Path | None:
        """Today's encrypted backup (once per day), pruning to the newest ``keep``."""
        if self.path is None:
            return None
        today = today or date.today()
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.chmod(0o700)
        target = backup_dir / f"orion-{today.isoformat()}.db"
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            dst = self._open(str(tmp))
            try:
                with self._lock:
                    self._conn.backup(dst)
            finally:
                dst.close()
            tmp.chmod(0o600)
            tmp.replace(target)
            log.info("Database backup written: %s", target.name)
        backups = sorted(backup_dir.glob("orion-*.db"))
        for old in backups[:-keep] if keep > 0 else []:
            old.unlink()
        return target


def open_database(data_dir: Path, data_key: bytes) -> Database:
    return Database(data_dir / "orion.db", derive_db_key(data_key))


def now() -> float:
    return time.time()
