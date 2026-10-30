"""Encrypted database: nothing readable on disk, wrong key rejected, backups."""

import secrets
from datetime import date, timedelta

import numpy as np
import pytest

from brain.embeddings import from_blob, rank, to_blob, unit
from storage.db import Database, DatabaseError, derive_db_key


@pytest.fixture
def key():
    return derive_db_key(secrets.token_bytes(32))


def test_contents_are_encrypted_on_disk(tmp_path, key):
    db = Database(tmp_path / "orion.db", key)
    db.execute("INSERT INTO memories (user_id, text, created) VALUES (?, ?, ?)",
               ("u1", "my sister's name is Sara", 1.0))
    db.close()
    raw = (tmp_path / "orion.db").read_bytes()
    assert b"Sara" not in raw and b"memories" not in raw and b"SQLite format" not in raw
    assert oct((tmp_path / "orion.db").stat().st_mode & 0o777) == "0o600"
    again = Database(tmp_path / "orion.db", key)
    assert again.query("SELECT text FROM memories") == [("my sister's name is Sara",)]


def test_wrong_key_is_rejected(tmp_path, key):
    Database(tmp_path / "orion.db", key).close()
    with pytest.raises(DatabaseError):
        Database(tmp_path / "orion.db", derive_db_key(secrets.token_bytes(32)))


def test_db_key_is_not_the_data_key():
    data_key = secrets.token_bytes(32)
    assert derive_db_key(data_key) != data_key and derive_db_key(data_key) == derive_db_key(data_key)


def test_daily_backups_are_encrypted_and_pruned(tmp_path, key):
    db = Database(tmp_path / "orion.db", key)
    db.execute("INSERT INTO rules (user_id, trigger, action, created, updated) VALUES (?,?,?,?,?)",
               ("u1", "leaving my room", "take my keys", 1.0, 1.0))
    start = date(2026, 1, 1)
    for i in range(35):
        db.backup(tmp_path / "backups", keep=30, today=start + timedelta(days=i))
    backups = sorted((tmp_path / "backups").glob("orion-*.db"))
    assert len(backups) == 30 and backups[0].name == "orion-2026-01-06.db"
    assert b"keys" not in backups[-1].read_bytes()
    restored = Database(backups[-1], key)
    assert restored.query("SELECT action FROM rules") == [("take my keys",)]
    assert db.backup(tmp_path / "backups", keep=30, today=start + timedelta(days=34)) == backups[-1]


def test_embedding_round_trip_and_ranking():
    a, b = unit(np.array([1.0, 0.0])), unit(np.array([0.6, 0.8]))
    assert np.allclose(from_blob(to_blob(a)), a)
    q = unit(np.array([0.9, 0.1]))
    ranked = rank(q, [("a", to_blob(a)), ("b", to_blob(b)), ("none", None)], k=5, min_score=0.5)
    assert [item for _, item in ranked] == ["a", "b"]
