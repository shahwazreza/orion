"""Encrypted user store: round trip, tamper/wrong-key detection, roles, voiceprints."""

import os
import secrets

import numpy as np
import pytest

from storage.encrypted_file import DecryptionError, EncryptedJSONFile
from storage.users import OWNER, MEMBER, UserStore


@pytest.fixture
def key():
    return secrets.token_bytes(32)


def store_at(tmp_path, key):
    return UserStore(EncryptedJSONFile(tmp_path / "users.enc", key, "users"))


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_round_trip_and_file_is_not_plaintext(tmp_path, key):
    s = store_at(tmp_path, key)
    s.add("Reza", [unit(1, 0, 0)])
    raw = (tmp_path / "users.enc").read_bytes()
    assert b"Reza" not in raw and raw.startswith(b"ORN1")
    assert oct(os.stat(tmp_path / "users.enc").st_mode & 0o777) == "0o600"
    again = store_at(tmp_path, key)
    assert [u.name for u in again.all()] == ["Reza"]
    assert np.allclose(again.all()[0].centroid(), unit(1, 0, 0), atol=1e-5)


def test_wrong_key_and_tampering_are_detected(tmp_path, key):
    store_at(tmp_path, key).add("Reza")
    with pytest.raises(DecryptionError):
        store_at(tmp_path, secrets.token_bytes(32))
    path = tmp_path / "users.enc"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(bytes(blob))
    with pytest.raises(DecryptionError):
        store_at(tmp_path, key)


def test_ciphertext_bound_to_its_purpose(tmp_path, key):
    EncryptedJSONFile(tmp_path / "x.enc", key, "users").save({"a": 1})
    with pytest.raises(DecryptionError):
        EncryptedJSONFile(tmp_path / "x.enc", key, "memory").load()


def test_first_user_is_owner_and_ownership_passes_on(tmp_path, key):
    s = store_at(tmp_path, key)
    a, b, c = s.add("A"), s.add("B"), s.add("C")
    assert (a.role, b.role, c.role) == (OWNER, MEMBER, MEMBER)
    s.delete(a.id)
    assert s.owner().name == "B"


def test_adaptive_samples_are_capped(tmp_path, key):
    s = UserStore(None, max_adaptive=3)
    u = s.add("A", [unit(1, 0)])
    for i in range(10):
        s.add_voice_sample(u.id, unit(1, i))
    assert len(s.get(u.id).adaptive) == 3
    s.set_voice(u.id, [unit(0, 1)])
    assert s.get(u.id).adaptive == []


def test_find_is_case_insensitive_and_memory_store_is_not_persistent():
    s = UserStore(None)
    s.add("Reza")
    assert s.find("reza").name == "Reza" and not s.persistent
