"""The data-encryption key, held in the system keyring (Secret Service / KWallet).

The key never touches config, code, git, or disk in plaintext. Deleting it from
the keyring makes every encrypted Orion file unreadable (crypto-shredding),
which is how the Phase 2.5 wipe will work.
"""

from __future__ import annotations

import base64
import logging
import secrets

log = logging.getLogger(__name__)

SERVICE = "orion"
DATA_KEY = "data-key-v1"


class KeyStoreError(RuntimeError):
    pass


def get_data_key(create: bool = True) -> bytes:
    """Return the 256-bit data key, creating it on first use."""
    try:
        import keyring

        stored = keyring.get_password(SERVICE, DATA_KEY)
        if stored:
            key = base64.b64decode(stored)
            if len(key) != 32:
                raise KeyStoreError("Keyring holds a malformed Orion key")
            return key
        if not create:
            raise KeyStoreError("No Orion key in the keyring")
        key = secrets.token_bytes(32)
        keyring.set_password(SERVICE, DATA_KEY, base64.b64encode(key).decode())
        log.info("Created a new data-encryption key in the system keyring")
        return key
    except KeyStoreError:
        raise
    except Exception as exc:  # no backend, locked wallet, D-Bus missing (headless) ...
        raise KeyStoreError(f"System keyring unavailable: {exc}") from exc
