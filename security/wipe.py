"""Level 2 kill switch: erase everything Orion knows. Silent: nothing is logged.

Order (from the spec):
  1. sign out of online accounts and delete token caches (sign-out hooks;
     Microsoft arrives in Phase 3 and registers one here);
  2. delete the data key from the keyring. This is the step that really
     protects you (crypto-shredding): every encrypted Orion file - database,
     backups, people, notes, metrics - becomes unreadable noise;
  3. overwrite and delete those files;
  4. delete every other Orion secret in the keyring (Anthropic, Brave, and
     whatever later phases add: GitHub, notifications);
  5. the caller exits and Orion stays off.

About step 3: on an SSD, and on btrfs (Fedora's default), overwriting a file
doesn't reliably overwrite the old blocks, because the filesystem and the drive
write new data elsewhere. That's why the key goes first: the old blocks are
encrypted and the key no longer exists. Full-disk encryption (LUKS) covers the
rest of the disk.

Never touched: the code, the models, ~/orion-projects/ and ~/orion-training/
(your builds and training data are yours).
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Callable

from security.keystore import DATA_KEY, SERVICE

# Every keyring entry Orion creates. Later phases add theirs here (a test checks
# that every keyring.set_password in the code is listed). The wipe also searches
# the keyring for anything else filed under "orion".
KNOWN_SECRETS = [DATA_KEY, "anthropic-api-key", "brave-api-key"]

# Phase 3+ register "sign out and forget tokens" steps here.
SIGN_OUT_HOOKS: list[Callable[[], None]] = []

PROTECTED = ("orion-projects", "orion-training")
SEARXNG_CONTAINER = "orion-searxng"


def searxng_config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "orion" / "searxng"


class Wipe:
    def __init__(self, root: Path, data_dir: Path, logs_dir: Path, keyring_module=None,
                 searxng: bool = True, run=subprocess.run) -> None:
        self.root = root.resolve()
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self._keyring = keyring_module
        self._real_keyring = keyring_module is None
        self.searxng = searxng
        self._run = run

    @property
    def keyring(self):
        if self._keyring is None:
            import keyring

            self._keyring = keyring
        return self._keyring

    def run(self) -> None:
        """Every step runs even if an earlier one fails: a half wipe beats none."""
        for hook in SIGN_OUT_HOOKS:                      # 1
            _quietly(hook)
        _quietly(lambda: self._delete_secret(DATA_KEY))  # 2: crypto-shred first
        for d in (self.data_dir, self.logs_dir):         # 3
            _quietly(lambda d=d: self._erase_dir(d))
        if self.searxng:
            _quietly(self._remove_searxng)
        _quietly(self._delete_all_secrets)               # 4

    # ---- keyring ---------------------------------------------------------------------
    def _delete_secret(self, name: str) -> None:
        try:
            self.keyring.delete_password(SERVICE, name)
        except Exception:
            pass        # not there

    def _delete_all_secrets(self) -> None:
        for name in KNOWN_SECRETS:
            self._delete_secret(name)
        if self._real_keyring:               # also sweep anything else filed under "orion"
            import secretstorage

            with closing(secretstorage.dbus_init()) as conn:
                for item in secretstorage.search_items(conn, {"service": SERVICE}):
                    item.delete()

    # ---- files -----------------------------------------------------------------------
    def _erase_dir(self, d: Path) -> None:
        d = d.resolve()
        if not self._safe_to_erase(d) or not d.exists():
            return
        for path in sorted(d.rglob("*"), reverse=True):     # files before their folders
            if path.is_symlink():
                path.unlink()
            elif path.is_file():
                shred(path)
            elif path.is_dir():
                path.rmdir()
        d.rmdir()

    def _safe_to_erase(self, d: Path) -> bool:
        """Only Orion's own folders: inside the Orion directory, never home, never your builds."""
        home = Path.home().resolve()
        if d in (Path("/"), home, self.root) or not d.is_relative_to(self.root):
            return False
        return not any((d / p).exists() or p in d.parts for p in PROTECTED)

    def _remove_searxng(self) -> None:
        if shutil.which("podman") is None:
            return
        self._run(["podman", "rm", "-f", SEARXNG_CONTAINER], capture_output=True, check=False)
        conf = searxng_config_dir()
        if conf.exists():
            # Owned by the container's user (see scripts/searxng.sh), so remove it as that user.
            self._run(["podman", "unshare", "rm", "-rf", str(conf)], capture_output=True, check=False)


def shred(path: Path) -> None:
    """Overwrite with random bytes, flush to disk, then delete. Best effort (see above)."""
    try:
        size = path.stat().st_size
        with open(path, "r+b", buffering=0) as fh:
            left = size
            while left > 0:
                n = min(left, 1 << 20)
                fh.write(secrets.token_bytes(n))
                left -= n
            os.fsync(fh.fileno())
    except OSError:
        pass
    path.unlink(missing_ok=True)


def _quietly(step: Callable[[], None]) -> None:
    try:
        step()
    except Exception:
        pass        # never log during a wipe
