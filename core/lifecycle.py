"""Starting and stopping: the level 1 kill switch, mute, and `python main.py stop`.

"Orion, shut down" (or `python main.py stop`, or a desktop shortcut that runs
it) stops Orion at once: speech stops, any Claude or search request is
abandoned, the mic closes, and the process exits. A watchdog guarantees the
exit within a couple of seconds even if something is stuck on the network.

`stop` finds the running Orion through a process file in your private runtime
folder ($XDG_RUNTIME_DIR, readable only by you) and sends it SIGTERM. No
network port is involved.

A deliberate shutdown exits with status 0, so the Phase 9 systemd service
(Restart=on-failure) won't start Orion again.
"""

from __future__ import annotations

import os
import re
import signal
import tempfile
import threading
import time
from pathlib import Path

import psutil


class Shutdown(KeyboardInterrupt):
    """Deliberate stop. A KeyboardInterrupt, so Ctrl+C and this share one exit path."""


class PassphraseHeard(Exception):
    """The wipe passphrase was spoken. Carries nothing: the words are never kept."""


class WipeConfirmed(Exception):
    """Passphrase plus confirmation: main wipes once everything is closed."""


# ---- phrases (matched on the whole utterance, after the wake phrase is removed) -------------
def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower().replace("'", "")))


_ORION = r"(?:(?:hey |ok |okay )?orion )?"
_PLEASE = r"(?: (?:now|please|orion|right now|immediately))*"
_SHUTDOWN = re.compile(rf"^{_ORION}(?:please )?(?:shut ?down|shut (?:yourself|orion) down|"
                       rf"turn (?:yourself |orion )?off|power (?:yourself )?(?:off|down)|switch (?:yourself )?off)"
                       rf"{_PLEASE}$")
_MUTE = re.compile(rf"^{_ORION}(?:please )?(?:stop listening|dont listen|mute(?: yourself)?|"
                   rf"stop listening to (?:me|us))(?: (?:for (?:a while|now|a bit)))?{_PLEASE}$")
_UNMUTE = re.compile(rf"^{_ORION}(?:please )?(?:start listening|(?:you can )?listen (?:again|now)|"
                     rf"unmute(?: yourself)?|you can listen again|start listening again){_PLEASE}$")
# Whisper sometimes hears "wipe" as "wiped" or "wipes"; "confirmed" for "confirm".
_CONFIRM_WIPE = re.compile(r"^(?:(?:yes|yeah|i) )?confirm(?:ed)? (?:the )?wipe[sd]?(?: now| please)*$")


def is_shutdown(text: str) -> bool:
    return bool(_SHUTDOWN.match(_words(text)))


def is_mute(text: str) -> bool:
    return bool(_MUTE.match(_words(text)))


def is_unmute(text: str) -> bool:
    return bool(_UNMUTE.match(_words(text)))


def is_wipe_confirmation(text: str) -> bool:
    return bool(_CONFIRM_WIPE.match(_words(text)))


# ---- the process file -----------------------------------------------------------------------
def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(tempfile.gettempdir(), f"orion-{os.getuid()}")
    d = Path(base) / "orion"
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


class ProcessFile:
    """Records which process is the running Orion, so `stop` and `wipe` can reach it."""

    def __init__(self, directory: Path | None = None) -> None:
        self.path = (directory or runtime_dir()) / "orion.pid"

    def running_pid(self) -> int | None:
        """PID of a live Orion (not just any process that reused the number)."""
        try:
            pid = int(self.path.read_text().strip())
            proc = psutil.Process(pid)
            if pid != os.getpid() and "main.py" in " ".join(proc.cmdline()):
                return pid
        except (OSError, ValueError, psutil.Error):
            pass
        return None

    def claim(self) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))

    def release(self) -> None:
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink()
        except OSError:
            pass


def stop_running(pidfile: ProcessFile, timeout_s: float = 5.0) -> bool | None:
    """Stop the running Orion. True if it stopped, False if it had to be killed, None if none was running."""
    pid = pidfile.running_pid()
    if pid is None:
        return None
    proc = psutil.Process(pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout_s)
        return True
    except psutil.TimeoutExpired:
        proc.kill()
        return False


# ---- signals and the exit watchdog ------------------------------------------------------------
def install_stop_signal() -> None:
    """SIGTERM (from `python main.py stop` or systemd) becomes a Shutdown in the main thread."""
    def handler(signum, frame):
        exit_soon()
        raise Shutdown()

    signal.signal(signal.SIGTERM, handler)


def exit_soon(after_s: float = 3.0, code: int = 0) -> None:
    """Make sure the process ends: clean-up gets ``after_s`` seconds, then a hard exit.
    A web page or Claude stream that won't let go can't keep Orion running."""
    def watchdog():
        time.sleep(after_s)
        os._exit(code)

    threading.Thread(target=watchdog, name="exit-watchdog", daemon=True).start()
