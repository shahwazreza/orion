"""Alarms, timers, and reminders: the encrypted store, and the systemd timer that starts Orion.

Everything is kept in the encrypted database (``schedule`` table). The only
plain-text trace is one systemd user timer holding the time of the next thing due
(no person, label, or reminder text), so Orion can be started in time when he
isn't running. The wipe deletes it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from core.when import local_zone, next_on_days
from storage.db import Database

log = logging.getLogger(__name__)

ALARM, TIMER, REMINDER = "alarm", "timer", "reminder"
GUEST = "guest"           # a guest's timers belong to everyone

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('alarm', 'timer', 'reminder')),
    due REAL NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    hour INTEGER,
    minute INTEGER,
    days TEXT,                      -- repeating alarms: weekday numbers, Monday = 0
    seconds INTEGER,                -- a timer's length
    created REAL NOT NULL,
    snoozed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS schedule_due ON schedule(due);
CREATE INDEX IF NOT EXISTS schedule_user ON schedule(user_id);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


@dataclass
class Item:
    id: int
    user_id: str
    kind: str
    due: float                       # epoch seconds
    label: str = ""
    hour: int | None = None
    minute: int | None = None
    days: list[int] | None = None
    seconds: int | None = None
    created: float = 0.0
    snoozed: bool = False

    @property
    def repeats(self) -> bool:
        return bool(self.days)

    def due_at(self, tz=None) -> datetime:
        return datetime.fromtimestamp(self.due, tz or local_zone())


_COLUMNS = "id, user_id, kind, due, label, hour, minute, days, seconds, created, snoozed"


def _item(row) -> Item:
    i, uid, kind, due, label, hour, minute, days, seconds, created, snoozed = row
    return Item(i, uid, kind, due, label, hour, minute,
                [int(d) for d in days.split(",")] if days else None, seconds, created, bool(snoozed))


class Schedule:
    """Thread-safe. ``on_change`` runs after every change (re-arms the systemd timer)."""

    def __init__(self, db: Database, clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.clock = clock
        with db._lock:
            db._conn.executescript(SCHEMA)
            db._conn.commit()
        self.on_change: list[Callable[[Item | None], None]] = []
        self._next: float | None = None
        self._refresh()

    # ---- changes ---------------------------------------------------------------------
    def add(self, user_id: str, kind: str, due: datetime | float, label: str = "", *,
            hour: int | None = None, minute: int | None = None, days: list[int] | None = None,
            seconds: int | None = None) -> Item:
        ts = due.timestamp() if isinstance(due, datetime) else float(due)
        item_id = self.db.execute(
            "INSERT INTO schedule (user_id, kind, due, label, hour, minute, days, seconds, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, kind, ts, label, hour, minute, ",".join(map(str, days)) if days else None, seconds,
             self.clock()))
        self._changed()
        return self.get(item_id)

    def delete(self, item_id: int) -> None:
        self.db.execute("DELETE FROM schedule WHERE id = ?", (item_id,))
        self._changed()

    def delete_user(self, user_id: str) -> int:
        n = len(self.db.query("SELECT id FROM schedule WHERE user_id = ?", (user_id,)))
        self.db.execute("DELETE FROM schedule WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
        self._changed()
        return n

    def finish(self, item: Item, now: float | None = None) -> Item | None:
        """It went off (stopped, given up on, or said). Repeating alarms move to their next
        time and are returned; everything else is deleted."""
        now = self.clock() if now is None else now
        if item.repeats and item.hour is not None:
            after = datetime.fromtimestamp(max(now, item.due if not item.snoozed else now), local_zone())
            nxt = next_on_days(item.hour, item.minute or 0, item.days, after)
            self.db.execute("UPDATE schedule SET due = ?, snoozed = 0 WHERE id = ?", (nxt.timestamp(), item.id))
            self._changed()
            return self.get(item.id)
        self.delete(item.id)
        return None

    def snooze(self, item: Item, minutes: float, now: float | None = None) -> Item:
        now = self.clock() if now is None else now
        self.db.execute("UPDATE schedule SET due = ?, snoozed = 1 WHERE id = ?", (now + minutes * 60, item.id))
        self._changed()
        return self.get(item.id)

    # ---- reading ---------------------------------------------------------------------
    def get(self, item_id: int) -> Item | None:
        rows = self.db.query(f"SELECT {_COLUMNS} FROM schedule WHERE id = ?", (item_id,))
        return _item(rows[0]) if rows else None

    def for_user(self, user_id: str, kinds: tuple[str, ...] | None = None) -> list[Item]:
        """This person's items, soonest first. Guests' timers are everyone's."""
        rows = self.db.query(f"SELECT {_COLUMNS} FROM schedule WHERE user_id IN (?, ?) ORDER BY due",
                             (user_id, GUEST))
        items = [_item(r) for r in rows]
        items = [i for i in items if i.user_id == user_id or i.kind == TIMER]
        return [i for i in items if kinds is None or i.kind in kinds]

    def all(self) -> list[Item]:
        return [_item(r) for r in self.db.query(f"SELECT {_COLUMNS} FROM schedule ORDER BY due")]

    def due_now(self, now: float | None = None) -> list[Item]:
        now = self.clock() if now is None else now
        return [_item(r) for r in self.db.query(f"SELECT {_COLUMNS} FROM schedule WHERE due <= ? ORDER BY due",
                                                (now,))]

    def has_due(self, now: float | None = None) -> bool:
        """Cheap enough to call on every microphone frame: no database query."""
        now = self.clock() if now is None else now
        return self._next is not None and self._next <= now

    def next_due(self) -> Item | None:
        rows = self.db.query(f"SELECT {_COLUMNS} FROM schedule ORDER BY due LIMIT 1")
        return _item(rows[0]) if rows else None

    # ---- settings (weather city, ...) ------------------------------------------------
    def setting(self, user_id: str, key: str) -> str | None:
        rows = self.db.query("SELECT value FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key))
        return rows[0][0] if rows else None

    def set_setting(self, user_id: str, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO user_settings (user_id, key, value) VALUES (?, ?, ?)",
                        (user_id, key, value))

    # ---- internals ---------------------------------------------------------------------
    def _refresh(self) -> Item | None:
        nxt = self.next_due()
        self._next = nxt.due if nxt else None
        return nxt

    def _changed(self) -> None:
        nxt = self._refresh()
        for cb in list(self.on_change):
            try:
                cb(nxt)
            except Exception:
                log.exception("Schedule change hook failed")


# ---- the systemd user timer -----------------------------------------------------------
TIMER_UNIT = "orion-alarm.timer"
SERVICE_UNIT = "orion-alarm.service"


def unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd" / "user"


class WakeupTimer:
    """One systemd user timer, always set for the next thing due (minus ``early_s``),
    which starts Orion if he isn't running. Holds a time and nothing else."""

    def __init__(self, root: Path, *, early_s: float = 90, directory: Path | None = None,
                 python: str = sys.executable, run=subprocess.run) -> None:
        self.root = Path(root).resolve()
        self.early_s = early_s
        self.dir = directory or unit_dir()
        self.python = python
        self._run = run
        self._lock = threading.Lock()

    @property
    def timer_path(self) -> Path:
        return self.dir / TIMER_UNIT

    @property
    def service_path(self) -> Path:
        return self.dir / SERVICE_UNIT

    def service_text(self) -> str:
        return (
            "[Unit]\n"
            "Description=Orion, started in time for the next alarm\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"WorkingDirectory={self.root}\n"
            f"ExecStart={self.python} {self.root / 'main.py'} alarm wake\n"
            # What Orion says and hears is printed to stdout: never into the journal.
            "StandardOutput=null\n"
            "StandardError=journal\n")

    def timer_text(self, start: datetime) -> str:
        return (
            "[Unit]\n"
            "Description=Start Orion for the next alarm\n\n"
            "[Timer]\n"
            f"OnCalendar={start:%Y-%m-%d %H:%M:%S}\n"
            "AccuracySec=1s\n"
            "Persistent=false\n\n"
            "[Install]\n"
            "WantedBy=timers.target\n")

    def sync(self, item: Item | None) -> None:
        """Arm the timer for ``item`` (the next thing due), or remove it when nothing is."""
        with self._lock:
            if item is None:
                self._remove()
                return
            start = datetime.fromtimestamp(item.due - self.early_s, local_zone())
            timer = self.timer_text(start)
            if self.timer_path.exists() and self.timer_path.read_text() == timer and self.service_path.exists():
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            self.service_path.write_text(self.service_text())
            self.timer_path.write_text(timer)
            self._systemctl("daemon-reload")
            self._systemctl("enable", TIMER_UNIT)
            self._systemctl("restart", TIMER_UNIT)
            log.info("Wake-up timer set for %s", f"{start:%a %H:%M:%S}")

    def remove(self) -> None:
        with self._lock:
            self._remove()

    def _remove(self) -> None:
        if not (self.timer_path.exists() or self.service_path.exists()):
            return
        self._systemctl("disable", "--now", TIMER_UNIT)
        for p in (self.timer_path, self.service_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self._systemctl("daemon-reload")

    def _systemctl(self, *args: str) -> None:
        try:
            r = self._run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=10)
            if getattr(r, "returncode", 0) != 0:
                log.warning("systemctl --user %s failed: %s", " ".join(args), (r.stderr or "").strip()[:200])
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("Can't reach systemd (%s); alarms only ring while Orion is running", exc)


def remove_hook(directory: Path | None = None, run=subprocess.run) -> Callable[[], None]:
    """For the wipe: delete the unit files (they hold the next alarm's time)."""
    return lambda: WakeupTimer(Path("."), directory=directory, run=run).remove()
