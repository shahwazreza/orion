"""Encrypted audit log: what went where.

Every routing decision, every cloud request exactly as sent (no credentials),
and (with web search) every query and fetched URL. Stored in the SQLCipher
database; read with ``python main.py audit``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from storage.db import Database


class AuditLog:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, kind: str, **detail: Any) -> None:
        self.db.execute("INSERT INTO audit (ts, kind, detail) VALUES (?, ?, ?)",
                        (time.time(), kind, json.dumps(detail, default=str)))

    def recent(self, limit: int = 50, kind: str | None = None) -> list[tuple[float, str, dict]]:
        rows = self.db.query("SELECT ts, kind, detail FROM audit" + (" WHERE kind = ?" if kind else "")
                             + " ORDER BY id DESC LIMIT ?", ((kind, limit) if kind else (limit,)))
        return [(ts, k, json.loads(d)) for ts, k, d in rows]

    def month_cost(self, now: datetime | None = None) -> tuple[int, float]:
        """(cloud requests, estimated USD) this calendar month."""
        now = now or datetime.now()
        start = datetime(now.year, now.month, 1).timestamp()
        n, cost = 0, 0.0
        for (d,) in self.db.query("SELECT detail FROM audit WHERE kind = 'cloud_request' AND ts >= ?", (start,)):
            n += 1
            cost += (json.loads(d).get("usage") or {}).get("cost_usd", 0.0)
        return n, cost
