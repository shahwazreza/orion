"""Per-turn latency and memory metrics.

Only numbers are recorded; transcripts and replies are deliberately excluded.
Rows are still encrypted (logs/metrics.enc, read with `python main.py metrics`),
like every other log Orion keeps.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

log = logging.getLogger(__name__)

_MB = 1024 * 1024


def memory_snapshot() -> dict[str, float]:
    """RSS of this process and of Ollama, plus system RAM/swap, in MB."""
    me = psutil.Process(os.getpid())
    ollama_rss = 0
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            # The server is "ollama"; model runners are e.g. /usr/local/lib/ollama/llama-server.
            argv0 = ((proc.info["cmdline"] or [""])[0] or "").lower()
            name = (proc.info["name"] or "").lower()
            if (name.startswith("ollama") or "ollama" in argv0) and proc.info["memory_info"]:
                ollama_rss += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    return {
        "orion_rss_mb": round(me.memory_info().rss / _MB, 1),
        "ollama_rss_mb": round(ollama_rss / _MB, 1),
        "sys_used_mb": round((vm.total - vm.available) / _MB, 1),
        "sys_total_mb": round(vm.total / _MB, 1),
        "swap_used_mb": round(sw.used / _MB, 1),
    }


@dataclass
class TurnMetrics:
    """Marks named instants within one turn; durations are derived on finish."""

    mode: str
    t0: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter())

    def since(self, start: str, end: str) -> float | None:
        a = self.t0 if start == "start" else self.marks.get(start)
        b = self.marks.get(end)
        return None if a is None or b is None else round(b - a, 3)

    def summary(self) -> dict[str, Any]:
        # "speech_end" is when you stop talking (voice) or press Enter (text).
        return {
            "mode": self.mode,
            "stt_s": self.since("speech_end", "stt_done"),
            "llm_first_token_s": self.since("llm_start", "llm_first_token"),
            "llm_total_s": self.since("llm_start", "llm_done"),
            "first_audio_s": self.since("speech_end", "first_audio"),
            "first_text_s": self.since("speech_end", "llm_first_token"),
            "turn_total_s": self.since("speech_end", "done"),
            **self.extra,
        }


class MetricsSink:
    """Writes one encrypted row per turn (``storage.encrypted_file.EncryptedLines``).
    Without a log (no keyring, or metrics off) rows are only returned, never written."""

    def __init__(self, log_file=None, enabled: bool = True) -> None:
        self.enabled = enabled and log_file is not None
        self.log_file = log_file

    def record(self, turn: TurnMetrics, profile: str) -> dict[str, Any]:
        row = {"ts": round(time.time(), 3), "profile": profile, **turn.summary(), **memory_snapshot()}
        if self.enabled:
            try:
                self.log_file.append(row)
            except OSError as exc:
                log.warning("Could not write metrics: %s", exc)
        return row
