"""Audio device lookup and the microphone stream."""

from __future__ import annotations

import logging
import queue
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


class AudioDeviceError(RuntimeError):
    pass


def _sd():
    # Imported lazily so text mode and tests work without PortAudio.
    import sounddevice as sd

    return sd


def resolve_device(spec: Any, kind: str) -> int | None:
    """Map a config value (None, index, or name substring) to a device index."""
    if spec is None or spec == "":
        return None
    sd = _sd()
    devices = sd.query_devices()
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    if isinstance(spec, int):
        return spec
    needle = str(spec).lower()
    for idx, dev in enumerate(devices):
        if needle in dev["name"].lower() and dev[channels_key] > 0:
            return idx
    raise AudioDeviceError(f"No {kind} device matching '{spec}'. Run: python main.py --list-devices")


def list_devices() -> str:
    return str(_sd().query_devices())


class Microphone:
    """16 kHz mono int16 capture, delivered in fixed-size frames via a queue."""

    def __init__(self, sample_rate: int = 16000, chunk_ms: int = 80, device: Any = None) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * chunk_ms // 1000
        self.device = resolve_device(device, "input")
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stream = None
        self.overflows = 0
        # Taps see every frame without consuming it (e.g. sampling Orion's echo).
        self.taps: list[Callable[[np.ndarray], None]] = []

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self.overflows += 1
        frame = indata[:, 0].copy()
        for tap in list(self.taps):
            try:
                tap(frame)
            except Exception:
                log.exception("Microphone tap failed")
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.overflows += 1  # consumer stalled; drop rather than block the audio thread

    def start(self) -> None:
        sd = _sd()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_samples,
                device=self.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # PortAudioError and friends
            raise AudioDeviceError(f"Could not open the microphone: {exc}") from exc

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard buffered audio, e.g. Orion's own voice picked up while speaking."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
