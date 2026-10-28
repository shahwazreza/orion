"""Text-to-speech (Kokoro, Piper), audio playback, chimes, and the background speaker."""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from audio.devices import AudioDeviceError, resolve_device
from core.hardware import resolve_onnx_cuda

log = logging.getLogger(__name__)


class TextToSpeech(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield mono float32 audio chunks in [-1, 1]."""
        ...


class PiperTTS:
    def __init__(self, voice_path: Path, use_cuda: bool | str = "auto", length_scale: float = 1.0,
                 speaker_id: int | None = None) -> None:
        from piper import PiperVoice, SynthesisConfig

        if not voice_path.exists():
            raise FileNotFoundError(f"{voice_path} missing. Run: python scripts/download_models.py")
        self._voice = PiperVoice.load(voice_path, use_cuda=resolve_onnx_cuda(use_cuda))
        self._syn = SynthesisConfig(length_scale=length_scale, speaker_id=speaker_id)
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self._voice.synthesize(text, syn_config=self._syn):
            yield chunk.audio_float_array.astype(np.float32, copy=False)


class KokoroTTS:
    """Kokoro-82M via ONNX: far more natural than Piper, but slower on CPU.

    The reply arrives as clause/sentence chunks; Kokoro trims edge silence, so a
    short pause matching the chunk's final punctuation is added back to keep the
    rhythm natural.
    """

    PAUSES = {",": 0.12, ";": 0.18, ":": 0.18, ".": 0.3, "!": 0.3, "?": 0.3}

    def __init__(self, model_path: Path, voices_path: Path, voice: str = "bm_george",
                 lang: str = "en-gb", speed: float = 1.0, use_cuda: bool | str = "auto",
                 threads: int = 4) -> None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        for f in (model_path, voices_path):
            if not f.exists():
                raise FileNotFoundError(f"{f} missing. Run: python scripts/download_models.py")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads   # more threads was slower on a 4-core i5
        providers = ["CPUExecutionProvider"]
        if resolve_onnx_cuda(use_cuda):
            providers.insert(0, "CUDAExecutionProvider")
        session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self._kokoro = Kokoro.from_session(session, str(voices_path))
        if voice not in self._kokoro.get_voices():
            raise ValueError(f"Unknown Kokoro voice '{voice}'")
        self.voice, self.lang, self.speed = voice, lang, speed
        self.sample_rate = 24000

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        audio, sr = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        pause = np.zeros(int(self.PAUSES.get(text.rstrip()[-1:], 0.08) * sr), dtype=np.float32)
        yield np.concatenate([audio.astype(np.float32, copy=False), pause])


def make_tts(tts_cfg: dict[str, Any], models_dir: Path) -> TextToSpeech:
    backend = tts_cfg.get("backend", "kokoro")
    if backend == "kokoro":
        k = tts_cfg.get("kokoro") or {}
        return KokoroTTS(
            models_dir / "kokoro" / k.get("model", "kokoro-v1.0.onnx"),
            models_dir / "kokoro" / "voices-v1.0.bin",
            voice=k.get("voice", "bm_george"),
            lang=k.get("lang", "en-gb"),
            speed=k.get("speed", 1.0),
            use_cuda=tts_cfg.get("use_cuda", "auto"),
            threads=k.get("threads", 4),
        )
    if backend == "piper":
        p = tts_cfg.get("piper") or {}
        return PiperTTS(
            models_dir / "piper" / f"{p.get('voice', 'en_GB-alan-medium')}.onnx",
            use_cuda=tts_cfg.get("use_cuda", "auto"),
            length_scale=p.get("length_scale", 1.0),
            speaker_id=p.get("speaker_id"),
        )
    raise ValueError(f"Unknown TTS backend '{backend}'")


def tone(freqs: list[float], sample_rate: int, note_s: float = 0.09, volume: float = 0.25) -> np.ndarray:
    """A short sequence of soft sine notes with fades, so it doesn't click."""
    n = int(note_s * sample_rate)
    t = np.arange(n) / sample_rate
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
    notes = [np.sin(2 * np.pi * f * t) * fade * volume for f in freqs]
    return np.concatenate(notes).astype(np.float32)


class Chimes:
    """Audible state cues. Phase 2 adds a distinct 'going to the cloud' chime."""

    def __init__(self, sample_rate: int) -> None:
        self.listening = tone([660, 880], sample_rate)    # rising: recording you now
        self.follow_up = tone([880], sample_rate, note_s=0.07, volume=0.15)  # soft: still listening
        self.done = tone([880, 660], sample_rate)         # falling: stopped recording
        self.error = tone([330, 247], sample_rate, note_s=0.15)


class AudioPlayer:
    """Blocking playback through one long-lived output stream."""

    def __init__(self, sample_rate: int, device: Any = None) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=1, dtype="float32",
                device=resolve_device(device, "output"),
            )
            self._stream.start()
        except AudioDeviceError:
            raise
        except Exception as exc:
            raise AudioDeviceError(f"Could not open the speakers: {exc}") from exc

    BLOCK_S = 0.05   # write in small blocks so playback can stop within ~50 ms

    def play(self, audio: np.ndarray, stop: Callable[[], bool] | None = None) -> None:
        if not audio.size:
            return
        block = max(1, int(self.sample_rate * self.BLOCK_S))
        for start in range(0, len(audio), block):
            if stop is not None and stop():
                return
            self._stream.write(audio[start:start + block].reshape(-1, 1))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class Speaker:
    """Speaks text on background threads so the LLM can keep streaming.

    Two stages: one thread synthesizes, the other plays, so the next clause is
    being synthesized while the current one is heard (no gaps with slower TTS
    like Kokoro). Raw audio such as chimes goes through the same queue, so
    everything plays in the order it was queued.

    ``on_first_audio`` fires once per utterance when speech starts, which is how
    the response latency metric is measured.
    """

    _STOP = object()
    _END = object()   # marks the end of one queued item on the playback queue

    def __init__(self, tts: TextToSpeech | None, player: AudioPlayer | None,
                 echo: Callable[[str], None] | None = None) -> None:
        self.tts = tts
        self.player = player
        self.echo = echo
        self._synth_q: queue.Queue[Any] = queue.Queue()
        self._play_q: queue.Queue[Any] = queue.Queue(maxsize=8)
        self._idle = threading.Event()
        self._idle.set()
        self._pending = 0
        self._lock = threading.Lock()
        self._first_audio_cb: Callable[[], None] | None = None
        # Set by interrupt() (barge-in): queued speech is skipped, playback stops.
        self._interrupted = threading.Event()
        # What Orion said most recently, so an "interruption" that is really his
        # own echo can be recognised from its words.
        self.recent_text: deque[str] = deque(maxlen=6)
        self._threads = [
            threading.Thread(target=self._synth_loop, name="speaker-synth", daemon=True),
            threading.Thread(target=self._play_loop, name="speaker-play", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def begin(self, on_first_audio: Callable[[], None] | None = None) -> None:
        self._first_audio_cb = on_first_audio
        self._interrupted.clear()

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def interrupt(self) -> None:
        """Stop talking now: the current chunk stops within ~50 ms and everything
        still queued is dropped. Cleared by begin() or resume()."""
        self._interrupted.set()

    def resume(self) -> None:
        self._interrupted.clear()

    def _enqueue(self, item: Any) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()
        self._synth_q.put(item)

    def say(self, text: str) -> None:
        self._enqueue(text)

    def play(self, audio: np.ndarray) -> None:
        if self.player is not None:
            self._enqueue(audio)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        self._synth_q.put(self._STOP)
        for t in self._threads:
            t.join(timeout=5)
        if self.player is not None:
            self.player.close()

    def _synth_loop(self) -> None:
        while True:
            item = self._synth_q.get()
            if item is self._STOP:
                self._play_q.put(self._STOP)
                return
            try:
                if isinstance(item, np.ndarray):
                    self._play_q.put(("chime", item))
                else:
                    if not self._interrupted.is_set():
                        self.recent_text.append(item)
                        if self.echo:
                            self.echo(item)
                    if (self.tts is not None and self.player is not None
                            and not self._interrupted.is_set()):
                        for chunk in self.tts.synthesize(item):
                            if self._interrupted.is_set():
                                break
                            self._play_q.put(("speech", chunk))
            except Exception:
                log.exception("Speech synthesis failed")
            finally:
                self._play_q.put(self._END)

    def _play_loop(self) -> None:
        while True:
            item = self._play_q.get()
            if item is self._STOP:
                return
            if item is self._END:
                with self._lock:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.set()
                continue
            kind, audio = item
            if self._interrupted.is_set():
                continue    # dropped; its END marker still settles the pending count
            if kind == "speech":
                cb, self._first_audio_cb = self._first_audio_cb, None
                if cb:
                    cb()
            try:
                self.player.play(audio, stop=self._interrupted.is_set)
            except Exception:
                log.exception("Playback failed")
