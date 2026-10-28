#!/usr/bin/env python3
"""Reproducible evaluation of speaker recognition and end-of-speech detection.

Uses LibriSpeech dev-clean (40 speakers, CC BY 4.0; download from
https://www.openslr.org/12) and the same models and code paths as Orion.

Speaker recognition:
- each speaker is enrolled from ~12 s of clean, close-mic speech (like Orion's
  voice learning: 3 samples, averaged)
- tested on command-length clips (1.2-3 s) under three conditions: clean,
  noisy (10 dB SNR), and far-field (synthetic room reverb, -12 dB, 15 dB SNR)
- households of 3 enrolled people are simulated with Orion's own identify()
  (threshold + margin). Reported per threshold: how often an enrolled person
  is missed or mistaken for someone else, and how often a stranger is accepted.

End-of-speech detection: full sentences in far-field noise, run through
Orion's Endpointer with Silero VAD; a sentence is "cut off" if the endpointer
finishes before the speech actually ends.

    python scripts/voice_eval.py --librispeech /path/to/LibriSpeech/dev-clean
    python scripts/voice_eval.py --librispeech ... --write   # append to docs/benchmarks.md

No audio or embeddings are saved.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.speaker_id import SherpaSpeakerEncoder, identify  # noqa: E402
from audio.vad import Endpointer, SileroVAD, Status  # noqa: E402
from storage.users import User  # noqa: E402

SR = 16000
THRESHOLDS = [0.40, 0.45, 0.50, 0.55]
CONDITIONS = ["clean", "noisy", "far-field"]


def load(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz (LibriSpeech FLAC is already 16 kHz mono)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if stream.rate != SR:
            raise ValueError(f"{path.name}: expected {SR} Hz, got {stream.rate}")
        chunks = [frame.to_ndarray().reshape(-1) for frame in container.decode(stream)]
    x = np.concatenate(chunks)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    f /= np.sqrt(np.arange(1, len(f) + 1))
    x = np.fft.irfft(f, n)
    return (x / (np.std(x) or 1)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = pink_noise(len(x), rng)
    p_sig = np.mean(x ** 2) or 1e-9
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)))
    return (x + noise).astype(np.float32)


def reverb(x: np.ndarray, rng: np.random.Generator, rt60: float = 0.5) -> np.ndarray:
    from scipy.signal import fftconvolve

    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir /= np.max(np.abs(rir))
    y = fftconvolve(x, rir)[: len(x)]
    return (y / (np.max(np.abs(y)) or 1) * np.max(np.abs(x))).astype(np.float32)


def degrade(x: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return x
    if condition == "noisy":
        return add_noise(x, 10, rng)
    return add_noise(reverb(x, rng) * 0.25, 15, rng)     # far-field: -12 dB, reverb, noise


def speakers(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for spk in sorted(p for p in root.iterdir() if p.is_dir()):
        out[spk.name] = sorted(spk.rglob("*.flac"))
    return out


def speaker_eval(data: dict[str, list[Path]], encoder, rng: random.Random, nprng, tests_per: int):
    enrolled: dict[str, User] = {}
    tests: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for spk, files in data.items():
        files = files[:]
        rng.shuffle(files)
        audio, used = [], 0
        while sum(len(a) for a in audio) < 12 * SR and used < len(files):
            audio.append(load(files[used]))
            used += 1
        enroll = np.concatenate(audio)
        third = len(enroll) // 3
        embs = [encoder.embed(enroll[i * third:(i + 1) * third]) for i in range(3)]
        enrolled[spk] = User(id=spk, name=spk, enrollment=[e.tolist() for e in embs])
        for f in files[used:used + tests_per]:
            x = load(f)
            dur = int(rng.uniform(1.2, 3.0) * SR)
            start = rng.randint(0, max(0, len(x) - dur))
            clip = x[start:start + dur]
            for c in CONDITIONS:
                tests[spk][c].append(encoder.embed(degrade(clip, c, nprng)))

    ids = list(enrolled)
    results = {}
    for c in CONDITIONS:
        row = {}
        for thr in THRESHOLDS:
            miss = wrong = gen = fa = imp = 0
            for _ in range(300):                       # random households of 3
                house = rng.sample(ids, 3)
                members = [enrolled[i] for i in house]
                strangers = [i for i in ids if i not in house]
                for spk in house:
                    for e in tests[spk][c]:
                        m = identify(e, members, threshold=thr, margin=0.05).user
                        gen += 1
                        miss += m is None
                        wrong += m is not None and m.id != spk
                for spk in rng.sample(strangers, 3):
                    for e in tests[spk][c]:
                        imp += 1
                        fa += identify(e, members, threshold=thr, margin=0.05).user is not None
            row[thr] = (100 * miss / gen, 100 * wrong / gen, 100 * fa / imp)
        results[c] = row
    return results


def endpoint_eval(data: dict[str, list[Path]], vad_path: Path, rng: random.Random, nprng, n: int):
    settings = {
        "Phase-1 (800 ms, no hysteresis)": dict(end_silence_ms=800, neg_threshold=0.5),
        "voice-patch (1000 ms, hysteresis 0.35)": dict(end_silence_ms=1000, neg_threshold=0.35),
    }
    files = [f for fs in data.values() for f in fs]
    rng.shuffle(files)
    clips = []
    for f in files:
        x = load(f)
        if 3 * SR <= len(x) <= 15 * SR:
            clips.append(x)
        if len(clips) >= n:
            break
    out = {}
    vad = SileroVAD(vad_path)
    for name, kw in settings.items():
        cut = 0
        lag = []
        for x in clips:
            local = np.random.default_rng(hash(len(x)) % 2**32)
            pad = np.concatenate([np.zeros(SR // 2, np.float32), x, np.zeros(3 * SR, np.float32)])
            y = degrade(pad, "far-field", local)
            y = (np.clip(y, -1, 1) * 32767).astype(np.int16)
            frames = y[: len(y) // 1280 * 1280].reshape(-1, 1280)
            ep = Endpointer(vad, chunk_ms=80, threshold=0.5, start_timeout_s=3.0,
                            max_utterance_s=30.0, **_ep_kw(kw))
            done_at = None
            for i, fr in enumerate(frames):
                if ep.push(fr) in (Status.DONE, Status.TIMEOUT):
                    done_at = (i + 1) * 1280
                    break
            speech_end = SR // 2 + len(x)
            if done_at is None or done_at < speech_end - SR // 4:
                cut += 1
            else:
                lag.append((done_at - speech_end) / SR)
        out[name] = (100 * cut / len(clips), float(np.median(lag)) if lag else float("nan"))
    return out, len(clips)


def _ep_kw(kw: dict) -> dict:
    # The Phase-1 Endpointer has no neg_threshold argument; it behaves like neg = threshold.
    import inspect

    if "neg_threshold" not in inspect.signature(Endpointer).parameters:
        kw = {k: v for k, v in kw.items() if k != "neg_threshold"}
    return kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True, type=Path, help="path to LibriSpeech/dev-clean")
    ap.add_argument("--tests-per-speaker", type=int, default=12)
    ap.add_argument("--endpoint-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rng, nprng = random.Random(args.seed), np.random.default_rng(args.seed)
    data = speakers(args.librispeech)
    models = ROOT / "models"
    encoder = SherpaSpeakerEncoder(models / "speaker" / "nemo_en_titanet_small.onnx")
    print(f"{len(data)} speakers; enrolling and embedding test clips...")
    spk = speaker_eval(data, encoder, rng, nprng, args.tests_per_speaker)
    print("end-of-speech detection...")
    ep, nclips = endpoint_eval(data, models / "openwakeword" / "silero_vad.onnx", rng, nprng,
                               args.endpoint_clips)

    lines = [f"\n### {date.today()} - voice evaluation (LibriSpeech dev-clean, seed {args.seed})\n",
             f"TitaNet-small, households of 3 enrolled people, command-length clips (1.2-3 s), "
             f"{args.tests_per_speaker} per speaker per condition. Cells: missed / mistaken for "
             f"another member / stranger accepted, in %.\n",
             "| Condition | " + " | ".join(f"threshold {t:.2f}" for t in THRESHOLDS) + " |",
             "|---|" + "---|" * len(THRESHOLDS)]
    for c, row in spk.items():
        lines.append(f"| {c} | " + " | ".join(f"{m:.1f} / {w:.1f} / {f:.1f}" for m, w, f in row.values())
                     + " |")
    lines += ["", f"End-of-speech on {nclips} far-field sentences:", "",
              "| Setting | Sentences cut off | Median delay after speech ends |", "|---|---|---|"]
    for name, (cut, lag) in ep.items():
        lines.append(f"| {name} | {cut:.1f}% | {lag:.2f} s |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.write:
        with open(ROOT / "docs" / "benchmarks.md", "a", encoding="utf-8") as fh:
            fh.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
