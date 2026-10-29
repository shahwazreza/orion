#!/usr/bin/env python3
"""Test wake word models on your own voice and room before switching to one.

Two measurements, for every model at once (e.g. hey_jarvis vs a new hey_orion):

1. Detection: you say the phrase N times when prompted; how often does each
   model fire, and how confidently?
2. False alarms: Orion listens for M minutes while you talk normally, play
   music or a video. Every activation is a false alarm.

    python scripts/wakeword_test.py                       # models from config + hey_orion if present
    python scripts/wakeword_test.py --say 15 --listen 10
    python scripts/wakeword_test.py --models models/openwakeword/hey_orion.onnx

Nothing is recorded to disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.devices import Microphone  # noqa: E402
from audio.wakeword import model_specs, wakeword_model_path  # noqa: E402
from core.config import load_config  # noqa: E402

FRAME = 1280   # 80 ms at 16 kHz


def beep() -> None:
    try:
        import sounddevice as sd

        t = np.arange(int(0.12 * 22050)) / 22050
        sd.play((0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), 22050, blocking=True)
    except Exception:
        print("\a", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model names or .onnx paths (default: config + hey_orion)")
    ap.add_argument("--say", type=int, default=10, help="times to say the phrase (0 to skip)")
    ap.add_argument("--listen", type=float, default=5.0, help="minutes of false-alarm listening (0 to skip)")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config()
    models_dir = cfg.path("paths.models")
    specs = args.models or model_specs(cfg.section("wakeword"))
    custom = models_dir / "openwakeword" / "hey_orion.onnx"
    if not args.models and custom.exists() and str(custom) not in specs:
        specs = [*specs, str(custom)]
    paths = [wakeword_model_path(s, models_dir, cfg.root) for s in specs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Missing:", *missing, sep="\n  ")
        return 1

    from openwakeword.model import Model

    oww = models_dir / "openwakeword"
    model = Model(wakeword_models=[str(p) for p in paths], inference_framework="onnx",
                  melspec_model_path=str(oww / "melspectrogram.onnx"),
                  embedding_model_path=str(oww / "embedding_model.onnx"))
    names = list(model.models)
    mic = Microphone(16000, 80, cfg.get("audio.input_device"))
    mic.start()
    print("Models:", ", ".join(names), f"| threshold {args.threshold}\n")

    if args.say:
        best = {n: [] for n in names}
        for i in range(args.say):
            input(f"[{i + 1}/{args.say}] Press Enter, then after the beep say the wake phrase... ")
            model.reset()
            mic.drain()
            beep()
            mic.drain()
            peak = {n: 0.0 for n in names}
            end = time.monotonic() + 3.0
            while time.monotonic() < end:
                frame = mic.read(timeout=1.0)
                if frame is None:
                    continue
                for n, s in model.predict(frame).items():
                    peak[n] = max(peak[n], float(s))
            for n in names:
                best[n].append(peak[n])
            print("    " + "  ".join(f"{n}: {peak[n]:.2f}" for n in names))
        print("\nDetection (your voice):")
        for n in names:
            s = np.array(best[n])
            print(f"  {n:<24} fired {np.mean(s >= args.threshold):>4.0%}   "
                  f"median score {np.median(s):.2f}   lowest {s.min():.2f}")

    if args.listen:
        input(f"\nFalse alarms: press Enter, then talk normally / play a video for "
              f"{args.listen:g} minutes. Don't say the wake phrase. ")
        model.reset()
        mic.drain()
        hits = {n: 0 for n in names}
        last = {n: 0.0 for n in names}
        end = time.monotonic() + args.listen * 60
        while time.monotonic() < end:
            frame = mic.read(timeout=1.0)
            if frame is None:
                continue
            now = time.monotonic()
            for n, s in model.predict(frame).items():
                if s >= args.threshold and now - last[n] > 1.5:
                    hits[n] += 1
                    last[n] = now
                    print(f"    false alarm: {n} ({s:.2f})")
        print("\nFalse alarms:")
        for n in names:
            print(f"  {n:<24} {hits[n]} in {args.listen:g} min  "
                  f"(= {hits[n] * 60 / args.listen:.1f} per hour)")
    mic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
