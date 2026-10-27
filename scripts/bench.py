#!/usr/bin/env python3
"""Reproducible latency/memory benchmark for the Phase 1 pipeline, no mic needed.

Each prompt is spoken by the configured TTS, resampled to 16 kHz, transcribed by Whisper,
answered by the LLM (streamed), and the first reply sentence synthesized. The
same script runs on the laptop and the Jetson, so numbers are comparable.

    python scripts/bench.py                 # auto-detected profile
    python scripts/bench.py --profile jetson --runs 3 --write

Excludes wake-word detection and the VAD end-silence wait (end_silence_ms in
config.yaml), which add a fixed delay on top of the numbers here.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from audio.stt import make_stt  # noqa: E402
from audio.tts import make_tts  # noqa: E402
from brain.llm import Conversation, make_llm  # noqa: E402
from brain.persona import system_prompt  # noqa: E402
from brain.speech_text import SentenceChunker  # noqa: E402
from core.config import load_config  # noqa: E402
from core.metrics import memory_snapshot  # noqa: E402

# General-knowledge prompts only: no personal data in benchmarks.
PROMPTS = [
    "What is the capital of Australia?",
    "Tell me a fun fact about octopuses.",
    "How far away is the moon?",
    "Give me a one sentence tip for staying focused while studying.",
    "What is the boiling point of water in Fahrenheit?",
]


def speak_to_16k(tts, text: str) -> np.ndarray:
    audio = np.concatenate(list(tts.synthesize(text)))
    g = np.gcd(16000, tts.sample_rate)
    return resample_poly(audio, 16000 // g, tts.sample_rate // g).astype(np.float32)


def run_once(tts, stt, llm, prompt: str) -> dict:
    audio = speak_to_16k(tts, prompt)
    t0 = time.perf_counter()
    heard = stt.transcribe(audio)
    t_stt = time.perf_counter()

    convo = Conversation(system_prompt())  # fresh context per prompt
    chunker = SentenceChunker()
    t_tok = t_audio = None
    first_sentence = None
    for piece in llm.stream_chat(convo.messages_with(heard)):
        if t_tok is None:
            t_tok = time.perf_counter()
        if first_sentence is None:
            first_sentence = next(iter(chunker.feed(piece)), None)
            if first_sentence:
                next(iter(tts.synthesize(first_sentence)))
                t_audio = time.perf_counter()
    t_done = time.perf_counter()
    if t_audio is None:  # reply was a single unterminated sentence
        tail = next(iter(chunker.flush()), "")
        if tail:
            next(iter(tts.synthesize(tail)))
        t_audio = time.perf_counter()

    return {
        "prompt": prompt,
        "heard": heard,
        "audio_s": len(audio) / 16000,
        "stt": t_stt - t0,
        "ttft": (t_tok or t_done) - t_stt,
        "first_audio": t_audio - t0,
        "llm_total": t_done - t_stt,
    }


def cpu_name() -> str:
    """CPU model for the report (never the hostname, which can identify a person)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def tts_voice(cfg) -> str:
    backend = cfg.get("tts.backend")
    return cfg.get(f"tts.{backend}.voice", "?")


def fmt(xs: list[float]) -> str:
    return f"{statistics.median(xs):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--runs", type=int, default=1, help="passes over the prompt list")
    ap.add_argument("--write", action="store_true", help="append results to docs/benchmarks.md")
    args = ap.parse_args()

    cfg = load_config(profile=args.profile)
    models = cfg.path("paths.models")
    print(f"profile={cfg.profile}  loading models...")
    t = time.perf_counter()
    tts = make_tts(cfg.section("tts"), models)
    stt = make_stt(cfg.section("stt"), models)
    stt.warm_up()
    llm = make_llm(cfg.section("llm"))
    llm.warm_up()
    load_s = time.perf_counter() - t

    rows = []
    for _ in range(args.runs):
        for p in PROMPTS:
            r = run_once(tts, stt, llm, p)
            rows.append(r)
            print(f"  stt {r['stt']:.2f}s  ttft {r['ttft']:.2f}s  first audio {r['first_audio']:.2f}s"
                  f"  | heard: {r['heard']!r}")
    mem = memory_snapshot()
    exact = sum(r["heard"].strip().lower().rstrip(".?!") == r["prompt"].lower().rstrip(".?!")
                for r in rows)

    stt_cfg = cfg.section("stt")
    device = getattr(stt, "device", "?")
    compute = getattr(stt, "compute_type", "?")
    report = (
        f"\n### {date.today()} - profile `{cfg.profile}` ({platform.machine()}, "
        f"{cpu_name()})\n\n"
        f"LLM `{cfg.get('llm.model')}` - Whisper `{stt_cfg.get('model')}` on {device}/{compute} - "
        f"TTS `{cfg.get('tts.backend')}` ({tts_voice(cfg)}). {len(rows)} utterances, medians. Models loaded in {load_s:.1f}s.\n\n"
        "| STT | LLM first token | LLM full reply | End of speech to first audio |\n"
        "|---|---|---|---|\n"
        f"| {fmt([r['stt'] for r in rows])} s | {fmt([r['ttft'] for r in rows])} s | "
        f"{fmt([r['llm_total'] for r in rows])} s | **{fmt([r['first_audio'] for r in rows])} s** |\n\n"
        f"Transcripts exactly right: {exact}/{len(rows)}. Memory after run: Orion {mem['orion_rss_mb']:.0f} MB, "
        f"Ollama {mem['ollama_rss_mb']:.0f} MB, system {mem['sys_used_mb']:.0f}/{mem['sys_total_mb']:.0f} MB "
        f"used, swap {mem['swap_used_mb']:.0f} MB.\n"
        f"Add `vad.end_silence_ms` ({cfg.get('vad.end_silence_ms')} ms) for perceived latency in voice mode.\n"
    )
    print(report)
    if args.write:
        out = ROOT / "docs" / "benchmarks.md"
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(report)
        print(f"appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
