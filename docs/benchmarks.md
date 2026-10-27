# Benchmarks

Measured with `python scripts/bench.py --runs 2 --write` (offline pipeline: Piper speaks
the prompt, Whisper transcribes it, the LLM answers, the first sentence is synthesized).
"End of speech to first audio" is the number that matters: how long after you stop
talking Orion starts answering, excluding the VAD's end-of-speech silence window.

Each section is one run on one device. Compare laptop vs Jetson rows per phase.

## Phase 1 — voice loop

Optimization history (laptop): the first measurement, with Whisper `small` and
sentence-only chunking, gave **4.35 s** to first audio and 9/10 exact transcripts.
Clause-level chunking of the first sentence plus `small.en` brought it to the
result below.


### 2026-10-26 - profile `laptop` (x86_64, Intel(R) Core(TM) i5-10400H CPU @ 2.60GHz)

LLM `llama3.2:3b` - Whisper `small.en` on cpu/int8 - Piper `en_GB-alan-medium`. 10 utterances, medians. Models loaded in 4.5s.

| STT | LLM first token | LLM full reply | End of speech to first audio |
|---|---|---|---|
| 1.44 s | 0.46 s | 2.34 s | **3.18 s** |

Transcripts exactly right: 10/10. Memory after run: Orion 973 MB, Ollama 2580 MB, system 9108/31686 MB used, swap 0 MB.
Add `vad.end_silence_ms` (800 ms) for perceived latency in voice mode.
