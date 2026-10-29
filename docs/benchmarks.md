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

## Voice recognition and end-of-speech detection

Measured with `python scripts/voice_eval.py --librispeech .../LibriSpeech/dev-clean`
(LibriSpeech dev-clean, 40 speakers, CC BY 4.0). Same models and code paths as Orion.
These results led to the merged defaults: `people.threshold` 0.55 → 0.40,
`vad.neg_threshold` 0.35, and `vad.end_silence_ms` 800 → 1000.

### 2026-10-28 - voice evaluation (LibriSpeech dev-clean, seed 7)

TitaNet-small, households of 3 enrolled people, command-length clips (1.2-3 s), 12 per speaker per condition. Cells: missed / mistaken for another member / stranger accepted, in %.

| Condition | threshold 0.40 | threshold 0.45 | threshold 0.50 | threshold 0.55 |
|---|---|---|---|---|
| clean | 2.1 / 0.0 / 1.1 | 3.3 / 0.0 / 0.1 | 4.6 / 0.0 / 0.1 | 8.3 / 0.0 / 0.0 |
| noisy | 4.3 / 0.0 / 1.0 | 11.6 / 0.0 / 0.3 | 16.9 / 0.0 / 0.1 | 29.0 / 0.0 / 0.0 |
| far-field | 68.1 / 0.0 / 0.1 | 83.6 / 0.0 / 0.0 | 93.9 / 0.0 / 0.0 | 98.4 / 0.0 / 0.0 |

End-of-speech on 200 far-field sentences:

| Setting | Sentences cut off | Median delay after speech ends |
|---|---|---|
| Phase-1 (800 ms, no hysteresis) | 38.0% | 0.56 s |
| voice-patch (1000 ms, hysteresis 0.35) | 15.0% | 0.79 s |

**Reading the results:**

- **0.40 is the right recognition threshold.** It misses 4% of noisy commands
  instead of 29%. The cost is that about 1 stranger in 100 passes as a household
  member, which is why trust-changing commands require `people.verify_threshold`
  (0.55) on the request itself; no stranger reached that level here.
- **No one was ever mistaken for another household member**, at any threshold.
- **Far-field audio defeats TitaNet-small** at every threshold. The simulation
  is harsh (synthetic reverb, -12 dB, 15 dB SNR), but it shows the limit: speak
  within about a metre of the mic, or use a speakerphone. A larger embedding
  model is the next thing to try.
- **Hysteresis plus 1000 ms silence cuts early cut-offs from 38% to 15%** for
  about 0.23 s of extra delay before Orion answers.
- **Live check:** on the laptop with the mic at 100%, Reza scored 0.57 to 0.76
  when recognised. Other voices peaked at about 0.39, consistent with 0.40 as
  the cutoff.

## Phase 2 — memory, rules, web search (laptop, 2026-10-29)

Design decisions in Phase 2 were measured on the real models (llama3.2:3b, nomic-embed-text)
before choosing thresholds or approaches.

### Memory and conversation recall (nomic-embed-text)

| Matching | Related | Unrelated | Threshold chosen |
|---|---|---|---|
| Short topics vs memories ("my sister", "nuts") | 0.59 - 0.76 | 0.42 - 0.53 | recall/forget 0.56 |
| Full requests vs memories ("can I eat this peanut butter sandwich?") | 0.52 | up to 0.53 | context 0.52 (model told to use only if relevant) |
| Topics vs past questions ("my calculus exam") | 0.61 - 0.74 | 0.46 - 0.52 | 0.56 |
| Full requests vs past questions | 0.60 - 0.77 | 0.43 - 0.51 | 0.57 |

The top-ranked match was correct in every related case.

### Standing rules: deciding when "I'm off" means "I'm leaving my room"

4 rules, 13 utterances that should fire one, 11 that shouldn't:

| Approach | Fired correctly | False fires |
|---|---|---|
| Embedding similarity to the trigger | ranked poorly: "my roommate left the room" scored 0.73, above "leaving now" (0.63) | - |
| llama3.2:3b as a yes/no judge (3 prompt styles) | 1-6 / 13 | 1-8 / 75 pairs, ~1 s per check |
| **Model-written trigger variations + drift filter + guards, threshold 0.62** | **8 / 13** | **1 / 11** |

Remaining errors were mostly confusions between similar rules ("I'm off": leaving or bed).
Hence the two corrections: "that was wrong" (adds a not-this example) and "also remind me
when I say ...". Live check: "Okay I'm heading out" fired (0.71); "my roommate left the room"
and "how do I leave a group chat?" didn't.

### Web search: when to search

14 questions (6 need current info, 8 don't):

| Approach | Correct decisions | Notes |
|---|---|---|
| Offer llama3.2:3b a `web_search` tool | 7 / 14 | Searched for "tell me a joke" and "capital of France"; +2 s per turn |
| Ask it to reply `SEARCH: <query>` when unsure | 9 / 14 | Never over-searched; often wrote "I'm not familiar... SEARCH:" after a sentence |
| **Code gate for obvious cases + hold back the first sentence and search if it admits not knowing** | **14 / 14** | Used |

Live: "Who won the 2025 Nobel Prize in Physics?" → read NobelPrize.org, answered correctly with
the source named. Known weakness: the 3B model sometimes misreads pages (asked for the newest
Python, it answered with the in-development branch). The answer step now uses only pages
actually read (it had trusted a third-party snippet over python.org).
