# ORION

**Offline Reasoning and Intelligent Operations Node**: a private, local-first voice
assistant. Speech recognition, language model, and speech synthesis all run on the
device. The same code runs on an x86_64 Fedora laptop (CPU only) and an NVIDIA
Jetson Orin Nano (aarch64, CUDA), with hardware choices in `config.yaml` profiles.

> Status: **Phase 2 built:** permanent memory, conversation memory, standing rules, Claude mode
> with a privacy guard and encrypted audit log, and agentic web search, all on an encrypted
> SQLCipher database. On top of **Phase 1** (voice loop, people and voice recognition, barge-in).
> **Phase 6 in progress:** custom "hey orion" wake word (training notebook ready).
> Wake word → listen → who's speaking? → transcribe → reply → speak.

```
                                         ┌─► TitaNet ──────────► who is it? (name, per-person history)
 mic ─► openWakeWord ─► Silero VAD ──────┤
        "hey jarvis"    end of speech    └─► faster-whisper ───► Ollama (streaming) ─► clause chunker ─► Kokoro / Piper ─► speaker
                                             small.en int8       llama3.2:3b                             British male voice
```

**Using Orion:** see the [user manual](docs/user-manual.md) for every command,
setting, and what to expect. This README is the developer overview.

**Honest about limits:** requests that need data Orion doesn't have yet (weather,
email, calendar, news, prices, scores, alarms) get a fixed "I can't do that yet,
and I won't guess" before the model sees them (`brain/capabilities.py`). A live test
caught the 3B model inventing a forecast despite its instructions.

## Design principles

- **Private by default.** Nothing leaves the machine. Ollama must be on loopback,
  and Orion refuses to start if Ollama is reachable from the network.
- **Outbound only.** No listening ports, tunnels, or webhooks (dorm rule).
  After the one-time model download, Phase 1 makes no internet requests.
- **Personal data only encrypted, and as little as possible.** Audio is kept in
  memory and discarded after transcription; transcripts are never written. The only
  personal data on disk is names and voiceprints, AES-256-GCM encrypted with a key
  in the system keyring. Metrics record timings and memory only, never what was said.
- **Swappable components.** STT, LLM, and TTS sit behind small interfaces
  (`audio/stt.py`, `brain/llm.py`, `audio/tts.py`), so a GPU backend or a cloud
  model is a new class, not a rewrite.
- **Latency first.** The reply is spoken while the model is still generating it.

## Layout

```
main.py              entry point: voice mode, text mode, device listing
config.yaml          common settings + per-device profiles (laptop, jetson, ...)
core/                config & profile detection, hardware probe, metrics, network
                     guard, turn orchestration, voice loop, identity (who's speaking),
                     people (onboarding + people commands), barge-in, feedback
                     ("that was wrong"), dialog helpers
audio/               mic, wake word, VAD/endpointing, STT, speaker recognition,
                     TTS (Kokoro, Piper) + pipelined playback + chimes
brain/               LLM backends + history, persona prompt, speech text cleanup,
                     language modes (profanity), capability guard (honest "can't yet")
storage/             encrypted JSON files (AES-GCM), user store, training notes
security/            data key in the system keyring
tools/               placeholder for Phase 2+
scripts/             download_models.py (hash-pinned), bench.py (latency), voice_eval.py
                     (speaker recognition + endpointing on LibriSpeech), wakeword_test.py
notebooks/           train_hey_orion.ipynb: Colab training for the custom wake word
tests/               pytest suite (no models or audio hardware needed)
docs/                user-manual.md, wake-word.md, benchmarks.md (measured results)
```

## Setup: Fedora laptop

```bash
# System packages. portaudio, gcc, alsa-utils and pipewire-utils ship with
# Fedora Workstation/KDE; listed for completeness.
sudo dnf install portaudio alsa-utils pipewire-utils git uv espeak-ng

# Python 3.12 venv. Fedora 44's default Python (3.14) is newer than some audio/ML
# wheels; uv fetches a standalone 3.12 into your home directory, no root needed.
cd ~/orion
uv venv --python 3.12 --seed venv
source venv/bin/activate
pip install --no-deps -r requirements.txt

# Models: wake word, VAD, speaker recognition, Kokoro and Piper voices (about
# 575 MB, SHA-256 verified). Whisper small.en (about 465 MB) downloads on first run.
python scripts/download_models.py

# Ollama must already be installed with the model pulled:
ollama pull llama3.2:3b
```

If you'd rather use the system packages: `sudo dnf install python3.12 python3.12-devel`
and `python3.12 -m venv venv` works the same.

Optional extra: `espeak-ng` is only used for testing the wake word with synthetic
speech (see Troubleshooting).

## Usage

```bash
python main.py                  # voice mode: say "hey jarvis", then your request
python main.py --text           # type; replies are still spoken
python main.py --text --no-tts  # pure text, for debugging the brain
python main.py --list-devices   # find your mic / speaker names
python main.py --profile jetson # force a profile (or ORION_PROFILE=jetson)
python main.py --demo           # demo mode for IT demos / interviews (clean language)
python main.py -v               # debug logging
```

Text-mode commands: `/reset` clears the conversation, `/stats` shows the last
turn's timings, `/quit` exits. Ctrl+C exits either mode.

**Barge-in:** say "hey jarvis" while Orion talks and he stops within ~50 ms
(`conversation.barge_in: wake_word`, default). An experimental `voice` mode lets
you just talk over him: his own echo is told apart from people by speaker
embedding (his TTS voiceprint plus room echo sampled at startup), two
consecutive checks, and a transcript-overlap check. Through laptop speakers it
still triggers on room noise, so it needs echo cancellation (headset,
speakerphone, or PipeWire echo-cancel).

**Conversation mode:** after Orion answers, he keeps listening for 10 seconds
(`conversation.follow_up_s`), so follow-ups don't need the wake word. A soft
single chime means he's still listening. Stay silent, or say "thanks, that's all",
to end the conversation. Set `follow_up_s: 0` to require the wake word every time.

Chimes: rising = recording you after the wake word, soft single note = listening
for a follow-up, falling = recording stopped.

## People

Orion learns who you are instead of having a name configured:

- **First start:** he introduces himself, asks your name, and confirms it. In
  voice mode he then asks whether he may learn your voice and asks 2 to 5 casual
  questions (about 10 s of speech). The first person is the **owner**.
- **Every utterance** is matched against known voiceprints (NVIDIA TitaNet-small,
  about 30 ms on CPU). Orion uses the speaker's name and keeps a separate chat
  history per person. Unknown voices get answers without a name, and he won't
  list who he knows to a stranger.
- **Voice commands:** "who am I?", "who do you know?", "call me ...",
  "learn my voice" / "relearn my voice", "delete my profile". Owner only:
  "add a new person", "learn *Joe*'s voice", "delete *Joe*". Changes always ask
  to confirm. Names are matched the way Whisper spells them ("delete by" finds
  "Bye"), and the name is always repeated back first.
- **Terminal:** `python main.py users list`, `users rename OLD NEW`,
  `users delete NAME` (type the name to confirm), `users relearn NAME` (voice
  learning from the keyboard, for when Orion no longer recognises someone).
- Text mode always acts as the owner, since it's your keyboard.

**Privacy:** only voice embeddings (lists of numbers) are kept, never audio or
what was said during enrollment. Names and voiceprints are stored in
`data/users.enc`, encrypted with AES-256-GCM, and readable only by your user. The
key lives in the system keyring (KWallet / GNOME Keyring via Secret Service),
never on disk or in git. If the keyring is unavailable, Orion warns you and
remembers people for that session only.

**Two levels of match:** recognising someone uses `people.threshold` (0.40),
measured with `scripts/voice_eval.py` on LibriSpeech: 4% of noisy commands missed,
about 1% of strangers accepted. Anything that changes who Orion trusts (adding,
deleting, relearning or renaming people) needs `people.verify_threshold` (0.55)
on that request itself, which no stranger reached in testing.

**Not security:** voice recognition is for personalisation. A recording of
someone can fool it, so it never unlocks anything sensitive on its own. Later
phases keep passphrases and spoken confirmations for that.

**Tuning:** run with `-v` to see match scores. If Orion mixes people up, raise
`people.threshold` (e.g. 0.50). If he often fails to recognise you, lower it
(e.g. 0.35) or say "relearn my voice" in the room and with the mic you normally
use. Confident matches also refine your voiceprint automatically over time.

## Memory, conversations, rules (Phase 2)

All per person, encrypted in `data/orion.db` (SQLCipher, key derived from the keyring data
key), with daily encrypted backups (last 30). Used by the local model only.

- **Memory:** "remember that..." and lasting facts said in passing; recalled by meaning
  (`nomic-embed-text` via Ollama) and given to the local model when relevant.
- **Conversations:** every exchange saved; recent turns restored after a restart; recall by
  day ("yesterday") or meaning ("what did I say about my exam?"); daily summaries written by
  the local model; "don't save this conversation".
- **Rules:** "whenever X, do Y". Embeddings alone and the 3B model as a yes/no judge both
  failed in testing (see [benchmarks](docs/benchmarks.md)), so the model writes casual
  variations of each trigger at creation, drifted ones are dropped, matching uses the closest
  variation, and guards reject questions, other people and plans. "That was wrong" and "also
  remind me when I say..." correct it.

Thresholds were calibrated on real embeddings, not guessed (`core/memory.py`,
`core/conversations.py`, `core/rules.py`).

## Claude mode and web search (Phase 2)

- **Router** (`core/router.py`): local by default; "ask Claude ...", "switch to Claude mode" /
  "back to local"; optional auto-router off by default. Every decision is logged.
- **Privacy guard:** cloud requests are built from scratch, with no names, memories, conversations
  or rules unless the person says "include ..." for that request. Every request is audited
  exactly as sent (`python main.py audit`), with token cost. No audit log, no cloud.
- **Claude** (`brain/claude.py`): `claude-opus-5-5`, effort `low`, server-side refusal fallbacks,
  key from the keyring. Any failure is spoken and answered locally.
- **Web search** (`core/web_search.py`, `brain/web.py`): local SearXNG container on
  `127.0.0.1` (`scripts/searxng.sh`) or Brave. Code catches obvious "needs current info"
  requests; otherwise the local answer's first sentence is held back and searched instead if it
  admits not knowing. Pages are read with trafilatura, answers name their source, queries are
  privacy-filtered, pages are untrusted data, and private network addresses are refused.
  Big questions offer a handoff to Claude with its own web search.

## Language

Orion is a personal assistant, so he's never offended by swearing and never
lectures you about it. With `assistant.profanity: "on"` (the default) he mirrors
your register: swear at him and he swears back casually, ask a clean question and
you get a clean answer. He never uses slurs and never goes harsher than you did.

Clean mode applies when any of these is true:

| Reason | How it starts and ends |
|---|---|
| You asked | "Orion, keep it clean" until "back to normal" |
| Demo | `--demo` or `ORION_DEMO=1`, for IT demos and interviews. "Back to normal" doesn't override it. |
| Meeting notes | Automatically while taking notes (Phase 5) |
| Settings | `assistant.profanity: "off"` |
| Stranger | Voices Orion doesn't recognise always get clean language |

In clean mode a filter also scrubs anything that slips through from the model
("holy shit, well done" → "wow, well done"). Swearing is only ever spoken: emails,
push notifications, commit messages, and anything written to files must go through
`brain.profanity.scrub()` (Phases 3 to 8).

Measured on llama3.2:3b: with the examples given as earlier chat turns, Orion
swears back 14 times out of 15 when you swear, and swears unprompted on a clean
question 1 time in 9. Phase 10 fine-tuning should make his voice consistent.

## Training notes (opt-in)

Set `training_notes.enabled: true` to let Orion learn from mistakes. Say "that
was wrong" after a bad answer and he logs the failure type plus an optional
few-word note you give him ("too long", "should have asked first"). Misroutes and
wrong tools will be logged automatically once Phase 2 adds the router and tools.

- **No conversation content, ever.** Your question and his reply are not stored.
  The only context kept is an allowlist of facts about the turn (mode, model,
  language mode, reply length, latency), and the code rejects anything else.
- Encrypted in `data/training_notes.enc` with the keyring key. The nuclear wipe
  (Phase 2.5) deletes it.
- Only people Orion recognises can add notes. View or clear them with
  `python main.py notes list` and `python main.py notes clear`.
- Phase 10 uses these notes only to decide which **synthetic** examples to
  write. The notes themselves are never training data.

## Voices

`tts.backend` in `config.yaml`:

| Backend | Sound | Laptop CPU cost | Notes |
|---|---|---|---|
| `kokoro` (default) | Natural | About 2.5 s more before the first word | British males: `bm_george`, `bm_lewis`, `bm_daniel`, `bm_fable`. Should be fast on the Jetson GPU. |
| `piper` | Robotic but clear | Fastest | `en_GB-alan-medium`, `en_GB-northern_english_male-medium`, `en_GB-semaine-medium` (speaker_id 1 spike, 2 obadiah) |

Samples of every voice saying the same line are generated into `voice-samples/`
(git-ignored).

The wake word is temporarily **"hey jarvis"** (openWakeWord's pretrained model).
To train and switch to **"hey orion"**, see [docs/wake-word.md](docs/wake-word.md):
a Colab notebook trains it, `scripts/wakeword_test.py` checks it on your voice, and
`wakeword.models` can run both during the switch. The wake phrase is stripped from
what you say, so the model never sees "Hey Jarvis, ..." and corrects you.

## Configuration and profiles

`config.yaml` has a `common` block and a block per device profile. A profile
contains only what differs. With `active_profile: auto`, the first profile whose
`detect` rule matches the machine wins (the Jetson matches on
`/proc/device-tree/model`), otherwise `default_profile` is used.

To add a device, such as a 16 GB Orin NX, copy the `jetson` block, give it a more
specific rule (`device_tree_model_contains: "Orin NX"`), and place it **above**
`jetson`. An example is commented in the file.

`device: auto` / `compute_type: auto` choose CUDA when it's actually usable and
fall back to CPU otherwise, so a profile asking for CUDA still runs on a CPU build.

Audio devices: leave `input_device` / `output_device` as `null` for the system
default, or set a name substring from `--list-devices` (e.g. `"Jabra"` for a USB
speakerphone).

## Performance

Measured numbers live in [docs/benchmarks.md](docs/benchmarks.md). Reproduce them with:

```bash
python scripts/bench.py --runs 2 --write
```

Laptop (i5-10th gen, CPU only), Phase 1 with the Piper voice: **about 3.2 s from
end of speech to first audio** (STT 1.4 s, LLM first token 0.46 s), with 10/10
transcripts correct. With the Kokoro voice, add about 2.5 s on this CPU.
Memory: Orion about 1.0 GB, Ollama about 2.6 GB. Voice mode adds the 5 s silence
window that decides you've finished talking (`vad.end_silence_ms`; it was 0.8 s when
the latency above was measured).

Optimizations made, each one measured:

| Change | Effect |
|---|---|
| Stream LLM output; speak per sentence on a background thread | Speech starts before the reply is finished |
| Split long first sentences at clauses (small models write long sentences) | 4.35 s → 3.18 s first audio |
| `small.en` instead of `small` | Faster, and transcripts went from 9/10 to 10/10 correct |
| Greedy decoding (`beam_size: 1`), 4 threads (8 hyperthreads was 25% slower) | STT about 1.4 s |
| Model warm-up at start, Ollama `keep_alive: 30m` | No cold-start penalty on the first request |

On CPU, Whisper's fixed 30 s encoder window puts a floor of about 1.4 s on STT for
`small`. The Jetson's GPU is the real fix (see below). Speaker recognition adds
about 30 ms per utterance.

## Tests

```bash
pytest -q
```

No models, microphone, or keyring are needed: everything is faked. The tests cover:

- profile detection and merging; the loopback network guard
- markdown stripping and sentence/clause chunking; endpointing (silence timeout,
  trailing silence, max length, pre-roll)
- the turn pipeline (error speech, separate history per person, no names for
  strangers, metrics never containing conversation text) and the Ollama client
  against a mocked HTTP transport
- the pipelined speaker (ordering, first-audio timing, failure doesn't hang)
- the voice loop's follow-up window and dismissals
- encrypted storage (no plaintext on disk, wrong key and tampering detected,
  file permissions), owner hand-over, voiceprint adaptation limits
- speaker matching (threshold, refusing near-ties, continuity in follow-ups)
- onboarding and people commands: name extraction, yes/no, consent, too little
  speech, owner-only adding, look-alike voices, rename/delete confirmations,
  misheard names, the strong-match safeguard for trust-changing commands
- barge-in: wake-word and voice triggers, echo rejection, stopping speech and
  generation mid-reply, echo transcripts
- the capability guard: live/personal-data requests refused honestly, ordinary
  questions still answered
- language modes and the scrub filter; training notes (opt-in, no content, whitelist)
- wake words: phrase stripping (incl. Whisper's mishearings), several models with
  per-model thresholds

## Updating dependencies

Everything is pinned in `requirements.txt` (a full lock, installed with `--no-deps`
because openWakeWord hard-requires `tflite-runtime`, which has no Python 3.12 wheels
and which we don't use; we run its ONNX models). Direct dependencies are listed in
`requirements.in`. To update:

```bash
source venv/bin/activate
uv pip install -U <package>          # or edit requirements.in and reinstall
pytest -q && python scripts/bench.py # check nothing regressed
{ head -3 requirements.txt; uv pip freeze; } > requirements.txt.new && mv requirements.txt.new requirements.txt
```

Models are pinned by SHA-256 in `scripts/download_models.py`. To change one,
update its URL, blank its hash, run the script, then pin the printed hash.

## Security notes

- **Full-disk encryption:** use LUKS on both the laptop and the Jetson's NVMe.
  Orion encrypts its own data (users and voiceprints now; everything else in
  Phase 2.5). LUKS protects everything else if a device is lost.
- **Keyring:** the data key is in KWallet / GNOME Keyring. Deleting the `orion`
  entry there makes `data/users.enc` permanently unreadable (that's how the
  Phase 2.5 wipe will work). On the Jetson, a systemd service has no desktop
  keyring session; Phase 9 will set up a headless key store.
- Check that Ollama isn't exposed: `ss -ltnp | grep 11434` should show `127.0.0.1`.
  Never set `OLLAMA_HOST=0.0.0.0`.

## Jetson Orin Nano (preview; validated on device in a later phase)

- JetPack 6.x (Ubuntu 22.04, aarch64), booting from NVMe.
- MAXN Super power mode: `sudo nvpmodel -m 2` (check `nvpmodel -q`; mode numbers
  vary by JetPack release), then `sudo jetson_clocks`.
- Ollama: the official installer detects the Jetson and uses the GPU. Then
  `ollama pull qwen3:4b`.
- Python: the same `uv venv --python 3.12` approach. JetPack's Python 3.10 is
  too old for the pinned numpy.
- **GPU Whisper:** PyPI's aarch64 CTranslate2 wheels are CPU-only. Use a
  CUDA build of CTranslate2 (build from source, or NVIDIA's `jetson-containers`
  faster-whisper image). Until then, `device: auto` falls back to CPU.
- Memory: the 8 GB is shared between CPU and GPU. Add a swap file on the NVMe as
  a safety net:
  ```bash
  sudo fallocate -l 8G /ssd/swapfile && sudo chmod 600 /ssd/swapfile
  sudo mkswap /ssd/swapfile && sudo swapon /ssd/swapfile
  echo '/ssd/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  ```
  Disabling zram (`sudo systemctl disable nvzramconfig`) leaves more real RAM.
  Run headless (`sudo systemctl set-default multi-user.target`) to free about 800 MB.

## Troubleshooting

- **Wake word never triggers:** run `python main.py -v` and watch the logs, try
  `wakeword.threshold: 0.35`, and check the right mic is selected
  (`--list-devices`, `arecord -l`). Quick self-test with synthetic speech:
  `espeak-ng "hey jarvis"` played near the mic. (Piper's British voice is too far
  from the model's training voices to trigger it; espeak scores 0.99.)
- **Cuts you off mid-sentence:** raise `vad.end_silence_ms` (e.g. 1000).
- **"I can't reach Ollama":** `systemctl status ollama` or `ollama serve`.
- **Doesn't recognise you / mixes people up:** see People → Tuning. Say
  "relearn my voice" with your usual mic and room.
- **"I can't reach the system keyring":** make sure KWallet is enabled (System
  Settings → KDE Wallet) and unlocked. Without it, people are forgotten at restart.
- **Orion hears itself:** the mic buffer is discarded while Orion speaks, but
  very loud speakers can still leak. Lower the volume or use a speakerphone with
  echo cancellation.

## Roadmap

- [x] Phase 1: Voice loop, conversation follow-ups, natural voice
- [x] Learn people's names; recognise who's speaking by voice (multi-user)
- [x] Phase 2: Router, Claude mode with privacy guard, web search, permanent memory,
  conversation memory, standing rules ([spec](docs/specs/phase-2.md)) *(built; awaiting
  live voice testing)*
- [ ] Phase 2.5: Shutdown and wipe kill switches *(encryption at rest done: keyring key,
  AES-GCM files, SQLCipher database, encrypted backups)*
- [ ] Phase 3: Outlook mail and calendar (Microsoft Graph, device code flow, read-only)
- [ ] Phase 4: Alarms and morning briefing
- [ ] Phase 5: Local meeting notes
- [ ] Phase 6: Custom "hey orion" wake word *(Colab training notebook, voice test script,
  multi-model wake words ready: see [docs/wake-word.md](docs/wake-word.md); waiting on a trained model)*
- [ ] Phase 7: "Build me something" via sandboxed Claude Code
- [ ] Phase 8: Orion's own allowlisted inbox
- [ ] Phase 9: System control via fixed, validated tools under systemd
- [ ] Phase 10: Fine-tuning for behaviour (last, after everything else works):
  router classifier (measured accuracy and false-"general" privacy-leak rate),
  tool calling, then Orion's voice. Synthetic or hand-written data only, in
  `~/orion-training/` (survives the wipe, holds no personal data), LoRA via Unsloth
  on Colab, exported to GGUF for Ollama, with base vs fine-tuned evaluation reports.
  *(started: opt-in training notes)*
