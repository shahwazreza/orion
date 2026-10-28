# Orion user manual

**Version:** Phase 1 (voice loop, people, language modes, training notes), October 2026.

Orion is a private voice assistant that runs entirely on your own computer.
You talk to him; he listens, works out who you are, thinks with a local language
model, and answers out loud in a British voice. Nothing you say leaves the machine.

This manual covers everything Orion can do as of Phase 1. For developer
details (architecture, tests, benchmarks), see the [README](../README.md).

---

## Contents

1. [What Orion can and can't do yet](#1-what-orion-can-and-cant-do-yet)
2. [Installing](#2-installing)
3. [Starting Orion](#3-starting-orion)
4. [Meeting Orion for the first time](#4-meeting-orion-for-the-first-time)
5. [Talking to Orion](#5-talking-to-orion)
6. [Voice command reference](#6-voice-command-reference)
7. [Text mode](#7-text-mode)
8. [More than one person](#8-more-than-one-person)
9. [Language: casual and clean modes](#9-language-casual-and-clean-modes)
10. [Demo mode](#10-demo-mode)
11. [Choosing Orion's voice](#11-choosing-orions-voice)
12. [Training notes (opt-in)](#12-training-notes-opt-in)
13. [Terminal command reference](#13-terminal-command-reference)
14. [Settings reference](#14-settings-reference)
15. [Your privacy: what's stored and where](#15-your-privacy-whats-stored-and-where)
16. [What to expect: speed and memory](#16-what-to-expect-speed-and-memory)
17. [Troubleshooting](#17-troubleshooting)
18. [Known limitations](#18-known-limitations)

---

## 1. What Orion can and can't do yet

**He can:**

- Wake up when you say **"hey jarvis"**, listen until you stop talking, and answer
  out loud.
- Keep a conversation going: after he answers, you can just keep talking for a
  few seconds without saying the wake word again.
- Learn your name, learn your voice (with your permission), and recognise who's
  talking, so he uses the right name and keeps each person's chat separate.
- Answer general questions, chat, explain things, and help you think, using
  the local model (llama3.2:3b on the laptop).
- Swear back casually when you swear, or keep it clean when asked.
- Note his mistakes when you tell him "that was wrong" (opt-in).
- Run in text mode, where you type instead of talk.

**He can't yet** (these come in later phases):

| Not yet | Coming in |
|---|---|
| Use the internet, Claude, or look things up | Phase 2 |
| Remember facts about you between sessions ("remember that...") | Phase 2 |
| Standing rules ("whenever I say I'm leaving, remind me...") | Phase 2 |
| Shut down by voice, nuclear wipe | Phase 2.5 |
| Read your email or calendar | Phase 3 |
| Alarms, morning briefing | Phase 4 |
| Meeting notes | Phase 5 |
| Respond to "hey Orion" | Phase 6 |
| Build software for you | Phase 7 |

Ask him about any of these and he should tell you he can't do it yet. He's told
never to invent emails, events, or facts.

---

## 2. Installing

You need a Fedora laptop (or the Jetson later), a microphone, and speakers.
Ollama must be installed with the model pulled.

```bash
# 1. System packages
sudo dnf install portaudio alsa-utils pipewire-utils git uv espeak-ng

# 2. Python environment (Python 3.12)
cd ~/orion
uv venv --python 3.12 --seed venv
source venv/bin/activate
pip install --no-deps -r requirements.txt

# 3. Models (~575 MB; every file is checksum-verified)
python scripts/download_models.py

# 4. The language model
ollama pull llama3.2:3b
```

The speech-recognition model (Whisper small.en, ~465 MB) downloads the first
time you start voice mode. After that, Orion works with no internet at all.

**KDE Wallet must be enabled** (System Settings → KDE Wallet). Orion keeps his
encryption key there. Without it he still works, but forgets everyone when he
restarts.

**Recommended:** turn on full-disk encryption (LUKS) for the whole machine.

---

## 3. Starting Orion

Always activate the environment first:

```bash
cd ~/orion
source venv/bin/activate
```

| You want to | Run |
|---|---|
| Talk to Orion | `python main.py` |
| Type to Orion (he still speaks) | `python main.py --text` |
| Type, with no speech at all | `python main.py --text --no-tts` |
| Show him at a demo or interview | `python main.py --demo` |
| See what's going on under the hood | add `-v`, e.g. `python main.py -v` |

Startup takes about 10 seconds while models load. You'll see lines like
`Whisper STT ready in 2.2s`, then Orion speaks.

**To stop Orion, press Ctrl+C** in the terminal. (A spoken "Orion, shut down"
arrives in Phase 2.5. Until then, saying it just starts a conversation.)

---

## 4. Meeting Orion for the first time

The first time you start him, Orion doesn't know anyone. He introduces himself
and walks you through a short setup. Nothing needs configuring beforehand.

**In voice mode:**

1. *"Hello, I'm Orion. I don't think we've met. What's your name?"*
   Say your name any natural way: "Reza", "I'm Reza", "my name is Reza".
2. *"Nice to meet you, Reza. Did I get your name right?"*
   Say **yes**, or **no** and he'll ask again.
3. *"Can I learn your voice, so I know it's you when you talk to me?"*
   Say **yes** to continue, or **no** to skip. He won't ask again unless you
   say "learn my voice" later.
4. He asks a few casual questions (your day, a favourite meal, a place you'd
   like to visit). Answer each in a sentence or two, in your normal voice.
   He needs about 10 seconds of speech in total, usually 2 or 3 answers. If an
   answer is very short ("Pizza."), he asks you to say a bit more.
5. *"Got it, Reza. I'll recognise your voice from now on."*

He keeps a mathematical voice signature, not recordings, and your answers to
those questions aren't saved.

**In text mode,** he asks only your name, and learns your voice the next time
you start voice mode.

The first person Orion meets becomes the **owner**. That matters for adding
other people (section 8) and, in later phases, for whose email and calendar he
can read.

---

## 5. Talking to Orion

### A conversation, step by step

1. Say **"hey jarvis"**. *(It's a placeholder wake word; "hey Orion" comes in Phase 6.)*
2. You hear a **rising chime**: he's recording.
3. Speak normally. When you pause for about a second, a **falling chime** means
   he's stopped recording.
4. He answers.
5. You hear a **soft single note**: he's still listening. Ask a follow-up
   without the wake word, within 6 seconds.
6. To end, stay silent or say something like **"thanks, that's all"**. You hear
   the falling chime and he goes back to waiting for "hey jarvis".

### The chimes

| Sound | Meaning |
|---|---|
| Rising (two notes up) | Recording you after the wake word |
| Soft single note | Listening for a follow-up |
| Falling (two notes down) | Recording stopped |

The microphone is always on, listening for the wake word, but nothing is
recorded or transcribed until the rising chime.

### Interrupting Orion

`conversation.barge_in` in `config.yaml` controls this:

| Setting | What interrupts him |
|---|---|
| `wake_word` (default) | "Hey jarvis" |
| `voice` | Anyone talking over him, or "hey jarvis". What you say becomes your next request. Needs echo cancellation (below) |
| `off` | Nothing; he always finishes |

In `voice` mode his microphone also hears his own voice from the speakers. He
compares what he hears with his own voice (his TTS voice plus his room echo,
sampled while he says "Orion online." at startup), requires two checks in a
row, and drops interruptions whose words match what he just said. Through
laptop speakers that still isn't enough: in testing, room sounds and dips in
his echo made him stop at random. Use `voice` only with a headset, a
speakerphone with built-in echo cancellation, or PipeWire's echo-cancel module.

### Ending a conversation

Any of these end it, with or without "thanks" or "Orion" around them:
"that's all", "that's it", "never mind", "nothing", "nothing else", "stop",
"cancel", "goodbye", "bye", "I'm done", "I'm good", "all done", "that'll do",
or just "thanks" / "thank you" on its own.

"Thanks, what's the weather tomorrow?" is **not** an ending; he answers it.

### Tips

- Wait for the rising chime before speaking. Words said during the chime may be cut.
- He stops listening after about a second of silence. If he cuts you off
  mid-sentence, see section 17.
- He waits up to 6 seconds for you to start talking after the wake word, and
  records at most 20 seconds in one go.
- **You can interrupt him:** say **"hey jarvis"** while he's talking. He stops
  within a moment and listens. See [Interrupting Orion](#interrupting-orion).
- Answers are short on purpose (one to three sentences). Ask for more detail
  if you want it.

---

## 6. Voice command reference

Most of what you say goes to Orion's language model and gets a normal answer.
These phrases are handled directly instead, and behave the same every time.

### People

| Say | What happens |
|---|---|
| "Who am I?" / "What's my name?" | He tells you who he thinks you are |
| "Who do you know?" | Lists the people he knows (only to people he recognises) |
| "Call me *Rez*" / "My name is *Rez*" | Renames you, after asking to confirm |
| "Add a new person" / "Add another person" | Meets someone new (owner only; section 8) |
| "Learn *Joe*'s voice" | Teaches Orion someone else's voice (owner only) |
| "Learn my voice" / "Relearn my voice" | Records a fresh voice signature |
| "Delete my profile" / "Forget me" | Deletes your name and voice, after confirming |
| "Delete *Joe*" | Deletes someone else, after confirming (owner only) |

### Language

| Say | What happens |
|---|---|
| "Keep it clean" / "No swearing" / "Watch your language" | Clean mode on |
| "Back to normal" / "You can swear again" | Clean mode off (unless something else requires it; section 9) |

### Feedback

| Say | What happens |
|---|---|
| "That was wrong" / "That's not right" / "Wrong answer" | He apologises and, if training notes are on, asks what went wrong (section 12) |

### Ending

See [Ending a conversation](#ending-a-conversation).

---

## 7. Text mode

```bash
python main.py --text
```

Type your messages at the `You:` prompt. Orion replies on screen and out loud
(add `--no-tts` for silent replies). All the voice commands in section 6 work
when typed too.

Text mode always acts as the **owner**, because it's your keyboard. He greets
you by name at startup.

Extra text-only commands:

| Type | What it does |
|---|---|
| `/reset` | Clears the current conversation's history |
| `/stats` | Shows timings and memory for the last reply |
| `/quit` | Exits (Ctrl+D or Ctrl+C also work) |

---

## 8. More than one person

Orion can know several people and tell them apart by voice.

### Adding someone

1. The **owner** says "hey jarvis, add a new person".
2. Orion asks the new person to come close, then asks their name, confirms it,
   and asks their permission to learn their voice, exactly as in section 4.
3. Everyone added after the owner is a **member**.

Only the owner can add people. If Orion doesn't recognise the voice asking,
he says so.

If the new person's voice sounds very close to someone he already knows, he asks
whether they're sure they aren't that person before saving it.

If voice learning didn't finish for someone (answers too short, or they said no
and changed their mind), the **owner** says "learn *Joe*'s voice" with Joe
nearby. Joe can't ask for it himself, because Orion can't recognise him yet.

### How recognition works day to day

- Every time someone speaks, Orion compares their voice with the voices he knows.
  This takes about 30 milliseconds.
- If he recognises you, he uses your name, and you get **your own chat history**.
  Other people's conversations never show up in yours.
- Very short replies ("yes", "and tomorrow?") don't contain enough voice to judge,
  so during a conversation he assumes it's still the same person.
- Each time he's confident it's you, he quietly refines your voice signature,
  so recognition improves over time (a new mic, a cold, a different room).

### People Orion doesn't recognise

- He answers them politely, without using a name.
- He won't tell them who he knows, only how many people.
- He always uses clean language with them.
- Their chat history is forgotten when the conversation ends.

### Changing or removing people

- By voice: "call me ...", "relearn my voice", "delete my profile" change or
  delete yourself. The owner can also say "delete *Joe*" or "learn *Joe*'s voice".
- If you add someone Orion already knows by name but has never heard (for
  example, their voice learning didn't finish), he recognises the name and just
  learns their voice, instead of creating a duplicate.
- From the terminal (any person): see section 13.
- If the owner is deleted, the next person Orion met becomes owner.

### Changes that need a clear match

Orion recognises you at a fairly relaxed level, so he still knows you in a noisy
room. That relaxed level lets about 1 stranger in 100 pass as someone he knows,
so **anything that changes who he trusts needs a clear match on that request
itself**:

- adding a person, deleting someone, learning someone else's voice (owner)
- relearning your own voice, renaming yourself, deleting your profile

If the match isn't clear, he says *"For that I need to be sure it's you…"*.
Say "hey jarvis" and ask again in a full sentence, close to the mic. Being
recognised earlier in the same conversation isn't enough on its own. Text mode
is your keyboard, so it skips this check. The level is `people.verify_threshold`.

> **Voice recognition is a convenience, not security.** A recording of someone's
> voice could fool it. Orion never unlocks anything sensitive on a voice match
> alone; later phases use passphrases and spoken confirmations for that.

---

## 9. Language: casual and clean modes

Orion is your personal assistant, so he's never offended by swearing and never
lectures you about it.

**Casual mode (the default):** he mirrors you. Swear at him and he swears back,
casually and in kind; ask something cleanly and he answers cleanly. He never uses
slurs, never insults groups of people, and never goes harsher than you did.

**Clean mode:** no swearing from him at all. He still won't comment on yours.

Clean mode is on whenever **any** of these is true:

| Reason | How it starts | How it ends |
|---|---|---|
| You asked | "Keep it clean" | "Back to normal" |
| Demo mode | `--demo` (section 10) | Restart without `--demo` |
| Meeting notes | Automatic (Phase 5) | When notes stop |
| Settings | `profanity: "off"` in config.yaml | Change the setting |
| Someone he doesn't recognise is talking | Automatic | When he recognises the speaker |

If you say "back to normal" while demo mode or the setting is keeping him clean,
he tells you which one, and stays clean.

As a safety net, in clean mode anything the model lets slip is filtered before
it's spoken ("holy shit, well done" becomes "wow, well done").

"Keep it clean" lasts until you say "back to normal" or restart Orion.

Swearing is only ever **spoken**. In later phases, emails he sends, notifications,
and anything written to files are always clean.

---

## 10. Demo mode

For IT demos, interviews, or showing Orion to anyone:

```bash
python main.py --demo
```

or set `ORION_DEMO=1`. In demo mode:

- Language is always clean, and "back to normal" can't change that.
- He waits 8 seconds for follow-ups instead of 6, which helps while you're
  explaining things to people.
- Any other demo-only settings can go in the `demo:` section of `config.yaml`.

---

## 11. Choosing Orion's voice

Two voice engines are installed. Pick one in `config.yaml` under `tts:`.

| Engine | Sounds | Speed on the laptop |
|---|---|---|
| **Kokoro** (default) | Natural | About 2.5 s extra before he starts speaking |
| **Piper** | More robotic, very clear | Fastest |

**Kokoro voices** (`tts.kokoro.voice`): `bm_george` (default), `bm_lewis`,
`bm_daniel`, `bm_fable`. All British male.

**Piper voices** (`tts.piper.voice`): `en_GB-alan-medium`,
`en_GB-northern_english_male-medium`, or `en_GB-semaine-medium` with
`speaker_id: 1` (Spike) or `2` (Obadiah).

To compare, listen to the samples in `voice-samples/`. Each voice says the same
line there.

Example, switching to fast Piper with the northern English voice:

```yaml
  tts:
    backend: piper
    piper:
      voice: en_GB-northern_english_male-medium
```

`tts.kokoro.speed` (e.g. `1.1`) or `tts.piper.length_scale` (e.g. `0.9`) make
him talk faster.

---

## 12. Training notes (opt-in)

Training notes help improve Orion later (Phase 10) without collecting anything
private. They're **off by default**.

**Turn on:** in `config.yaml`, set

```yaml
  training_notes:
    enabled: true
```

**Use:** after a bad answer, say "that was wrong". Orion asks *"In a few words,
what went wrong? Or say skip."* Answer briefly ("too long", "it made that up",
"should have asked first") or say "skip".

**What's saved:** the type of mistake, your few words if you gave any, and
facts about that reply: voice or text mode, which model, casual or clean,
reply length, how long it took. **Your question and his answer are never saved.**

**Who:** only people Orion recognises can add notes.

**See or delete them:** `python main.py notes list` and `python main.py notes clear`.

Notes are encrypted on disk. In Phase 10 they guide which made-up practice examples
get written for training; the notes themselves are never used as training data.

---

## 13. Terminal command reference

Run these from `~/orion` with the environment active.

### Running Orion

| Command | Purpose |
|---|---|
| `python main.py` | Voice mode |
| `python main.py --text` | Text mode |
| `python main.py --text --no-tts` | Text mode, silent |
| `python main.py --demo` | Demo mode (clean language) |
| `python main.py --profile laptop` | Force a hardware profile (also `ORION_PROFILE=...`) |
| `python main.py --config other.yaml` | Use a different settings file |
| `python main.py -v` | Debug output: wake word and voice-match scores, timings |
| `python main.py --list-devices` | List microphones and speakers |
| `python main.py --help` | All options |

### People

| Command | Purpose |
|---|---|
| `python main.py users list` | Everyone Orion knows, their role, and voice status |
| `python main.py users rename OLD NEW` | Rename someone |
| `python main.py users delete NAME` | Delete someone (type their name to confirm) |
| `python main.py users relearn NAME` | Learn someone's voice again, started from the keyboard. Use this if Orion stops recognising you (stop Orion first) |

### Training notes

| Command | Purpose |
|---|---|
| `python main.py notes list` | Show all notes |
| `python main.py notes clear` | Delete all notes (type `yes` to confirm) |

### Maintenance

| Command | Purpose |
|---|---|
| `python scripts/download_models.py` | Download or re-verify all models |
| `python scripts/bench.py --runs 2` | Measure speed and memory (add `--write` to save to `docs/benchmarks.md`) |
| `pytest -q` | Run the test suite |

---

## 14. Settings reference

All settings live in `config.yaml`. Settings under `common:` apply everywhere.
The `laptop` and `jetson` profiles override only what differs for that device,
and the right one is picked automatically. Restart Orion after changing anything.

### Assistant

| Setting | Default | Meaning |
|---|---|---|
| `assistant.name` | `Orion` | What he calls himself |
| `assistant.profanity` | `"on"` | `"on"` casual mode allowed, `"off"` always clean. Keep the quotes. |

### Audio

| Setting | Default | Meaning |
|---|---|---|
| `audio.input_device` | `null` | Microphone: `null` = system default, or part of its name from `--list-devices` |
| `audio.output_device` | `null` | Speakers, same rules |
| `audio.chimes` | `true` | Play the listening chimes |

### Wake word

| Setting | Default | Meaning |
|---|---|---|
| `wakeword.model` | `hey_jarvis` | Built-in model name, or path to a custom `.onnx` |
| `wakeword.threshold` | `0.5` | Lower (e.g. `0.35`) = triggers more easily, more false alarms |
| `wakeword.cooldown_s` | `1.5` | Ignore the wake word briefly after it fires |

### Listening

| Setting | Default | Meaning |
|---|---|---|
| `conversation.follow_up_s` | `6.0` | Seconds he waits for a follow-up; `0` = wake word every time |
| `vad.end_silence_ms` | `1000` | Silence that means you've finished. Raise if he cuts you off |
| `vad.neg_threshold` | `0.35` | Once you're talking, quieter sounds still count as speech, so soft word endings don't cut you off |
| `vad.start_timeout_s` | `6.0` | How long he waits for you to start after the wake word |
| `vad.max_utterance_s` | `20.0` | Longest single recording |
| `vad.threshold` | `0.5` | How sure the speech detector must be that you're talking |

### People

| Setting | Default | Meaning |
|---|---|---|
| `people.enabled` | `true` | Voice recognition on or off |
| `people.threshold` | `0.40` | How similar a voice must be to count as you. Raise if he mixes people up; lower if he doesn't recognise you |
| `people.verify_threshold` | `0.55` | Stricter match needed for adding, deleting, relearning or renaming people |
| `people.margin` | `0.05` | How clearly the best match must beat the second-best |
| `people.min_audio_s` | `0.8` | Shorter clips aren't used to judge who's speaking |
| `people.adapt_threshold` | `0.72` | Matches this confident refine your voice signature |
| `people.enroll_target_s` | `10.0` | Seconds of speech collected when learning a voice |

### Speech recognition

| Setting | Default | Meaning |
|---|---|---|
| `stt.model` | `small.en` | Whisper model. `small.en` is the best speed/accuracy balance on the laptop |
| `stt.device` | `auto` | `auto`, `cpu`, or `cuda` |
| `stt.language` | `en` | English |

### Language model

| Setting | Default | Meaning |
|---|---|---|
| `llm.model` | `llama3.2:3b` (laptop), `qwen3:4b` (Jetson) | Ollama model |
| `llm.temperature` | `0.6` | Higher = more varied answers |
| `llm.max_history_turns` | `8` | Exchanges remembered per person, in memory only |
| `llm.keep_alive` | `30m` | How long Ollama keeps the model loaded between requests |
| `llm.host` | `http://127.0.0.1:11434` | Must be this machine; Orion refuses anything else |

### Voice

See section 11: `tts.backend`, `tts.kokoro.*`, `tts.piper.*`.

### Data and logs

| Setting | Default | Meaning |
|---|---|---|
| `paths.data` | `data` | Encrypted personal data |
| `paths.models` | `models` | Downloaded models |
| `training_notes.enabled` | `false` | Section 12 |
| `metrics.enabled` | `true` | Log timings and memory per reply (no conversation text) |

---

## 15. Your privacy: what's stored and where

**Nothing leaves your computer.** After the one-time model downloads, Orion makes
no internet connections at all. He also opens no network ports, and refuses to
start if Ollama is reachable from the network.

| What | Stored? | Where |
|---|---|---|
| Your voice audio | **Never.** Held in memory, discarded after each reply | none |
| What you said, what he said | **Never** written to disk. Chat history is in memory and gone when Orion stops | none |
| Names, roles, voice signatures | Yes, **encrypted** (AES-256) | `data/users.enc` |
| Training notes (if on) | Yes, **encrypted**, no conversation text | `data/training_notes.enc` |
| The encryption key | In KDE Wallet, never on disk or in git | your system keyring |
| Speed and memory metrics | Yes, numbers only | `logs/metrics.jsonl` |

The `data/` folder and its files can only be read by your user account. None of
`data/`, `logs/`, `models/`, or `voice-samples/` is ever committed to git.

A voice signature is a list of numbers describing how your voice sounds. It can't
be played back as audio.

**To remove a person:** "delete my profile", or `python main.py users delete NAME`.
**To remove everything:** stop Orion and delete the `data/` folder. Removing the
`orion` entry from KDE Wallet also makes any remaining copies permanently
unreadable. (A one-command nuclear wipe arrives in Phase 2.5.)

---

## 16. What to expect: speed and memory

Measured on the laptop (Intel i5-10400H, no GPU):

| | Typical |
|---|---|
| Time from when you stop talking to his first word (Piper voice) | about 3.2 s |
| Same, with the Kokoro voice | about 5.5 to 6 s |
| Speech recognition | about 1.3 to 1.4 s |
| Voice recognition | about 0.03 s |
| Orion's memory use | about 1 to 1.5 GB |
| Ollama with llama3.2:3b loaded | about 2.6 GB |

Speech recognition and the language model are warmed up at startup, so the first
reply isn't noticeably slower than the rest. On the Jetson, the GPU should make all of this much faster;
it gets measured when the board arrives. Full numbers are in
[benchmarks.md](benchmarks.md).

---

## 17. Troubleshooting

**He doesn't react to "hey jarvis".**
Run `python main.py -v` and watch the scores while you say it. Check the right
mic is selected (`--list-devices`), speak a bit closer, or lower
`wakeword.threshold` to `0.35`.

**He cuts me off mid-sentence.**
Raise `vad.end_silence_ms` to `1000` or `1200`.

**He takes ages to stop listening.**
Background noise (fan, music) can sound like speech. Move the mic or lower the
noise, or lower `vad.end_silence_ms` slightly.

**He doesn't recognise me, or mixes people up.**
Run with `-v` to see match scores. Say "relearn my voice" using the mic and room
you normally use. If he doesn't recognise you at all, he can't tell who's asking,
so stop Orion and run `python main.py users relearn YOURNAME` instead. If he still misses you, lower `people.threshold` to `0.35`; if
he confuses two people, raise it to `0.50`.

**"Heads up: I can't reach the system keyring."**
KDE Wallet is off or locked. Enable it in System Settings → KDE Wallet. Until
then he forgets everyone when he restarts.

**"I can't reach Ollama."**
Start it: `systemctl status ollama`, or `ollama serve`.

**"The model ... isn't installed in Ollama."**
`ollama pull llama3.2:3b`.

**"Ollama is listening on 0.0.0.0" and Orion won't start.**
Ollama is exposed to the network. Unset `OLLAMA_HOST` (or set it to `127.0.0.1`)
and restart Ollama.

**He seems to hear himself.**
Lower the speaker volume, or use a speakerphone with echo cancellation.

**His voice is slow to start.**
That's the Kokoro voice on CPU. Switch to Piper (section 11) for speed.

**He made something up.**
The 3B model sometimes does. Say "that was wrong" (with training notes on) so it
gets noted. Later phases add real data sources and fine-tuning.

---

## 18. Known limitations

- The wake word is **"hey jarvis"** until Phase 6.
- **No spoken shutdown yet.** Use Ctrl+C. "Orion, shut down" and the wipe arrive in Phase 2.5.
- **Interrupting by just talking** (`barge_in: voice`) isn't reliable through
  laptop speakers yet, so the default is "hey jarvis" to interrupt.
- **English only.**
- **No memory between restarts** except who people are. "Remember that..." arrives in Phase 2.
- **The small model can be wrong** or invent details. Don't rely on it for facts
  that matter.
- **Casual mode isn't perfect:** in testing he swore back about 14 times out of 15
  and occasionally swore on a clean question. Phase 10 fine-tuning targets this.
- **"Keep it clean" doesn't survive a restart.** Use `profanity: "off"` or `--demo`
  if you need it guaranteed.
- **Voice recognition can be fooled** by a recording. It's for convenience only.
- **The Jetson setup isn't validated yet.** It's planned, and the settings
  exist, but it hasn't run on the board.
