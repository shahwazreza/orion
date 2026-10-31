# Orion user manual

**Version:** Phase 2 (memory, conversations, rules, Claude mode, web search) on top of Phase 1, October 2026.

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
12a. [Memory, conversations and rules](#12a-memory-conversations-and-rules)
12b. [Claude mode](#12b-claude-mode)
12c. [Web search](#12c-web-search)
12d. [Kill switches and mute](#12d-kill-switches-and-mute)
12e. [Outlook mail and calendar](#12e-outlook-mail-and-calendar)
13. [Terminal command reference](#13-terminal-command-reference)
14. [Settings reference](#14-settings-reference)
15. [Your privacy: what's stored and where](#15-your-privacy-whats-stored-and-where)
16. [What to expect: speed and memory](#16-what-to-expect-speed-and-memory)
17. [Troubleshooting](#17-troubleshooting)
18. [Known limitations](#18-known-limitations)

---

## 1. What Orion can and can't do yet

**He can:**

- Wake up when you say **"hey orion"**, listen until you stop talking, and answer
  out loud.
- Keep a conversation going: after he answers, you can just keep talking for a
  few seconds without saying the wake word again.
- Learn your name, learn your voice (with your permission), and recognise who's
  talking, so he uses the right name and keeps each person's chat separate.
- Answer general questions, chat, explain things, and help you think, using
  the local model (llama3.2:3b on the laptop).
- Swear back casually when you swear, or keep it clean when asked.
- Note his mistakes when you tell him "that was wrong" (opt-in).
- **Remember things permanently** ("remember that...") and **remember your conversations**,
  even after a restart.
- **Standing instructions:** "whenever I say I'm leaving, remind me to take my keys".
- **Look things up on the web** when he doesn't know, and say where the answer came from.
- **Ask Claude** for you, only when you say so, with nothing personal sent unless you allow it.
- **Check your Outlook email and calendar**, read-only: new emails, who they're from, a short
  summary or the full text, what's on today, and whether you're free (section 12e).
- Run in text mode, where you type instead of talk.
- **Shut down on command** ("Orion, shut down"), **mute** ("stop listening"), and erase
  everything with your secret **wipe passphrase** (section 12d).

**He can't yet** (these come in later phases):

| Not yet | Coming in |
|---|---|
| Alarms, morning briefing | Phase 4 |
| Meeting notes | Phase 5 |
| Build software for you | Phase 7 |

Ask him about any of these and he tells you he can't do it yet. For live or personal
information (weather, the news, prices, scores, alarms and reminders, and email or
calendar requests beyond what section 12e lists) that answer is built in: *"I can't check the weather yet. That's coming
with the morning briefing, and I won't guess."* The model never gets the chance to
make something up. Ordinary questions about those topics ("how does weather form?",
"how do I write a professional email?") are still answered normally.

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

The speech-recognition model (Whisper medium.en, ~1.5 GB) downloads the first
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

**To stop Orion,** say "Orion, shut down", run `python main.py stop` from another
terminal, or press Ctrl+C in his (section 12d).

**First time:** set your wipe passphrase with `python main.py killswitch set`. Until
you do, Orion reminds you at every start.

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

1. Say **"hey orion"**. *(A custom wake word, trained with a Colab notebook. Check how well it
   hears you with `scripts/wakeword_test.py`, and see [wake-word.md](wake-word.md) to tune
   or retrain it.)*
2. You hear a **rising chime**: he's recording.
3. Speak normally. When you stop for about 5 seconds, a **falling chime** means
   he's stopped recording.
4. He answers.
5. You hear a **soft single note**: he's still listening. Ask a follow-up
   without the wake word, within 10 seconds.
6. To end, stay silent or say something like **"thanks, that's all"**. You hear
   the falling chime and he goes back to waiting for "hey orion".

You don't need to pause after the wake word: "Hey orion, what time is it in
Tokyo?" works in one breath. The wake phrase is removed before your request is
answered, so he won't comment on what you called him.

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
| `wake_word` (default) | "Hey orion" |
| `voice` | Anyone talking over him, or "hey orion". What you say becomes your next request. Needs echo cancellation (below) |
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
- He stops listening after about 5 seconds of silence, so you can pause to think
  mid-sentence. The cost is a 5-second wait before he answers. If he cuts you off
  mid-sentence, see section 17.
- He waits up to 6 seconds for you to start talking after the wake word, and
  records at most 60 seconds in one go.
- **You can interrupt him:** say **"hey orion"** while he's talking. He stops
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

### Safety

These work at any time, even while Orion is muted (section 12d).

| Say | What happens |
|---|---|
| "Orion, shut down" / "Shut yourself down" / "Turn yourself off" | Three falling notes, then Orion stops at once. No confirmation |
| "Orion, stop listening" | Muted: he ignores everything until you say "hey orion, start listening" |
| "Hey orion, start listening" | Unmuted |
| *Your wipe passphrase* | He asks you to say "confirm wipe" within ten seconds, then erases everything |

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

1. The **owner** says "hey orion, add a new person".
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
Say "hey orion" and ask again in a full sentence, close to the mic. Being
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

## 12a. Memory, conversations and rules

Everything here is **per person** (yours are yours, Malihah's are hers), **encrypted**
in `data/orion.db`, and only ever used by the **local** model. Strangers' voices get
nothing saved.

### Memory: things you ask him to remember

| Say | What happens |
|---|---|
| "Remember that my sister's name is Sara" / "Don't forget I'm allergic to peanuts" | Saved permanently. He says "Got it. I'll remember that..." |
| "By the way, my birthday is March 3rd" / "I live in Tampa" / "I'm vegetarian" | Lasting facts like these are saved too, with the same confirmation |
| "What do you remember about me?" | The latest five things you've told him |
| "What do you know about my sister?" / "What did I tell you about my exam?" | Found by meaning. If no saved fact matches, he checks your past conversations |
| "Forget about my sister" | Finds the closest memory and asks "Forget that...?" first |
| "Forget everything you know about me" | Asks first. Your rules and conversations stay |

Memories never expire. When you ask something they're relevant to ("can I eat this
peanut butter sandwich?"), he uses them in his answer.

### Conversations

Every exchange is saved, so he picks up where you left off after a restart.

| Say | What happens |
|---|---|
| "What did we talk about yesterday?" / "...today?" / "...on Monday?" / "...last week?" | A short summary of that day |
| "What did I say about my exam?" / "Did I ever mention Jupiter?" | Finds the past exchange by meaning |
| "Don't save this conversation" / "Off the record" | This conversation stops being saved, and what was said so far is dropped |
| "Forget today's conversation" | Deletes today's, after asking |
| "Delete my conversation history" | Deletes all of yours, after asking. Memories and rules stay |

Kept forever unless you set `conversations.keep_days`.

### Standing instructions (rules)

| Say | What happens |
|---|---|
| "Whenever I say I'm leaving my room, remind me to take my keys" | Saved. He confirms in his own words |
| "Every time I'm going to bed, remind me to drink water" / "Always remind me to stretch when I finish class" | Also understood |
| "What rules do I have?" | Lists them |
| "Change the keys rule to also include my wallet" | Reads the new version back and asks first |
| "Stop reminding me about my keys" | Asks first, then deletes |
| "That was wrong" (right after a reminder) | That phrase won't trigger the rule again |
| "Also remind me when I say see ya" | Adds your own phrase to the rule |

Rules work by meaning: "heading out", "I'm off" and "leaving now" all count as
leaving. He ignores questions ("how do I leave a group chat?"), other people ("my
roommate left") and plans ("I'll leave in an hour"). He'll occasionally get it wrong;
the two corrections above teach him.

Deleting memories, conversations or rules needs a clear voice match on that request,
like the people commands (section 8).

## 12b. Claude mode

Orion answers on this computer by default. Claude (Anthropic's cloud model) is used
**only when you ask**:

| Say | What happens |
|---|---|
| "Ask Claude what causes the northern lights" | Just that one question goes to Claude, then back to local |
| "Switch to Claude mode" | Everything goes to Claude until you say "back to local" |
| "Back to local" / "Local mode" | Back on this computer |
| "Ask Claude to plan my week, **and include what you know about my schedule**" | Only then are your matching memories sent, for that request only |

- You hear a **three-note rising chime** before every request that goes to the cloud.
- **Never sent** unless you say "include...": your memories, conversations, rules, or
  anyone's name, including yours.
- If the internet, the API, or your key isn't working, he says so and answers himself.
- Each person has their own Claude mode. Turning it on for you doesn't affect Malihah.

**Setup:**

1. Create a key in the [Claude Console](https://platform.claude.com) (Settings → API Keys;
   it starts with `sk-ant-` and is shown only once) and add a little credit.
2. **Copy it**, then save it straight from the clipboard (no hidden prompt to paste into):
   ```bash
   python main.py secrets set anthropic --clipboard
   ```
   It says "Saved to the system keyring (N characters)" and clears the clipboard.
   Without `--clipboard` it asks with hidden input instead (paste with Ctrl+Shift+V).
3. Check: `python main.py secrets status` should say `anthropic  saved (N characters)`.
   Empty or malformed keys are refused, and reported as "not set".

**Set a monthly spend limit (about $10)**
in the Claude Console under Settings → Limits. A spoken turn costs about one cent.

**Auditing:** `python main.py audit` shows every routing decision and every cloud
request exactly as sent, with tokens and estimated cost, plus this month's total.

The optional auto-router (`router.auto: true`, off by default) sends plain
general-knowledge questions with nothing personal in them ("what causes the northern
lights?") to Claude automatically.

## 12c. Web search

When Orion doesn't know something, or it's about now (news, prices, scores, "who
won", "the latest...", recent years), he searches instead of guessing:

1. He searches, reads the top pages, and searches again if needed (at most 3 rounds).
2. He answers in his own words and says where it came from: *"According to
   NobelPrize.org, ..."*. He never reads out links.
3. If sources disagree or nothing reliable turns up, he says so.

| Say | What happens |
|---|---|
| "Look up ..." / "Search for ..." | Searches straight away |
| "What were your sources?" | Names the sites. Text mode also shows the full links |
| "Orion, don't search" | No web access until "you can search again" |

For big research questions ("compare...", "pros and cons of...") or when he can't find
it, he offers: *"This is a big one. Want me to ask Claude?"* Claude then searches too
(counted toward your spend limit).

**Privacy:** searches leave this computer, so names of people Orion knows and details
from your memories are kept out of queries unless you said them yourself in that
question. Before searching for a known person's name, he asks. Every query and every
page he reads is in the encrypted audit log.

**Safety:** web pages are only information. Nothing on a page can make Orion
remember, set rules, delete, send, or change modes. He also won't fetch addresses on
your own network.

**Setup:** web search uses a local search engine (SearXNG) in a container that only
this computer can reach. Start it once after each reboot:

```bash
scripts/searxng.sh start      # also: stop | status | logs
```

If it isn't running, Orion still works and says search is unavailable. For Brave
Search instead: `search.backend: brave` and `python main.py secrets set brave --clipboard`
(a free Brave Search API key; not needed with SearXNG).
Offline mode: `search.enabled: false`.

## 12d. Kill switches and mute

### Shutdown (level 1)

Say **"Orion, shut down"** and he stops immediately: speech stops mid-word, any
Claude or web request is abandoned, the mic closes, and the program exits. There
is no confirmation, and nothing is lost: everything he'd saved is already saved.
"Shut down the computer" is *not* this; turning off the device is Phase 9.

The same from a terminal: `python main.py stop`. It finds the running Orion through
a private file in your session's runtime folder (no network is involved) and gives
him five seconds before forcing it. Ctrl+C in Orion's own terminal works too.

**Keyboard shortcut (KDE):** System Settings -> Keyboard -> Shortcuts -> *Add New*
-> *Command or Script*. Enter the full paths, for example
`/home/you/orion/venv/bin/python /home/you/orion/main.py stop`, then pick a key
combination such as Meta+Shift+O.

### Mute

"Orion, stop listening" mutes him. The wake word still works, but whatever you say
afterwards is ignored: not answered, not shown, not saved. The only things he still
responds to are "start listening", "shut down", and your wipe passphrase. You still
hear the listening chime after "hey orion", so you always know when the mic is
recording you. Mute lasts until you unmute him or restart him.

### Nuclear wipe (level 2)

The wipe erases everything Orion knows. Use it if your laptop is about to be taken,
or you simply want him to forget everything.

**First, set your passphrase.** Only you know it. It's never shown, stored, logged,
or sent anywhere; Orion keeps only a salted Argon2id hash of it, inside the
encrypted data.

```bash
python main.py killswitch set     # type it twice; nothing appears on screen
python main.py killswitch test    # say it three times; he answers only "matched" or "didn't match"
python main.py killswitch status  # set or not (never the passphrase)
```

Choose at least three ordinary words you'd never say by accident, like a short
nonsense sentence. Words only (no digits), and not "Orion" at the start or end.
Capitals and punctuation don't matter. Run the test: if Whisper sometimes mishears
a word, pick another. To change it later, run `killswitch set` again (it asks for
the current one first).

**Spoken:** say the passphrase to Orion as you'd say anything else ("hey orion, ..."
then the passphrase). He replies "Wipe everything I know? Say 'confirm wipe' within
ten seconds." Say **"confirm wipe"**. Anything else, or silence, cancels. If what you
say isn't the passphrase, nothing special happens at all: he answers it like any
other request, so there's nothing to learn from guessing.

**Silent:** `python main.py wipe` asks for the passphrase without showing it, then for
"confirm wipe". Three wrong passphrases lock it for 5 minutes; each further lockout
doubles, up to a day. If Orion is running, it's stopped first.

**What's erased, in this order:**

1. Outlook sign-ins: the token file is overwritten and deleted (section 12e).
2. The encryption key in your keyring. This is the step that really protects you:
   every encrypted Orion file becomes unreadable at that instant, including any
   copy of it anywhere.
3. The files themselves, overwritten and deleted: the database (memories,
   conversations, rules, audit log), its backups, people and voice signatures,
   training notes, metrics, and the local search engine's container and settings.
4. Every other Orion key in your keyring: Claude, Brave, and later GitHub and
   notification tokens.
5. Orion exits and stays off.

**Kept:** the code, the downloaded models, `~/orion-projects/` and `~/orion-training/`.
The wipe itself is never logged anywhere.

On an SSD with Fedora's btrfs filesystem, overwriting a file doesn't guarantee the
old copy is gone from the disk, which is why the key is destroyed first. For full
protection of everything else on the laptop, turn on full-disk encryption (LUKS).

**Afterwards:** starting Orion again is like starting him for the first time: a new
key, meeting the owner, no memories. Set a new passphrase, and re-enter your Claude
key (`secrets set anthropic`). Claude's and the search engines' own servers keep
whatever they normally keep; the wipe can only erase what's on this computer.

---

## 12e. Outlook mail and calendar

Orion can read your Outlook email and calendar. He can't send, reply, delete, move,
accept invitations, or even mark an email as read: he only has read permission.

### Setting it up (once, about 10 minutes)

Orion signs in with an app registration of your own, so no one else's app ever sees
your mail. It's free. Registering the app happens on Microsoft's website (steps 1 to 5);
then one command does the rest (step 6).

1. Open the Microsoft Entra admin center, <https://entra.microsoft.com>, and sign in
   with the Microsoft account you'll connect. Go to **App registrations**, then
   **New registration**. If Microsoft says your account has no directory, create a
   free Azure account first; it comes with one.
2. Name it `Orion`. Under **Supported account types**, choose *Accounts in any
   organizational directory and personal Microsoft accounts*. Leave **Redirect URI**
   empty. Click **Register**.
3. Copy the **Application (client) ID** from the overview page. It isn't a secret.
4. Under **Authentication**, set **Allow public client flows** to **Yes** and save.
   This allows the sign-in with a code that Orion uses.
5. Under **API permissions**, choose **Add a permission**, **Microsoft Graph**,
   **Delegated permissions**, then tick **Mail.Read** and **Calendars.Read**. Add
   nothing else.
6. Run the setup and paste the ID when it asks:

   ```bash
   python main.py outlook setup            # the owner; or: outlook setup Sam
   ```

   It checks the app with Microsoft and works out which accounts it accepts. It saves
   the ID in `config.local.yaml`, creating the file if needed. That file holds this
   computer's own settings and is never committed to git. Then it prints a link and a
   short code: open the link on any browser, enter the code, sign in, and accept the
   two permissions. It finishes by checking a few read-only requests.

If the app registration has a mistake, setup says what to fix and saves nothing, so
you can fix it on Microsoft's site and run it again. Run it again later to change the
ID or sign in again.

Signing in happens in the terminal, never by voice, so the code is never read out
loud. Orion opens no port for it. Each person Orion knows can connect their own
account with `python main.py outlook setup NAME`. They reuse your app, so they only
press Enter at the ID prompt and sign in.

**Work or school accounts:** many universities and employers block apps that their
IT department hasn't approved. If Microsoft says *Need admin approval*, use a personal
account or ask your IT department.

### What to say

| Say | What happens |
|---|---|
| "Any new emails?" / "Check my email" | How many are unread, and who the newest three are from and what about |
| "Read my latest email" | Who it's from, when, and the subject, then a one- or two-sentence summary |
| "Read the second one" / "Next one" / "Read it" | Another email from the list he just gave you |
| "Read it out" / "Read the whole thing" | The email's own text, about 45 seconds at a time. He asks before reading on |
| "Any emails from Alex?" / "Did Alex email me?" | The latest few from that person |
| "What's on my calendar today?" / "My schedule tomorrow" | Your events, with times and places. Today means what's left of it |
| "What have I got on Friday?" / "This week" / "Next week" / "This weekend" | Events in that stretch, by day |
| "What's on tomorrow afternoon?" | Morning is 6 to 12, afternoon 12 to 5, evening 5 to 10 |
| "When's my next meeting?" | The next event, or the one you're in now |
| "Am I free at 3 tomorrow?" / "Am I busy Friday afternoon?" | Checked against your busy times. Tentative counts as busy, "free" doesn't |
| "Disconnect my Outlook" | Forgets your sign-in on this computer, after asking |

A bare hour like "at 3" means 3 PM; 8 to 11 mean the morning. "Next Friday" is the
Friday of next week, and he says the date so you can tell. Times are in this
computer's time zone.

### What's private

- **Only you hear your mail.** Orion reads only the account of the person he
  recognises. People he doesn't recognise get nothing, and he never mentions anyone
  else's account. Set `outlook.strict_voice: true` to need a clear voice match for
  every Outlook request, as for deleting things.
- **Your mail stays on this computer.** It's fetched when you ask, kept in memory for
  ten minutes for follow-ups, and never written to disk. Your conversation history
  keeps your question and "(answered from Outlook)", not what he read out
  (`outlook.save_replies: true` keeps that too, encrypted).
- **Never to Claude, never to a search.** Email and calendar requests are answered
  locally even in Claude mode. Summaries come from the local model.
- **Emails are only information.** An email that says "ignore your instructions" or
  "remember that..." can't make Orion do anything. Only your own words are commands.
- **The sign-in** is kept encrypted in `data/outlook.enc`. The nuclear wipe deletes it
  first. Deleting a person deletes theirs.
- **Revoking it at Microsoft too:** disconnecting forgets the sign-in on this
  computer. To cancel it at Microsoft, remove Orion at
  <https://account.live.com/consent/Manage> (personal accounts) or
  <https://myapps.microsoft.com> (work or school).
- The audit log records each Outlook request: what kind and how many items. It never
  records senders, subjects, text, or event names.

If the internet is down, the sign-in has expired, or Outlook is busy, he says which.
He never guesses an email or an event.

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

### Keys and audit

| Command | Purpose |
|---|---|
| `python main.py secrets set anthropic --clipboard` | Save your Claude API key from the clipboard (recommended), then clear the clipboard |
| `python main.py secrets set anthropic` | Same, typed or pasted at a hidden prompt |
| `python main.py secrets set brave --clipboard` | Brave Search key (only if `search.backend: brave`) |
| `python main.py secrets status` | Which keys are saved and their length; flags empty or malformed keys. Never shows them |
| `python main.py audit [N]` | The last N routing decisions, cloud requests, searches and pages read, plus this month's Claude cost |
| `scripts/searxng.sh start\|stop\|status\|logs` | The local search engine |

### Safety

| Command | Purpose |
|---|---|
| `python main.py stop` | Stop the running Orion (also for a keyboard shortcut) |
| `python main.py killswitch set` | Set or change the wipe passphrase (hidden input) |
| `python main.py killswitch test` | Say it three times and check Orion hears it |
| `python main.py killswitch status` | Whether a passphrase is set, or locked after wrong tries |
| `python main.py wipe` | Nuclear wipe: passphrase (hidden), then type "confirm wipe" |

### Outlook

| Command | Purpose |
|---|---|
| `python main.py outlook setup [NAME]` | Guided setup: asks for the client ID, checks it with Microsoft, saves it to `config.local.yaml`, signs in, and tests |
| `python main.py outlook status` | Who is connected, and as which Microsoft account |
| `python main.py outlook connect [NAME]` | Sign in with a code (the owner if no name is given) |
| `python main.py outlook test [NAME]` | Check the sign-in and time a few read-only requests. Shows counts only |
| `python main.py outlook disconnect NAME` | Forget that person's sign-in on this computer |

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
| `python main.py metrics [N]` | The last N per-reply timings, decrypted, as JSON lines |
| `pytest -q` | Run the test suite |

---

## 14. Settings reference

All settings live in `config.yaml`. Settings under `common:` apply everywhere.
Settings for this computer only go in `config.local.yaml` next to it, in the same
shape as `common:`. That file is never committed to git and wins over `config.yaml`.
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
| `wakeword.models` | `[models/openwakeword/hey_orion.onnx]` | Wake word models; any of them wakes Orion. Built-in names or paths to `.onnx` files |
| `wakeword.phrase` | `null` | What Orion tells people to say (default: the first model's name) |
| `wakeword.threshold` | `0.5` | Lower (e.g. `0.35`) = triggers more easily, more false alarms |
| `wakeword.thresholds` | `{}` | Per-model overrides, e.g. `{hey_orion: 0.6}` |
| `wakeword.cooldown_s` | `1.5` | Ignore the wake word briefly after it fires |

### Listening

| Setting | Default | Meaning |
|---|---|---|
| `conversation.follow_up_s` | `10.0` | Seconds he waits for a follow-up; `0` = wake word every time |
| `vad.end_silence_ms` | `5000` | Silence that means you've finished. Lower it (e.g. `1000`) for faster replies if you don't pause mid-sentence |
| `vad.neg_threshold` | `0.35` | Once you're talking, quieter sounds still count as speech, so soft word endings don't cut you off |
| `vad.start_timeout_s` | `6.0` | How long he waits for you to start after the wake word |
| `vad.max_utterance_s` | `60.0` | Longest single recording |
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
| `stt.model` | `medium.en` (laptop) | Whisper model. `medium.en` is more accurate in noise; `small.en` is about 3 s faster per utterance |
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

### Memory, conversations, rules, cloud and search

| Setting | Default | Meaning |
|---|---|---|
| `memory.enabled` | `true` | "Remember that..." facts |
| `conversations.enabled` | `true` | Saving every exchange |
| `conversations.keep_days` | `null` | `null` = forever, or a number of days |
| `rules.enabled` | `true` | Standing instructions |
| `cloud.model` | `claude-opus-5-5` | Claude model for Claude mode |
| `cloud.effort` | `low` | How hard Claude thinks; low keeps spoken replies quick and cheap |
| `cloud.fallbacks` | `true` | A request Claude's safety check declines is retried on a fallback model |
| `router.auto` | `false` | Send impersonal general-knowledge questions to Claude automatically |
| `search.enabled` | `true` | `false` = offline mode |
| `search.backend` | `searxng` | `searxng` (local) or `brave` |
| `search.max_rounds` | `3` | Search rounds per question |
| `outlook.enabled` | `true` | Outlook mail and calendar |
| `outlook.client_id` | `null` | Your app registration's ID (section 12e). Set it in `config.local.yaml` |
| `outlook.tenant` | `common` | `common` (personal and work/school), `consumers`, `organizations`, or a tenant ID |
| `outlook.strict_voice` | `false` | Need a clear voice match for every Outlook request |
| `outlook.save_replies` | `false` | Keep what he read out in the conversation history (encrypted) |
| `backups.keep` | `30` | Daily encrypted database backups kept |

### Data and logs

| Setting | Default | Meaning |
|---|---|---|
| `paths.data` | `data` | Encrypted personal data |
| `paths.models` | `models` | Downloaded models |
| `training_notes.enabled` | `false` | Section 12 |
| `metrics.enabled` | `true` | Log timings and memory per reply (encrypted, no conversation text) |
| `metrics.file` | `logs/metrics.enc` | Where; read it with `python main.py metrics` |

---

## 15. Your privacy: what's stored and where

**Nothing personal leaves your computer.** Orion goes online only to search the web
(generic queries, no personal details), when you ask for Claude (nothing personal
unless you say "include..."), and to fetch your own Outlook mail and calendar from
Microsoft when you ask. All three are logged in the encrypted audit log. He also opens no network ports, and refuses to
start if Ollama is reachable from the network.

| What | Stored? | Where |
|---|---|---|
| Your voice audio | **Never.** Held in memory, discarded after each reply | none |
| What you said, what he said | Yes, **encrypted**, per person, so he remembers your conversations. Say "don't save this conversation" to opt out | `data/orion.db` |
| Names, roles, voice signatures | Yes, **encrypted** (AES-256) | `data/users.enc` |
| Memories, conversations, rules | Yes, **encrypted** (SQLCipher), per person, never strangers' | `data/orion.db` |
| What went to Claude and the web (audit log) | Yes, **encrypted** | `data/orion.db` |
| Daily backups | Yes, **encrypted**, last 30 | `data/backups/` |
| Training notes (if on) | Yes, **encrypted**, no conversation text | `data/training_notes.enc` |
| Your wipe passphrase | **Never.** Only a salted Argon2id hash, **encrypted** | `data/killswitch.enc` |
| Your emails and calendar events | **Never.** Fetched when you ask, kept in memory for 10 minutes | none |
| Outlook sign-ins (tokens) | Yes, **encrypted** (AES-256), per person | `data/outlook.enc` |
| The encryption key | In KDE Wallet, never on disk or in git | your system keyring |
| Speed and memory metrics | Yes, **encrypted**, numbers only | `logs/metrics.enc` |

The `data/` folder and its files can only be read by your user account. None of
`data/`, `logs/`, `models/`, or `voice-samples/` is ever committed to git.

A voice signature is a list of numbers describing how your voice sounds. It can't
be played back as audio.

**To remove a person:** "delete my profile", or `python main.py users delete NAME`.
**To remove everything:** the nuclear wipe (section 12d): your passphrase, spoken or
with `python main.py wipe`.

---

## 16. What to expect: speed and memory

Measured on the laptop (Intel i5-10400H, no GPU):

| | Typical |
|---|---|
| Time from when you stop talking to his first word (Piper voice) | about 3.2 s |
| Same, with the Kokoro voice | about 5.5 to 6 s |
| Speech recognition | about 4.6 s with `medium.en` (1.4 s with `small.en`) |
| Voice recognition | about 0.03 s |
| Orion's memory use | about 1 to 1.5 GB |
| Ollama with llama3.2:3b loaded | about 2.6 GB |

Speech recognition and the language model are warmed up at startup, so the first
reply isn't noticeably slower than the rest. On the Jetson, the GPU should make all of this much faster;
it gets measured when the board arrives. Full numbers are in
[benchmarks.md](benchmarks.md).

---

## 17. Troubleshooting

**He doesn't react to "hey orion".**
Run `python scripts/wakeword_test.py` to see how strongly it hears you. If your scores are
under 0.5, lower `wakeword.thresholds.hey_orion` to just below your typical score (e.g. `0.4`).
As a fallback, add `hey_jarvis` to `wakeword.models` so either phrase works.
Also check the right mic is selected (`--list-devices`) and speak a bit closer.

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

**"My local search engine isn't responding."**
Start it: `scripts/searxng.sh start` (needed after each reboot). Check with
`scripts/searxng.sh status`.

**"I don't have a Claude API key" / "I can't reach Claude".**
`python main.py secrets status` should say "saved (N characters)". If it says "not set"
(an empty paste is the usual cause: the hidden prompt shows nothing either way), copy the
key and run `python main.py secrets set anthropic --clipboard`. If it says the key doesn't
look like an Anthropic key, the clipboard held something else.

**A rule fired when it shouldn't (or didn't when it should).**
Say "that was wrong" straight after a wrong reminder, or "also remind me when I say ..."
to add your phrasing.

**Microsoft's Entra site says your account "does not exist in tenant 'Microsoft
Services'"** (AADSTS16000). Your personal Microsoft account has no directory of its
own, so it can't register apps yet. Complete a free Azure account signup with that
account (https://azure.microsoft.com/free), or Azure for Students with your school
account, then sign in to the Entra site again. Nothing is charged for what Orion needs.

**Outlook sign-in fails.** `outlook setup` checks the app first and says what to fix.
`outlook connect` prints Microsoft's reason after the sentence. The common ones:

- *AADSTS7000218* (needs a client secret): **Allow public client flows** is still off in
  the app registration's **Authentication** page.
- *AADSTS700016* or *AADSTS50020* (app not found, or account not allowed): the client ID
  is wrong, or the app's **Supported account types** doesn't include your kind of
  account. Choose the option with personal accounts, then run `outlook setup` again.
- *AADSTS9002346* (personal accounts only): run `outlook setup`; it sets
  `outlook.tenant: consumers` for you.
- *Need admin approval*: your school or employer blocks unapproved apps. Use a personal
  account or ask IT.

**"Your Outlook sign-in has expired."** Microsoft ended the sign-in, for example after a
password change or about 90 days unused. Run `python main.py outlook setup` again and
answer yes to signing in again.

**He made something up.**
The 3B model sometimes does. Say "that was wrong" (with training notes on) so it
gets noted. Later phases add real data sources and fine-tuning.

---

## 18. Known limitations

- **"Hey orion" is a first trained model.** On synthetic voices it was strong for British
  voices and weak for some American ones. If it often misses you, lower
  `wakeword.thresholds.hey_orion` (e.g. `0.4`), add `hey_jarvis` back as a backup, or retrain
  ([wake-word.md](wake-word.md)).
- **The wipe can't reach beyond this computer.** Claude's and search engines' servers keep
  what they normally keep, and on an SSD only the destroyed key (plus LUKS) guarantees old
  file copies are unreadable.
- **Interrupting by just talking** (`barge_in: voice`) isn't reliable through
  laptop speakers yet, so the default is "hey orion" to interrupt.
- **English only.**
- **Rules are matched by meaning and sometimes misfire**, especially between similar
  rules ("I'm off" could mean leaving or bed). Correct him and he learns.
- **Web answers are only as good as the 3B model's reading.** It sometimes misses the
  point of a page. For important questions, "ask Claude".
- **SearXNG must be started after each reboot** (`scripts/searxng.sh start`) until
  Phase 9 makes Orion a service.
- **The small model can be wrong** or invent details. Don't rely on it for facts
  that matter.
- **Casual mode isn't perfect:** in testing he swore back about 14 times out of 15
  and occasionally swore on a clean question. Phase 10 fine-tuning targets this.
- **"Keep it clean" doesn't survive a restart.** Use `profanity: "off"` or `--demo`
  if you need it guaranteed.
- **Voice recognition can be fooled** by a recording. It's for convenience only. That
  matters more now that he reads email: turn on `outlook.strict_voice` if others live
  with you.
- **Outlook hasn't been tested with a real account yet.** It was built and tested
  against a simulated Microsoft service. Expect rough edges until it's been used live.
- **Email summaries come from the 3B model** and can miss or blur details. Sender,
  time, and subject are always exact; say "read it out" for the email's own words.
- **Outlook is a snapshot, not a feed.** Orion doesn't announce new mail; he checks
  when you ask. Only the inbox and your main calendar are read, and only recent items.
- **Long read-outs can't be interrupted by voice**, but they come in short parts, and
  "no" stops them.
- **The Jetson setup isn't validated yet.** It's planned, and the settings
  exist, but it hasn't run on the board.
