# Phase 4 specification

Status: **built** (2026-11-02), awaiting live testing.
Drafted from the roadmap line ("Alarms and morning briefing") and Reza's answers on
2026-11-02: a systemd timer starts Orion for alarms when he isn't running; the
briefing has weather, calendar, unread email, and today's alarms and reminders;
weather comes from Open-Meteo for a city that is set, not hard-coded; reminders are
in scope. Every other default below is a proposal; change any of them and the code
follows.

## Phase 4 — Alarms, timers, reminders, weather, morning briefing

Build on Phase 3. Branch `Phase-4` off `Phase-3`. Stop at the end of the phase for
live testing.

### 1. Alarms, timers, reminders by voice
- Alarms: "set an alarm for 7", "wake me up at 6:30 tomorrow", "alarm for 7 every
  weekday", "every day at 8", "every Monday at 9".
- Timers: "set a timer for 10 minutes", "10 minute timer", "pasta timer for 8
  minutes", "timer for an hour and a half".
- Reminders: "remind me to call Mom at 5", "remind me at 5 to call Mom", "remind
  me in 20 minutes to check the oven", "remind me tomorrow at 9 to submit the
  form". No time given: Orion asks "When?" once.
- Times are parsed by code (`core/when.py`), never by the model. A bare hour means
  the next time that hour comes round ("alarm for 7" said at 10 PM is 7 AM); "wake
  me up at 7" always means the morning. Orion repeats what he set, with the day:
  "Alarm set for 7 AM tomorrow."
- Listing: "what alarms do I have?", "what timers are running?", "how long left on
  the timer?", "what are my reminders?".
- Cancelling: "cancel my 7 AM alarm", "cancel the timer", "cancel all my alarms",
  "delete the reminder about Mom". If it's unclear which, Orion asks.
- Alarms and reminders belong to the person who set them; only people Orion
  recognises can set them, and each person hears and cancels only their own.
  Timers are for anyone (a guest's kitchen timer works); a guest's timers are
  shared by everyone.
- Everything is answered locally, before the Claude router. Nothing about alarms
  or reminders is sent to Claude or a web search.

### 2. When something goes off
- Alarm: Orion says "It's 7 AM, Reza." and rings (a generated tone that gets
  louder), with short gaps to listen. "Stop", "I'm up", "okay" or "turn it off"
  stops it; "snooze" (`alarms.snooze_minutes`, default 9) or "snooze for 5
  minutes" snoozes it. No wake word is needed while it rings, and anyone in the
  room can stop it. It gives up after `alarms.ring_minutes` (default 5).
- Timer: "Your 10 minute timer is done." / "Pasta timer's done." and the same
  ring.
- Reminder: a chime, then "Reza, a reminder: call Mom." Orion listens once for
  "remind me again in 10 minutes" / "snooze".
- Muted ("stop listening"): alarms and timers still ring and still hear "stop";
  nothing else changes.
- "Shut down" and the wipe passphrase work while something is ringing.
- Text mode: the ring plays and the terminal says what went off; type "stop" or
  "snooze".
- Anything due while Orion is busy talking goes off as soon as he finishes.

### 3. When Orion isn't running: the systemd timer
- After every change, Orion writes one user timer,
  `~/.config/systemd/user/orion-alarm.timer`, set for the next thing due, minus
  `alarms.start_early_s` (default 90 s, so the models are loaded in time). It
  starts `orion-alarm.service`, which runs `python main.py alarm wake`: if Orion is
  already running it does nothing; otherwise it starts Orion in voice mode, quietly
  (no "Orion online"), and he rings on time. He keeps running afterwards; `python
  main.py stop` stops him.
- The unit files hold only a time: no person, label, or reminder text. They're the
  one plain-text trace of a schedule; the wipe deletes them.
- The service sends Orion's spoken text and your words nowhere (stdout to
  /dev/null). Logs never contain transcripts anyway.
- `alarms.wake_orion: false` turns the timer off (Orion then only rings while
  running).
- Honest limits: the computer must be on, awake, and logged in (user timers run in
  your session, which also has the audio and the keyring). A suspended laptop
  doesn't wake for it.
- At start-up, anything that came due while Orion was off: an alarm or timer
  less than `ring_minutes` late rings now; older ones are reported once ("You
  missed a 7 AM alarm"); missed reminders are said late ("While I was off, you had a
  reminder: call Mom."). Repeating alarms move to their next time.
- `python main.py alarm [list | timer]` shows what's set and the timer state.

### 4. Weather (Open-Meteo)
- Open-Meteo: free, no key, no account. Forecasts use coordinates rounded to two
  decimals (about 1 km), from Open-Meteo's own geocoder.
- No city is hard-coded. "Set my city to Tampa", "change my weather location to
  Austin, Texas", or `python main.py weather city "Tampa, Florida" [NAME]`. Orion
  looks the name up, reads back the match ("Tampa, Florida, United States. Is that
  right?"), and saves it, encrypted, for that person. A guest gets the owner's city.
- No city yet: "I don't know your city yet. Say 'set my city to', then the city."
- Questions: "what's the weather?", "will it rain tomorrow?", "how hot is it?",
  "do I need an umbrella?", "what's the weather this weekend?". Answered by code
  from the forecast (current conditions, high, low, chance of rain), never by the
  model, so a number can't be invented.
- Units: `weather.units: auto` (Fahrenheit in the US, Celsius elsewhere),
  `fahrenheit` or `celsius`.
- Answers are cached for 10 minutes. Each call is in the encrypted audit log:
  kind and success, never the city or coordinates.
- `weather.enabled: false`, or no internet: "I can't get the weather right now."
  Never a guess.

### 5. Morning briefing
- Runs when you stop a morning alarm (one set between 4 and 11 AM;
  `briefing.after_alarm`), or when you say "good morning", "brief me", "morning
  briefing", "what's my day look like".
- In order, each part skipped if it isn't available:
  1. "Good morning, Reza. It's 7:02 on Monday, 2 November."
  2. Weather: now, today's high and low, chance of rain.
  3. Today's calendar (Outlook, read locally, as in Phase 3).
  4. Unread email: the count and up to three senders. No summaries.
  5. Your alarms, timers and reminders still to come today.
- The parts are fetched at the same time (a slow one doesn't hold up the others),
  each with its own timeout. A part that fails gets one short sentence ("I couldn't
  reach Outlook.").
- `briefing.sections` lists which parts to include and in what order.
- People Orion doesn't recognise get the greeting, the weather and nothing personal.
- The conversation log keeps "(morning briefing)", not what was read out.

### 6. Storage, privacy, and kill switches
- Alarms, timers, reminders and weather locations are kept in the encrypted
  database (`schedule` and `user_settings` tables). Deleting a person deletes theirs.
- The nuclear wipe deletes the systemd unit files (and reloads systemd) before the
  data key goes.
- Reminder text comes only from the person's own words, never from an email, a web
  page or the model.
- The capability guard stops refusing weather and alarms once those features are
  wired in; anything they can't do ("set an alarm on my phone") gets an honest no.

### 7. Tests, measurements, docs
- Tests with a fake clock, fake systemd, and a mocked Open-Meteo: time and duration
  parsing (bare hours, AM/PM, "in an hour and a half", tomorrow, weekdays, repeats,
  across DST), set/list/cancel per person, guest timers, firing order, snooze,
  stop, ring gives up, missed at start-up, repeating alarms, the unit file has only
  a time, wipe and user deletion, weather lookup and confirmation, units, cached
  forecast, weather phrases, audit has no city, briefing order and skipped parts,
  guests' briefing, briefing timeouts, the guard.
- Measured (docs/benchmarks.md): time from the systemd timer to Orion ready to
  ring, Open-Meteo latency, briefing time to first word.
- Update README, user manual, and benchmarks.
