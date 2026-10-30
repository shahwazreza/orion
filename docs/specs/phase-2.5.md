# Phase 2.5 specification

Status: **built** (2026-10-29), awaiting live voice testing. Decisions confirmed:
fixed confirmation phrase, shutdown chime, mute with the wake word still on.

## Phase 2.5 — Security: encryption at rest and both kill switches

Must be done before any real personal data comes in. Branch `Phase-2.5` off `Phase-2`.

### 1. Encryption at rest
- Everything Orion stores is encrypted with one 256-bit data key in the system
  keyring (Secret Service): SQLCipher for the database and its backups, AES-256-GCM
  for files (people and voiceprints, training notes, the passphrase hash), and an
  append-only AES-GCM record per line for the metrics log (`logs/metrics.enc`,
  read with `python main.py metrics`). The old plain-text metrics file is moved in
  and shredded once every row reads back.
- No keyring: Orion runs, but saves nothing.
- Raw audio is never written to disk (meeting audio arrives in Phase 5 and is
  deleted after transcription unless kept on request).
- README recommends LUKS on both devices.

### 2. Level 1 kill switch: shutdown
- "Orion, shut down" / "shut yourself down" / "turn yourself off" / "power off":
  matched on the whole utterance, no confirmation. "Shut down the computer" is not
  this (Phase 9).
- Stop now: speech interrupted, a short falling chime, Claude/search requests
  abandoned, mic closed, exit. A watchdog forces the exit after 3 s.
- `python main.py stop` (and a KDE shortcut running it): finds the running Orion via
  a process file in `$XDG_RUNTIME_DIR/orion/` (0700) and sends SIGTERM; 5 s, then kill.
  No port. A PID that isn't Orion's `main.py` is ignored.
- Ctrl+C takes the same path. Deliberate stops exit 0, so the Phase 9 systemd unit
  (`Restart=on-failure`) won't restart Orion.
- A second Orion refuses to start while one is running.

### 3. Level 2 kill switch: nuclear wipe
- Passphrase set by the owner with `python main.py killswitch set` (hidden input,
  twice; changing it needs the current one). Never hardcoded, logged, printed,
  committed, or asked for. Orion reminds at startup until one is set.
- Stored only as a salted Argon2id hash (OWASP: 19 MiB, t=2, p=1), inside
  `data/killswitch.enc` (encrypted with the data key).
- Rules for a passphrase: at least 3 words and 12 letters, no digits, none of
  Orion's command phrases, not starting or ending with "Orion". Compared
  case- and punctuation-insensitively.
- `killswitch test`: say it three times; Orion only answers matched / didn't match.
- Spoken trigger: every utterance is checked in the voice loop right after
  transcription, before it is shown, logged, identified, or saved. A match asks
  "Wipe everything I know? Say 'confirm wipe' within ten seconds." Anything else
  cancels. A non-match has no visible effect. Spoken checks aren't counted (no
  feedback means nothing to learn from guessing). Works while muted.
- Silent trigger: `python main.py wipe`: passphrase (hidden), then type
  "confirm wipe". 3 wrong passphrases lock for 5 minutes, doubling per lockout up
  to 24 h, persisted in the encrypted file. Failure says only "Not accepted."
- Wipe sequence (`security/wipe.py`), every step runs even if one fails:
  1. sign-out hooks (Microsoft from Phase 3);
  2. delete the data key (crypto-shredding: all encrypted files unreadable);
  3. overwrite with random bytes, fsync, delete: `data/` (database, backups, users,
     training notes, passphrase hash) and `logs/`; remove the SearXNG container and
     its config;
  4. delete every keyring entry under `orion` (known names plus a Secret Service
     search, so later phases' tokens are included);
  5. exit and stay off.
- Never touched: the code, models, `~/orion-projects/`, `~/orion-training/`. Only
  folders inside the Orion directory are erased, never one containing those.
- Nothing about the wipe is logged.
- Documented honestly: on SSD + btrfs, overwriting doesn't guarantee old blocks are
  gone; the destroyed key and LUKS do.

### 4. Mute
- "Orion, stop listening": the wake word stays on; after it, everything is ignored
  (not answered, shown, identified, or saved) except "start listening",
  "shut down", and the wipe passphrase. The listening chime still plays, so
  recording is always audible.

### 5. Housekeeping
- `requirements.in` / `requirements.txt` gain the Phase 2 packages that were missing
  (anthropic, sqlcipher3-wheels, trafilatura, lxml) plus argon2-cffi and secretstorage.

### 6. Tests and measurements
- Tests: passphrase matching and near misses, hash-only encrypted storage, salting,
  restart, weak-passphrase rules, lockout and doubling, spoken checks uncounted,
  shutdown / mute / confirm phrases, the voice loop (passphrase never shown, logged,
  or answered; cancel; shutdown; mute; kill switches while muted), wipe order (key
  first), protected folders, SearXNG removal, no logging during a wipe, every
  keyring name in the code is in the wipe list, `stop` via the process file, stale
  and foreign PIDs, encrypted metrics.
- Measured: Argon2id cost per check; command phrases through Whisper on six voices
  (docs/benchmarks.md).
