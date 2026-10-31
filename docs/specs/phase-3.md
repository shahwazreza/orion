# Phase 3 specification

Status: **built** (2026-10-30), awaiting live testing with a real Outlook account.
Drafted from the roadmap line ("Outlook mail and calendar, Microsoft Graph, device
code flow, read-only") and the Phase 2 / 2.5 rules. Every default below is a
proposal; change any of them and the code follows.

## Phase 3 — Outlook mail and calendar (read-only)

Build on Phase 2.5. Branch `Phase-3` off `Phase-2.5`. Stop at the end of the phase
for live testing.

### 1. Connecting an account
- Your own app registration in Microsoft Entra (free). Its client ID goes in
  `outlook.client_id`. It is not a secret: public clients have no client secret.
  Orion never borrows another app's client ID.
- `outlook.tenant: common` accepts personal (outlook.com) and work or school
  accounts. `consumers`, `organizations`, or a tenant ID narrow it.
- Sign-in uses the **device code flow** from the terminal:
  `python main.py outlook connect [NAME]`. Orion prints a Microsoft URL and a short
  code; you sign in on any browser. No redirect URI, no local web server, no open
  port (dorm rule). NAME defaults to the owner.
- Saying "connect my Outlook" explains the terminal command. Sign-in never
  happens by voice, because the code would have to be read out loud.
- Permissions requested: `Mail.Read` and `Calendars.Read` (delegated). MSAL adds
  `openid`, `profile` and `offline_access` itself. Nothing that can send, write,
  move, delete, or mark as read. A test fails if a write scope is ever added.
- `python main.py outlook status` lists who is connected, as which address.
  `outlook disconnect NAME` forgets that person's tokens.

### 2. Tokens and the kill switch
- The MSAL token cache is kept per person in `data/outlook.enc`, AES-256-GCM with
  the data key, like every other Orion file. Never in the keyring, config, logs,
  or plain text.
- The wipe gains its first sign-out hook: the token file is shredded before the
  data key is deleted. Deleting a person (`users delete`) deletes their tokens.
- Honest limit: a read-only app has no permission to revoke its own refresh token
  at Microsoft. Deleting the local cache makes it unusable from Orion. To revoke
  it at Microsoft too, remove the app at account.live.com/consent (personal) or
  myapps.microsoft.com (work or school). The manual says so.

### 3. Who can read what
- Each person Orion knows can connect their own account. Orion reads only the
  account of the person it recognises. Guests and unrecognised voices get nothing.
  Other people's accounts are never mentioned.
- Recognition uses the normal voice threshold. `outlook.strict_voice: true`
  requires the strong match (`people.verify_threshold`) for every Outlook request.
- Text mode acts as the owner, as before.

### 4. Mail (read-only)
- "Any new emails?", "check my email", "how many unread emails do I have?":
  unread count, then up to three senders and subjects, newest first.
- "Read my latest email", "read the email from Alex", "read the second one",
  "next one": sender, when, subject (spoken by code, never the model), then a
  one- or two-sentence summary from the **local** model.
- "Read it out" / "read the whole thing": the email's own text, cleaned for
  speech, about a minute at most.
- "Any emails from Alex?": the latest few from that sender.
- Reading an email never marks it as read (Graph GET only).

### 5. Calendar (read-only)
- "What's on my calendar today?", "my schedule tomorrow", "what have I got on
  Friday?", "this week", "next week", "this weekend", "tomorrow afternoon":
  events listed by code with times and places. Cancelled events are skipped.
- "What's my next meeting?": the next event, or the one you're in now.
- "Am I free at 3 tomorrow?", "am I busy Friday afternoon?": answered from your
  busy times. A bare hour means 8 to 11 in the morning, 12 to 7 in the
  afternoon or evening.
- Times are spoken in this computer's time zone ("10 AM", "7:30 PM").

### 6. Privacy and safety
- Email and calendar data is fetched live and kept in memory for follow-ups
  only (cleared after 10 minutes). It is never written to disk.
- The conversation log saves the question and "(answered from Outlook)", not
  the reply. `outlook.save_replies: true` saves the reply too (encrypted).
- Outlook requests are always answered locally, even in Claude mode. Email and
  calendar data never goes to Claude or into a web search query.
- Email text is untrusted DATA. The local model gets it inside `<email>` tags,
  told to ignore anything that looks like an instruction, with no tools. Email
  can never trigger commands, memory, rules, searches, or kill switches: those
  are parsed only from the person's own words.
- Added after measuring: an email whose text is addressed to an assistant
  ("ignore all previous instructions", "SYSTEM: the assistant must ...") isn't
  summarised at all. Orion says it contains instructions he ignored and may be
  phishing. The 3B model repeated planted text in half its summaries otherwise.
- Every Graph call is logged in the encrypted audit log: what kind, how many
  items, success or error. No subjects, senders, bodies, or tokens.

### 7. Failures and the capability guard
- No internet, sign-in expired, permission missing, Microsoft throttling, or
  Graph down: Orion says which, in one sentence. He never guesses an email or
  event.
- Not connected: "I can't read your email yet. Outlook isn't connected for you."
- Connected, but an email or calendar request Orion can't do yet (for example
  "summarise last week's emails"): the guard answers honestly instead of the
  model.

### 8. Tests, measurements, docs
- Tests with a mocked Graph and a fake MSAL: read-only scopes, device-code
  connect, encrypted token cache (no token on disk in plain text), per-person
  isolation, guests refused, strict voice, unread / latest / from / ordinal /
  next / read-out, calendar ranges and free/busy, date and time phrases,
  injection (an email saying "remember that ..." or "ignore your instructions"
  changes nothing), audit has no content, error messages, the wipe hook and
  user deletion, the guard with and without Outlook, Outlook before Claude mode.
- Measured (docs/benchmarks.md): summary time and injection resistance with
  llama3.2:3b on synthetic emails. Still to measure once an account is connected:
  Graph latency for the unread list, a calendar day, and one email
  (`python main.py outlook test`).
- Update README, user manual, and benchmarks.
