# Phase 2 specification

Status: **built** (2026-10-29), awaiting live voice testing. All proposed defaults were confirmed.

## Phase 2 — Router, Claude mode, web search, memory, conversations, rules

Build on Phase 1 (voice loop, people + voice recognition, language modes,
barge-in, capability guard, training notes). Work on a new branch `Phase-2` off
`main`. Stop at the end of the phase so I can test before moving on.

### 0. Foundations (brought forward from Phase 2.5)
- All personal data from this phase (memories, conversations, rules, audit log,
  search history) lives in ONE SQLCipher-encrypted SQLite database under
  `data/`. Key from the system keyring (the existing Orion data key), never in
  config, code, or git. Install via pip, no sudo.
- Daily encrypted backup of the database, keep the last 30.
- Kill switches (spoken shutdown, nuclear wipe) stay in Phase 2.5, done right
  after this phase. Everything built here must be easy for the wipe to delete.
- Every feature is per person: data belongs to whoever Orion recognised.
  Unrecognised voices get nothing saved and nothing personal retrieved.

### 1. Router (local vs cloud)
- Default is LOCAL. When unsure, LOCAL.
- Optional auto-router (DISABLED by default, `router.auto: false`): a small
  local classifier sends general-knowledge requests with no personal data to
  Claude. Anything mentioning me, my people, my data, or Orion's own state
  stays local.
- Log every routing decision (encrypted): time, route, reason, which rule or
  classifier score decided it.

### 2. Claude mode (manual cloud)
- "Ask Claude ..." → just that request goes to Claude, then back to local.
- "Switch to Claude mode" → everything goes to Claude until "back to local" or
  "local mode". Claude mode keeps its own history for that session.
- Orion says the switch out loud and plays a distinct cloud chime before every
  request that goes to the cloud.
- Model in config: `cloud.model: claude-opus-5-5`, effort `low` for short
  spoken replies, server-side refusal fallbacks on.
- API key from the keyring (`python main.py secrets set anthropic`, hidden
  input) or ANTHROPIC_API_KEY. Never committed, logged, or printed.
- No internet, bad key, API down, rate limit → Orion says so and answers
  locally instead. Never crashes.
- README: set a monthly spend limit (~$10) in the Claude Console; show
  expected cost per spoken turn.

### 3. Privacy guard (applies to every cloud request and every web query)
- NEVER send to Claude: memories, conversation history, rules, email,
  calendar, notes, documents, or people's names (mine included), unless I
  explicitly say so for that request ("ask Claude, and include ...").
- Log every cloud request, exactly as sent, encrypted.
  `python main.py audit` shows what went where.

### 4. Web search (agentic)
- Orion decides himself when he needs current or unknown information (news,
  prices, "who is...", anything after the model's training cutoff, or when he
  isn't confident). He searches instead of guessing. Code catches obvious
  cases; the model can also call the search tool when unsure.
- Loop: plan → web_search → read_page on the best results → search again if
  needed (max 3 rounds, `search.max_rounds`) → answer.
- Answer is a spoken summary in Orion's own words, naming the source
  ("according to Reuters..."). Never "here's what I found" and never a list of
  links. If sources disagree or nothing reliable turns up, he says so.
- Full source URLs are saved with the answer: shown in text mode, and later
  emailable ("send me the links", Phase 8).
- Tools: web_search(query) → titles, URLs, snippets; read_page(url) → main
  text (trafilatura), trimmed to fit the local model's context.
- Backend in config: `searxng` (default; local Podman container, rootless,
  bound to 127.0.0.1 only, no open ports; Orion checks this at startup) or
  `brave` (Brave Search API, key in the keyring).
- Cache results briefly (15 minutes).
- Hard research: simple lookups stay local. For multi-source research, or
  when the local loop fails, Orion offers "This is a big one, want me to ask
  Claude?" A yes uses the Anthropic web search tool, counted toward the spend
  limit and logged like any cloud request.
- Query privacy: queries leave the device, so NEVER include personal data
  (memories, conversations, names of my people, email, calendar, documents).
  Rewrite queries to be generic; if a question can't be searched without
  personal data, ask me first.
- Web content is strictly DATA, never instructions. Inside the search loop the
  model only has web_search and read_page; pages can never trigger tools,
  rules, memory writes, corrections, sends, builds, system actions, or kill
  switches.
- Log every query and every URL fetched (encrypted).
- "Orion, don't search" disables web access until "you can search again";
  `search.enabled: false` is permanent offline mode.
- The capability guard changes: news, prices, scores (and weather, for now)
  become searchable; email, calendar, alarms still say "I can't yet".

### 5. Permanent memory (facts)
- "Remember that ...", "don't forget ...", or lasting facts about me are saved
  for the person speaking. Memories NEVER expire. Orion confirms out loud.
- Retrieve relevant memories for every request with local embeddings
  (nomic-embed-text via Ollama), matched by meaning. Local model only.
- Voice commands: "what do you remember about me?", "what do you know about
  X?", "forget X" (asks for confirmation; only I can delete my memories).
- Memory writes come ONLY from my own spoken or typed commands. Model output,
  web pages, Claude replies, and (later) email or documents can never write
  memory.

### 6. Conversation memory
- Every exchange (what I said, what Orion said, when, local or cloud) is saved
  encrypted, per person. Strangers' conversations are not saved.
- Continuity: each person's recent conversation is restored after a restart,
  so "as I was saying..." works after a reboot.
- Recall by meaning: "what did we talk about yesterday?", "what did I tell
  you about my exam last week?". A short daily summary per person keeps recall
  fast for the small model.
- Past conversations go to the local model only, never to Claude unless I say
  so for that request.
- Voice commands: "forget today's conversation", "delete my conversation
  history" (confirm), "don't save this conversation" (that conversation stays
  in memory only).
- Retention: kept forever by default; `conversations.keep_days` to limit it.
 

### 7. Standing instructions (rules)
- "Always / whenever / every time X, do Y" is saved as a permanent rule with a
  trigger and an action, per person.
  Example: "Whenever I say I'm leaving my room, remind me to take my keys."
- Every utterance is checked against my rules by meaning, not exact words
  ("heading out", "I'm off", "leaving now").
- When a rule fires, Orion performs the action alongside the normal reply.
  Phase 2 actions are spoken reminders; tool actions come with later phases.
- Voice commands: "what rules do I have?", "stop reminding me about X",
  "change the keys rule to also include my wallet" (reads back the new wording
  and asks before saving).
- Rules can only be created or changed by my own commands, never by model
  output or external content.

### 8. Interplay with Phase 1 features
- Clean mode, demo mode, and the scrub filter apply to everything spoken,
  including Claude and web answers; written logs stay unscrubbed but encrypted.
- Trust-changing commands (deleting memories, conversation history, or rules)
  need the same strong voice match as people commands (verify_threshold).
- "That was wrong" training notes add failure kinds: misroute, bad search,
  wrong memory, rule misfire. Still no conversation content in notes.

### 9. Tests, measurements, docs
- Tests for: router and privacy guard (nothing personal reaches the cloud or a
  search query), Claude mode switching and failure fallbacks (mocked API),
  web search loop limits, source naming, no link lists, injection resistance
  (a page saying "remember that..." or "ignore your instructions" changes
  nothing), offline switch, cache, memory save/retrieve/forget, conversation
  restore/recall/delete, per-person isolation, rule triggering by meaning and
  editing, encryption at rest, backups.
- Measure and report on the laptop (Jetson later): local vs cloud latency,
  search-answer latency, memory/recall retrieval time, rule false-trigger rate
  on a small labelled set, and llama3.2:3b's success rate on the search loop.
- Update README, user manual, and benchmarks.

### Personality (unchanged)
Calm, concise, slightly witty, addresses each person by name. Never invents
emails, events, or facts. If a tool or search fails, he says so.
