#!/usr/bin/env python3
"""ORION: Offline Reasoning and Intelligent Operations Node.

    python main.py                 # voice mode (wake word)
    python main.py --text          # type instead of talk; replies still spoken
    python main.py --text --no-tts # pure text, for debugging the brain
    python main.py --list-devices
    python main.py users [list | rename OLD NEW | delete NAME | relearn NAME]
    python main.py notes [list | clear]
    python main.py secrets [set anthropic | status]
    python main.py audit [N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from core.config import ConfigError, load_config

log = logging.getLogger("orion")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="orion", description="Private, local-first voice assistant.")
    p.add_argument("--text", action="store_true", help="text mode: type instead of speaking")
    p.add_argument("--no-tts", action="store_true", help="don't speak replies (text mode only)")
    p.add_argument("--profile", help="hardware profile from config.yaml (default: auto-detect)")
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--demo", action="store_true",
                   help="demo mode for IT demos / interviews: clean language, demo overrides")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    # REMAINDER: everything after the command word is passed through, so subcommand
    # options like "secrets set anthropic --clipboard" aren't mistaken for Orion's own.
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="optional: users [list | rename OLD NEW | delete NAME | relearn NAME], "
                        "notes [list | clear]")
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    # Console only for now. Logs never include transcripts; file logging arrives
    # with encryption in Phase 2.5.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # phonemizer (used by Kokoro) warns "words count mismatch" whenever espeak splits
    # or merges words (contractions, numbers). It is harmless. A filter is used,
    # not a level, because phonemizer resets its logger's level on every backend load.
    logging.getLogger("phonemizer").addFilter(
        lambda record: "words count mismatch" not in record.getMessage())


def timed(label: str):
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if exc[0] is None:
                log.info("%s ready in %.1fs", label, time.perf_counter() - self.t)

    return _T()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if args.list_devices:
        from audio.devices import list_devices

        print(list_devices())
        return 0

    try:
        demo = args.demo or os.environ.get("ORION_DEMO", "") not in ("", "0")
        cfg = load_config(args.config, profile=args.profile, demo=demo)
    except (ConfigError, OSError) as exc:
        log.error("Config error: %s", exc)
        return 2

    from audio.devices import AudioDeviceError
    from audio.tts import AudioPlayer, Chimes, Speaker, make_tts
    from brain.llm import LLMError, make_llm
    from brain.profanity import DEMO, ProfanityPolicy
    from core.assistant import Assistant
    from core.metrics import MetricsSink, memory_snapshot
    from core.net_guard import NetworkPolicyError, exposed_listeners, port_of

    relearn = None
    if args.command[:2] == ["users", "relearn"]:
        # Voice learning started from the keyboard: the recovery path when Orion no
        # longer recognises someone (even the owner), since by voice he can't tell
        # who's asking. Whoever is at the keyboard is trusted, as in text mode.
        if len(args.command) != 3:
            log.error("Usage: python main.py users relearn NAME")
            return 2
        relearn = args.command[2]
    elif args.command:
        if args.command[0] == "users":
            return users_cli(cfg, args.command[1:])
        if args.command[0] == "notes":
            return notes_cli(cfg, args.command[1:])
        if args.command[0] == "secrets":
            return secrets_cli(args.command[1:])
        if args.command[0] == "audit":
            return audit_cli(cfg, args.command[1:])
        log.error("Unknown command %r. Try: users list, notes list", args.command[0])
        return 2

    models_dir = cfg.path("paths.models")
    log.info("Profile: %s", cfg.profile)

    llm_cfg = cfg.section("llm")
    try:
        llm = make_llm(llm_cfg)
    except NetworkPolicyError as exc:
        log.error("%s", exc)
        return 2
    exposed = exposed_listeners(port_of(llm_cfg["host"]))
    if exposed:
        log.error("Ollama is listening on %s, which exposes it to the network. "
                  "Unset OLLAMA_HOST (or set it to 127.0.0.1) and restart Ollama.", exposed)
        return 2

    want_tts = not (args.text and args.no_tts)
    tts = player = chimes = None
    try:
        if want_tts:
            with timed(f"TTS ({cfg.get('tts.backend')})"):
                tts = make_tts(cfg.section("tts"), models_dir)
            player = AudioPlayer(tts.sample_rate, cfg.get("audio.output_device"))
            if cfg.get("audio.chimes", True) and not args.text:
                chimes = Chimes(tts.sample_rate)
    except (FileNotFoundError, AudioDeviceError) as exc:
        log.error("%s", exc)
        return 1

    name = cfg.get("assistant.name", "Orion")
    speaker = Speaker(tts, player, echo=lambda s: print(f"{name}: {s}", flush=True))
    metrics = MetricsSink(cfg.path("metrics.file"), enabled=cfg.get("metrics.enabled", True))
    policy = ProfanityPolicy(cfg.get("assistant.profanity", "on"))
    if cfg.demo:
        policy.keep_clean(DEMO)
        log.info("Demo mode: clean language")
    store = open_user_store(cfg)
    notes = open_training_notes(cfg)
    knowledge = open_knowledge(cfg, llm, store)
    assistant = Assistant(llm, speaker, metrics, cfg.profile, name,
                          max_history_turns=cfg.get("llm.max_history_turns", 8), profanity=policy,
                          memory=knowledge.memory, conversations=knowledge.log)
    router = make_router(cfg, assistant, knowledge)
    assistant.web = make_web_searcher(cfg, llm, store, knowledge, router)

    try:
        with timed(f"LLM {llm_cfg['model']}"):
            llm.warm_up()
        if hasattr(llm, "loaded_models"):
            log.info("Ollama models in memory: %s", llm.loaded_models())
    except LLMError as exc:
        log.warning("LLM warm-up failed: %s", exc)
        speaker.say(f"{exc.spoken} I'll keep trying when you speak to me.")

    try:
        if relearn:
            return run_voice(cfg, assistant, speaker, chimes, models_dir, store, notes,
                             relearn=relearn, knowledge=knowledge, router=router)
        if args.text:
            log.info("Memory: %s", memory_snapshot())
            run_text(cfg, assistant, speaker, store, notes, knowledge, router)
        else:
            return run_voice(cfg, assistant, speaker, chimes, models_dir, store, notes,
                             knowledge=knowledge, router=router)
    except KeyboardInterrupt:
        print()
    finally:
        speaker.close()
        if knowledge.log is not None:
            knowledge.log.close()        # finish indexing the last exchanges
        if knowledge.db is not None:
            knowledge.db.close()
    return 0


def open_user_store(cfg):
    """Encrypted user store, or a memory-only one if the keyring is unavailable."""
    from security.keystore import KeyStoreError, get_data_key
    from storage.encrypted_file import EncryptedJSONFile
    from storage.users import UserStore

    try:
        key = get_data_key()
    except KeyStoreError as exc:
        log.warning("%s. People will be remembered for this session only.", exc)
        return UserStore(None)
    return UserStore(EncryptedJSONFile(cfg.path("paths.data") / "users.enc", key, "users"))


def open_training_notes(cfg, force: bool = False):
    """Opt-in (training_notes.enabled). Encrypted like everything personal."""
    from security.keystore import KeyStoreError, get_data_key
    from storage.encrypted_file import EncryptedJSONFile
    from storage.training_notes import TrainingNotes

    enabled = bool(cfg.get("training_notes.enabled", False))
    if not (enabled or force):
        return TrainingNotes(None)
    try:
        key = get_data_key()
    except KeyStoreError as exc:
        log.warning("%s. Training notes disabled.", exc)
        return TrainingNotes(None)
    path = cfg.path("paths.data") / "training_notes.enc"
    return TrainingNotes(EncryptedJSONFile(path, key, "training-notes"), enabled=enabled)


def notes_cli(cfg, args: list[str]) -> int:
    from datetime import datetime

    notes = open_training_notes(cfg, force=True)
    if notes.file is None:
        log.error("Keyring unavailable; can't open training notes.")
        return 1
    action = args[0] if args else "list"
    if action == "list":
        entries = notes.all()
        if not entries:
            state = "on" if notes.enabled else "off (training_notes.enabled in config.yaml)"
            print(f"No training notes. Logging is {state}.")
        for e in entries:
            ctx = " ".join(f"{k}={v}" for k, v in e.get("context", {}).items())
            print(f"{datetime.fromtimestamp(e['ts']):%Y-%m-%d %H:%M}  {e['kind']:<13} "
                  f"{e.get('note', '-')}\n                  {ctx}")
        return 0
    if action == "clear":
        if input("Delete all training notes? Type yes to confirm: ").strip().lower() != "yes":
            print("Cancelled.")
            return 1
        print(f"Deleted {notes.clear()} notes.")
        return 0
    print("Usage: python main.py notes [list | clear]")
    return 2


class Knowledge:
    """Phase 2 stores: the encrypted database, memories, and conversation log."""

    def __init__(self, db=None, memory=None, log=None, rules=None):
        self.db, self.memory, self.log, self.rules = db, memory, log, rules

    def commands(self, verified):
        """(memory, conversation, rule) command handlers, or Nones when unavailable."""
        from core.conversations import ConversationCommands
        from core.memory import MemoryCommands
        from core.rules import RuleCommands

        if self.db is None:
            return None, None, None
        convo_cmds = ConversationCommands(self.log, verified) if self.log else None
        mem_cmds = (MemoryCommands(self.memory, verified,
                                   recall_conversation=convo_cmds.recall_topic if convo_cmds else None)
                    if self.memory else None)
        rule_cmds = RuleCommands(self.rules, verified) if self.rules else None
        return mem_cmds, convo_cmds, rule_cmds


def open_knowledge(cfg, llm, store) -> Knowledge:
    """Open the encrypted database and the memory / conversation stores.

    Without the keyring nothing personal can be stored safely, so Orion runs
    without memory rather than writing anything unencrypted.
    """
    from brain.embeddings import OllamaEmbedder
    from core.conversations import ConversationLog, make_summarizer
    from core.memory import MemoryStore
    from core.rules import RuleBook, make_paraphraser
    from security.keystore import KeyStoreError, get_data_key
    from storage.db import DatabaseError, open_database

    if not (cfg.get("memory.enabled", True) or cfg.get("conversations.enabled", True)):
        return Knowledge()
    try:
        db = open_database(cfg.path("paths.data"), get_data_key())
    except (KeyStoreError, DatabaseError) as exc:
        log.warning("%s. Memory and conversation history are off this session.", exc)
        return Knowledge()
    embedder = OllamaEmbedder(cfg.get("llm.host"), cfg.get("embeddings.model", "nomic-embed-text"))
    memory = MemoryStore(db, embedder) if cfg.get("memory.enabled", True) else None
    convo = None
    if cfg.get("conversations.enabled", True):
        convo = ConversationLog(db, embedder, summarize=make_summarizer(llm))
        removed = convo.apply_retention(cfg.get("conversations.keep_days"))
        if removed:
            log.info("Retention: removed %d old messages", removed)
        convo.summarize_past_days(store.all())
    rules = (RuleBook(db, embedder, paraphrase=make_paraphraser(cfg.get("llm.host"), cfg.get("llm.model")))
             if cfg.get("rules.enabled", True) else None)
    try:
        db.backup(cfg.path("paths.data") / "backups", keep=cfg.get("backups.keep", 30))
    except Exception:
        log.exception("Daily backup failed")
    return Knowledge(db, memory, convo, rules)


def make_router(cfg, assistant, knowledge):
    """Claude mode needs an API key AND the encrypted audit log; otherwise local only."""
    from brain.claude import ClaudeBackend, api_key
    from core.router import Router
    from storage.audit import AuditLog

    audit = AuditLog(knowledge.db) if knowledge.db is not None else None
    have_key = bool(cfg.get("cloud.enabled", True) and api_key())
    if have_key and audit is not None:
        assistant.cloud = ClaudeBackend.from_config(cfg.section("cloud"))
        assistant.audit = audit
        log.info("Claude mode available (%s)", assistant.cloud.model)
    elif cfg.get("cloud.enabled", True):
        log.info("Claude mode unavailable: %s", "no API key" if audit is not None else "no encrypted audit log")
    router = Router(auto=bool(cfg.get("router.auto", False)), cloud_available=assistant.cloud is not None)
    router.audit = audit
    return router


def make_web_searcher(cfg, llm, store, knowledge, router):
    """SearXNG on loopback (checked) or Brave; None when search is off or unreachable."""
    from brain.web import BraveBackend, CachedSearch, PageReader, SearxngBackend
    from core.net_guard import exposed_listeners, port_of
    from core.web_search import PrivacyFilter, WebSearcher, memory_terms_from

    s = cfg.section("search")
    if not s.get("enabled", True):
        log.info("Web search: off (search.enabled: false)")
        return None
    if s.get("backend", "searxng") == "brave":
        import keyring

        key = (keyring.get_password("orion", "brave-api-key") or "").strip()
        if not key:
            log.warning("Web search: Brave selected but no key (python main.py secrets set brave)")
            return None
        backend = BraveBackend(key)
    else:
        url = s.get("searxng_url", "http://127.0.0.1:8888")
        exposed = exposed_listeners(port_of(url))
        if exposed:
            log.error("SearXNG is reachable from the network on %s. Not using it (dorm rule); "
                      "restart it with scripts/searxng.sh start.", exposed)
            return None
        backend = SearxngBackend(url)
        try:
            backend.search("test", n=1)
        except Exception:
            log.warning("Web search: SearXNG not reachable at %s (start it: scripts/searxng.sh start)", url)
            return None
    audit = getattr(router, "audit", None)
    privacy = PrivacyFilter(lambda: [u.name for u in store.all()], memory_terms_from(knowledge.memory))
    web = WebSearcher(CachedSearch(backend, ttl_s=s.get("cache_s", 900)), PageReader(max_chars=s.get("page_chars", 2500)),
                      llm, privacy, audit=audit, max_rounds=s.get("max_rounds", 3))
    log.info("Web search: %s", backend.name)
    return web


def secrets_cli(args: list[str]) -> int:
    """Store API keys in the system keyring with hidden input. Never printed."""
    import getpass

    import keyring

    names = {"anthropic": "anthropic-api-key", "brave": "brave-api-key"}
    prefixes = {"anthropic": "sk-ant-", "brave": ""}
    if args[:1] == ["status"] or not args:
        for label, name in names.items():
            try:
                value = (keyring.get_password("orion", name) or "").strip()
            except Exception:
                value = ""
            if not value:
                state = "not set"
            elif prefixes[label] and not value.startswith(prefixes[label]):
                state = f"saved, but it doesn't look like a {label} key (should start {prefixes[label]}...)"
            else:
                state = f"saved ({len(value)} characters)"
            print(f"  {label:<10} {state}")
        return 0
    if args[0] == "set" and len(args) in (2, 3) and args[1] in names:
        if args[2:] == ["--clipboard"]:
            # Read straight from the clipboard: no hidden-prompt paste to go wrong.
            import shutil
            import subprocess

            tool = (["wl-paste", "--no-newline"] if shutil.which("wl-paste")
                    else ["xclip", "-selection", "clipboard", "-o"] if shutil.which("xclip") else None)
            if tool is None:
                print("No clipboard tool found (install wl-clipboard or xclip).")
                return 1
            value = subprocess.run(tool, capture_output=True, text=True).stdout.strip()
        else:
            print("Paste the key and press Enter. It stays hidden (in most terminals paste is Ctrl+Shift+V).")
            print("Tip: copy the key, then run: python main.py secrets set anthropic --clipboard")
            value = getpass.getpass(f"{args[1]} key (hidden): ").strip()
        if not value:
            print("Nothing entered; unchanged.")
            return 1
        if args[1] == "anthropic" and not value.startswith("sk-ant-"):
            print("That doesn't look like an Anthropic API key (they start with sk-ant-). Not saved.")
            return 1
        keyring.set_password("orion", names[args[1]], value)
        print(f"Saved to the system keyring ({len(value)} characters).")
        if args[2:] == ["--clipboard"]:
            import shutil
            import subprocess

            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy", "--clear"], check=False)
                print("Clipboard cleared.")
        return 0
    print("Usage: python main.py secrets [status | set anthropic [--clipboard] | set brave [--clipboard]]")
    return 2


def audit_cli(cfg, args: list[str]) -> int:
    """What went where: routing decisions and cloud requests, newest first."""
    from datetime import datetime

    from security.keystore import KeyStoreError, get_data_key
    from storage.audit import AuditLog
    from storage.db import DatabaseError, open_database

    try:
        db = open_database(cfg.path("paths.data"), get_data_key(create=False))
    except (KeyStoreError, DatabaseError) as exc:
        print(f"Can't open the audit log: {exc}")
        return 1
    audit = AuditLog(db)
    limit = int(args[0]) if args and args[0].isdigit() else 30
    n, cost = audit.month_cost()
    print(f"This month: {n} cloud requests, about ${cost:.2f} (check the Claude Console for actual billing)\n")
    for ts, kind, d in audit.recent(limit):
        when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        if kind == "cloud_request":
            body = d.get("body", {})
            last = (body.get("messages") or [{}])[-1].get("content", "")
            usage = d.get("usage") or {}
            print(f"{when}  CLOUD  {body.get('model')}  personal data included: {d.get('included_personal')}"
                  f"  tokens {usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}"
                  f"  ${usage.get('cost_usd', 0):.4f}{'  ERROR' if d.get('error') else ''}\n"
                  f"                sent: {str(last)[:120]!r}")
        elif kind == "route":
            print(f"{when}  route  {d.get('route'):<5}  {d.get('reason')}")
        else:
            print(f"{when}  {kind}  {d}")
    return 0


def make_identity(cfg, store, encoder):
    from core.identity import Identity

    p = cfg.section("people")
    return Identity(store, encoder, threshold=p.get("threshold", 0.40), margin=p.get("margin", 0.05),
                    min_audio_s=p.get("min_audio_s", 0.8),
                    continuity_slack=p.get("continuity_slack", 0.1),
                    adapt_threshold=p.get("adapt_threshold", 0.72),
                    adapt_min_audio_s=p.get("adapt_min_audio_s", 2.0))


def run_text(cfg, assistant, speaker, store, notes, knowledge=None, router=None) -> None:
    """Text mode is the owner's terminal, so it acts as the owner."""
    from core.dialog import TextDialog
    from core.metrics import TurnMetrics
    from core.people import People

    identity = make_identity(cfg, store, encoder=None)
    people = People(store, identity)
    dialog = TextDialog(speaker)
    print("Text mode. Commands: /reset (forget this chat), /stats (last turn), /quit")
    people.greet_on_start(dialog)
    identity.set_current(store.owner())
    knowledge = knowledge or Knowledge()
    mem_cmds, convo_cmds, rule_cmds = knowledge.commands(people._verified)
    from core.dialog import yes_no
    from core.web_search import handle_command as web_command

    def confirm(prompt):
        answer = dialog.ask(prompt)
        return answer is not None and yes_no(answer.text) is True

    assistant.confirm = confirm
    if convo_cmds is not None:
        convo_cmds.log.start_conversation(identity.current)
    while True:
        try:
            line = input("You: ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/reset":
            assistant.conversation_for(identity.current).clear()
            print("(conversation cleared)")
            continue
        if line == "/stats":
            print(assistant.last_metrics or "(no turns yet)")
            continue
        from brain.profanity import handle_command
        from core.dialog import Heard

        from core import feedback

        if rule_cmds is not None and rule_cmds.handle(Heard(line), dialog, identity.current):
            continue
        web_reply = web_command(line, assistant.web)
        if web_reply:
            dialog.say(web_reply)
            continue
        if feedback.handle(Heard(line), dialog, identity.current, notes, assistant.last_turn):
            continue
        mode_reply = handle_command(line, assistant.profanity)
        if mode_reply:
            dialog.say(mode_reply)
            continue
        if people.handle(Heard(line), dialog):
            continue
        if mem_cmds is not None and mem_cmds.handle(Heard(line), dialog, identity.current):
            continue
        if convo_cmds is not None and convo_cmds.handle(Heard(line), dialog, identity.current):
            continue
        decision = router.decide(line, identity.current.id if identity.current else "guest") if router else None
        if decision is not None:
            if router.audit is not None:
                router.audit.record("route", route=decision.route, reason=decision.reason,
                                    person=identity.current.id if identity.current else None)
            if not router.in_claude_mode(identity.current.id if identity.current else "guest"):
                assistant.end_cloud_session(identity.current)
            if decision.announce:
                dialog.say(decision.announce)
            if decision.handled:
                continue
        if decision is not None and decision.route == "cloud":
            print("[-> Claude]")
            assistant.respond_cloud(decision.text, TurnMetrics(mode="text"), identity.current, decision,
                                    session=router.in_claude_mode(identity.current.id if identity.current else "guest"))
        else:
            assistant.respond(decision.text if decision and decision.text else line,
                              TurnMetrics(mode="text"), identity.current)
        if knowledge.rules is not None:
            for rule in knowledge.rules.check(identity.current, line):
                dialog.say(rule.spoken_action())


def users_cli(cfg, args: list[str]) -> int:
    """List, rename, or delete people from the terminal (deletion asks first)."""
    store = open_user_store(cfg)
    if not store.persistent:
        log.error("Keyring unavailable; can't open the user store.")
        return 1
    action = args[0] if args else "list"
    if action == "list":
        users = store.all()
        if not users:
            print("No users yet. Start Orion to meet the owner.")
        for u in users:
            voice = f"voiceprint ({len(u.enrollment)}+{len(u.adaptive)} samples)" if u.has_voice else "no voiceprint"
            print(f"  {u.name:<20} {u.role:<7} {voice}")
        return 0
    if action == "rename" and len(args) == 3:
        u = store.find(args[1])
        if u is None:
            print(f"No user called {args[1]!r}.")
            return 1
        store.rename(u.id, args[2])
        print(f"Renamed {args[1]} to {args[2]}.")
        return 0
    if action == "delete" and len(args) == 2:
        u = store.find(args[1])
        if u is None:
            print(f"No user called {args[1]!r}.")
            return 1
        if input(f"Delete {u.name} ({u.role}) and their voiceprint? Type the name to confirm: ").strip() != u.name:
            print("Cancelled.")
            return 1
        store.delete(u.id)
        print(f"Deleted {u.name}.")
        return 0
    print("Usage: python main.py users [list | rename OLD NEW | delete NAME]")
    return 2


def run_voice(cfg, assistant, speaker, chimes, models_dir, store, notes,
              relearn: str | None = None, knowledge=None, router=None) -> int:
    from audio.devices import AudioDeviceError, Microphone
    from audio.stt import make_stt
    from audio.vad import Endpointer, SileroVAD
    from audio.wakeword import WakeWordDetector, model_specs, wakeword_model_path
    from audio.speaker_id import SherpaSpeakerEncoder
    from core.metrics import memory_snapshot
    from core.people import People
    from core.voice_loop import VoiceLoop

    audio, vad = cfg.section("audio"), cfg.section("vad")
    ww = cfg.section("wakeword")
    try:
        with timed("Wake word"):
            specs = model_specs(ww)
            wake = WakeWordDetector(
                [wakeword_model_path(s, models_dir, cfg.root) for s in specs], models_dir,
                threshold=ww.get("threshold", 0.5), cooldown_s=ww.get("cooldown_s", 1.5),
                thresholds=ww.get("thresholds"),
            )
        endpointer = Endpointer(
            SileroVAD(models_dir / "openwakeword" / "silero_vad.onnx"),
            chunk_ms=audio["chunk_ms"],
            threshold=vad.get("threshold", 0.5),
            start_timeout_s=vad.get("start_timeout_s", 6.0),
            end_silence_ms=vad.get("end_silence_ms", 800),
            max_utterance_s=vad.get("max_utterance_s", 20.0),
            pre_roll_ms=vad.get("pre_roll_ms", 240),
            neg_threshold=vad.get("neg_threshold"),
        )
        with timed("Whisper STT"):
            stt = make_stt(cfg.section("stt"), models_dir)
            if hasattr(stt, "hotwords"):
                extra = cfg.get("stt.hotwords") or []
                stt.hotwords = lambda: ", ".join(dict.fromkeys(
                    [cfg.get("assistant.name", "Orion"), "Claude", *extra, *(u.name for u in store.all())]))
            stt.warm_up()
        people_cfg = cfg.section("people")
        encoder = None
        if people_cfg.get("enabled", True):
            with timed("Speaker recognition"):
                encoder = SherpaSpeakerEncoder(models_dir / "speaker" / people_cfg["model"],
                                               threads=people_cfg.get("threads", 2))
        mic = Microphone(audio["sample_rate"], audio["chunk_ms"], audio.get("input_device"))
    except (FileNotFoundError, AudioDeviceError) as exc:
        log.error("%s", exc)
        return 1

    if relearn:
        user = store.find(relearn)
        if user is None:
            log.error("No user called %r. See: python main.py users list", relearn)
            return 1
        if encoder is None:
            log.error("Voice recognition is disabled (people.enabled: false)")
            return 1

    log.info("Memory: %s", memory_snapshot())
    # Phrases the models listen for, e.g. "hey jarvis", "hey orion"; the first (or
    # wakeword.phrase) is the one Orion tells people to say.
    heard_phrases = [Path(s).stem.split("_v")[0].replace("_", " ") for s in specs]
    phrase = ww.get("phrase") or heard_phrases[0]
    barge = None
    conv = cfg.section("conversation")
    mode = conv.get("barge_in", "wake_word")
    if mode != "off":
        from core.barge_in import BargeIn, learn_own_voice

        own = None
        if mode == "voice" and encoder is not None and speaker.tts is not None:
            with timed("Barge-in voiceprint"):
                own = learn_own_voice(speaker.tts, encoder, [
                    "Good morning. Here's what's on today, and a quick look at the weather.",
                    "That's a cracking question. The short answer is yes, but there's a catch.",
                ])
        barge = BargeIn(mic, wake, endpointer.score, speaker, mode=mode, encoder=encoder,
                        own_voice=own, chunk_ms=audio["chunk_ms"],
                        echo_threshold=conv.get("barge_in_echo_threshold", 0.5),
                        speech_ms=conv.get("barge_in_speech_ms", 400))
        log.info("Barge-in: %s", barge.mode)

    identity = make_identity(cfg, store, encoder)
    people = People(store, identity, enroll_target_s=people_cfg.get("enroll_target_s", 10.0),
                    verify_threshold=people_cfg.get("verify_threshold", 0.55),
                    wake_phrase=phrase) if encoder else None
    loop = VoiceLoop(mic, wake, endpointer, stt, assistant, speaker, chimes, identity, people,
                     follow_up_s=cfg.get("conversation.follow_up_s", 6.0), notes=notes,
                     barge_in=barge, wake_phrases=heard_phrases)
    if people is not None:
        knowledge = knowledge or Knowledge()
        loop.memory_commands, loop.conversation_commands, loop.rule_commands = knowledge.commands(people._verified)
        loop.rules = knowledge.rules
    loop.router = router
    loop.audit = getattr(router, "audit", None)
    from core.dialog import yes_no

    def confirm(prompt):
        answer = loop.ask(prompt, timeout_s=10.0)
        return answer is not None and yes_no(answer.text) is True

    assistant.confirm = confirm
    try:
        if relearn:
            mic.start()
            speaker.say(f"Hello, {user.name}. Let's relearn your voice.")
            return 0 if people.enroll_voice(loop, user) else 1
        loop.run(phrase)
    except AudioDeviceError as exc:
        log.error("%s", exc)
        speaker.say("Sorry, I can't open the microphone.")
        speaker.wait()
        return 1
    finally:
        mic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
