#!/usr/bin/env python3
"""ORION: Offline Reasoning and Intelligent Operations Node.

    python main.py                 # voice mode (wake word)
    python main.py --text          # type instead of talk; replies still spoken
    python main.py --text --no-tts # pure text, for debugging the brain
    python main.py --list-devices
    python main.py users [list | rename OLD NEW | delete NAME]
"""

from __future__ import annotations

import argparse
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
    p.add_argument("command", nargs="*", help="optional: users [list | rename OLD NEW | delete NAME]")
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
    # or merges words (contractions, numbers). It is harmless.
    logging.getLogger("phonemizer").setLevel(logging.ERROR)


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

    if args.command:
        if args.command[0] != "users":
            log.error("Unknown command %r. Try: users list", args.command[0])
            return 2
        return users_cli(cfg, args.command[1:])

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
    assistant = Assistant(llm, speaker, metrics, cfg.profile, name,
                          max_history_turns=cfg.get("llm.max_history_turns", 8), profanity=policy)
    store = open_user_store(cfg)

    try:
        with timed(f"LLM {llm_cfg['model']}"):
            llm.warm_up()
        if hasattr(llm, "loaded_models"):
            log.info("Ollama models in memory: %s", llm.loaded_models())
    except LLMError as exc:
        log.warning("LLM warm-up failed: %s", exc)
        speaker.say(f"{exc.spoken} I'll keep trying when you speak to me.")

    try:
        if args.text:
            log.info("Memory: %s", memory_snapshot())
            run_text(cfg, assistant, speaker, store)
        else:
            return run_voice(cfg, assistant, speaker, chimes, models_dir, store)
    except KeyboardInterrupt:
        print()
    finally:
        speaker.close()
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


def make_identity(cfg, store, encoder):
    from core.identity import Identity

    p = cfg.section("people")
    return Identity(store, encoder, threshold=p.get("threshold", 0.55), margin=p.get("margin", 0.05),
                    min_audio_s=p.get("min_audio_s", 0.8),
                    adapt_threshold=p.get("adapt_threshold", 0.72))


def run_text(cfg, assistant, speaker, store) -> None:
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

        mode_reply = handle_command(line, assistant.profanity)
        if mode_reply:
            dialog.say(mode_reply)
            continue
        if people.handle(Heard(line), dialog):
            continue
        assistant.respond(line, TurnMetrics(mode="text"), identity.current)


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


def run_voice(cfg, assistant, speaker, chimes, models_dir, store) -> int:
    from audio.devices import AudioDeviceError, Microphone
    from audio.stt import make_stt
    from audio.vad import Endpointer, SileroVAD
    from audio.wakeword import WakeWordDetector, wakeword_model_path
    from audio.speaker_id import SherpaSpeakerEncoder
    from core.metrics import memory_snapshot
    from core.people import People
    from core.voice_loop import VoiceLoop

    audio, vad = cfg.section("audio"), cfg.section("vad")
    ww = cfg.section("wakeword")
    try:
        with timed("Wake word"):
            wake = WakeWordDetector(
                wakeword_model_path(ww["model"], models_dir, cfg.root), models_dir,
                threshold=ww.get("threshold", 0.5), cooldown_s=ww.get("cooldown_s", 1.5),
            )
        endpointer = Endpointer(
            SileroVAD(models_dir / "openwakeword" / "silero_vad.onnx"),
            chunk_ms=audio["chunk_ms"],
            threshold=vad.get("threshold", 0.5),
            start_timeout_s=vad.get("start_timeout_s", 6.0),
            end_silence_ms=vad.get("end_silence_ms", 800),
            max_utterance_s=vad.get("max_utterance_s", 20.0),
            pre_roll_ms=vad.get("pre_roll_ms", 240),
        )
        with timed("Whisper STT"):
            stt = make_stt(cfg.section("stt"), models_dir)
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

    log.info("Memory: %s", memory_snapshot())
    phrase = ww["model"].replace("_", " ") if not str(ww["model"]).endswith(".onnx") else "the wake word"
    identity = make_identity(cfg, store, encoder)
    people = People(store, identity, enroll_target_s=people_cfg.get("enroll_target_s", 10.0)) if encoder else None
    loop = VoiceLoop(mic, wake, endpointer, stt, assistant, speaker, chimes, identity, people,
                     follow_up_s=cfg.get("conversation.follow_up_s", 6.0))
    try:
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
