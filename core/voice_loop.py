"""The hands-free loop: wake word -> listen -> who's speaking? -> transcribe -> reply -> speak."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Callable, Iterable

import numpy as np

from audio.devices import Microphone
from audio.stt import SpeechToText
from audio.tts import Chimes, Speaker
from audio.vad import Endpointer, Status
from audio.wakeword import WakeWordDetector, strip_wake_phrase
from brain import profanity
from core.assistant import Assistant, sorry
from core import feedback
from core.barge_in import BargeIn
from core.dialog import Heard
from core.identity import Identity
from core.metrics import TurnMetrics
from core.people import People
from storage.training_notes import TrainingNotes

log = logging.getLogger(__name__)


class VoiceLoop:
    hears_voice = True   # also serves as the Dialog for onboarding
    # "Hey jarvis" is said in the same voice as the request that follows, so the
    # audio just before the wake word fired is added when identifying the speaker.
    # It turns a too-short "who am I?" (~0.6 s) into enough voice to judge.
    WAKE_AUDIO_S = 1.5

    def __init__(self, mic: Microphone, wake: WakeWordDetector, endpointer: Endpointer,
                 stt: SpeechToText, assistant: Assistant, speaker: Speaker,
                 chimes: Chimes | None, identity: Identity, people: People | None,
                 follow_up_s: float = 6.0, show: Callable[[str], None] = print,
                 notes: TrainingNotes | None = None, barge_in: BargeIn | None = None,
                 wake_phrases: list[str] | None = None) -> None:
        self.mic = mic
        self.wake = wake
        self.endpointer = endpointer
        self.stt = stt
        self.assistant = assistant
        self.speaker = speaker
        self.chimes = chimes
        self.identity = identity
        self.people = people
        self.follow_up_s = follow_up_s
        self.show = show
        self.notes = notes or TrainingNotes(None)
        self.barge_in = barge_in
        self.wake_phrases = wake_phrases or ["hey jarvis"]
        self.memory_commands = None          # core.memory.MemoryCommands, set by main
        self.conversation_commands = None    # core.conversations.ConversationCommands
        self.rule_commands = None            # core.rules.RuleCommands
        self.rules = None                    # core.rules.RuleBook (checked after each reply)
        self.router = None                   # core.router.Router (local vs Claude)
        self.audit = None                    # storage.audit.AuditLog
        self._prefill: list[np.ndarray] = []   # user speech captured by a barge-in
        frames = max(1, int(self.WAKE_AUDIO_S * 1000 / endpointer.chunk_ms))
        self._recent: deque[np.ndarray] = deque(maxlen=frames)
        self._wake_audio: np.ndarray | None = None   # memory only, dropped after one turn

    def _quiet(self) -> None:
        """Wait for Orion to finish talking, then forget whatever the mic heard meanwhile."""
        self.speaker.wait()
        self.mic.drain()
        self.wake.reset()

    def _chime(self, name: str) -> None:
        if self.chimes is not None:
            self.speaker.play(getattr(self.chimes, name))

    # ---- Dialog interface (used by onboarding) ----------------------------
    def say(self, text: str) -> None:
        self.speaker.say(text)
        self._quiet()

    def ask(self, prompt: str, timeout_s: float | None = None) -> Heard | None:
        self.speaker.say(prompt)
        return self.listen(timeout_s=timeout_s)

    # ---- listening ----------------------------------------------------------
    def listen(self, timeout_s: float | None = None, follow_up: bool = False,
               turn: TurnMetrics | None = None) -> Heard | None:
        """Record one utterance and transcribe it. None if nobody spoke or STT failed."""
        prefill, self._prefill = self._prefill, []
        self._heard_over_orion = bool(prefill)
        self.endpointer.reset(start_timeout_s=timeout_s)
        status = Status.WAITING
        if prefill:
            # Barge-in: the user is already mid-sentence. Keep what they said
            # while Orion was stopping, and don't chime over them.
            self.speaker.wait()
            self.wake.reset()
            for frame in prefill:
                status = self.endpointer.push(frame)
            self.show("[listening: you interrupted]")
        else:
            self._chime("follow_up" if follow_up else "listening")
            self._quiet()                       # don't record the chime or Orion himself
            self.show("[listening for a follow-up]" if follow_up else "[listening]")

        while status in (Status.WAITING, Status.SPEAKING):
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                log.warning("Microphone stopped delivering audio")
                status = Status.TIMEOUT
                break
            status = self.endpointer.push(frame)
        self._chime("done")
        if status is Status.TIMEOUT:
            return None

        if turn:
            turn.mark("speech_end")
            turn.extra["end_silence_ms"] = self.endpointer.end_frames * self.endpointer.chunk_ms
        audio = self.endpointer.audio()     # held in memory only, never written to disk
        try:
            text = self.stt.transcribe(audio, self.mic.sample_rate)
        except Exception:
            log.exception("Transcription failed")
            self.say(f"{sorry(self.identity.current)} My speech recognition just failed.")
            return None
        if turn:
            turn.mark("stt_done")
        if text:
            self.show(f"You: {text}")
        return Heard(text, audio, self.mic.sample_rate)

    # ---- main loop ----------------------------------------------------------
    def run(self, wake_phrase: str) -> None:
        self.mic.start()
        with self._sampling_echo():
            if self.people is not None:
                self.people.greet_on_start(self)
            else:
                self.say("Orion online.")
        self.show(f"Listening for '{wake_phrase}'. Ctrl+C to quit.")
        while True:
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                continue
            self._recent.append(frame)
            if not self.wake.detect(frame):
                continue
            log.debug("Wake word score %.2f", self.wake.last_score)
            self._wake_audio = np.concatenate(self._recent)
            self._recent.clear()
            if self.people is not None and self.people.store.is_empty():
                # Introductions were skipped at startup; nobody can own Orion yet.
                self.people.greet_on_start(self)
                self.show(f"[waiting for '{wake_phrase}']")
                continue
            # After each reply, keep listening briefly so a conversation doesn't
            # need the wake word on every line. Silence ends it.
            self.identity.start_conversation()
            follow_up = False
            while self.turn(follow_up) and self.follow_up_s > 0:
                follow_up = True
            self.assistant.forget_guest()
            self.show(f"[waiting for '{wake_phrase}']")

    def turn(self, follow_up: bool = False) -> bool:
        """Capture and answer one utterance. Returns True if the conversation continues."""
        turn = TurnMetrics(mode="voice")
        heard = self.listen(self.follow_up_s if follow_up else None, follow_up, turn)
        if heard is None:
            if not follow_up:
                self.show("[no speech]")
            return False

        id_audio = heard.audio
        if not follow_up and self._wake_audio is not None:
            id_audio = np.concatenate([self._wake_audio, heard.audio])
        self._wake_audio = None
        user = self.identity.observe(id_audio, heard.sample_rate, follow_up,
                                     unreliable=getattr(self, "_heard_over_orion", False))
        del id_audio
        if not follow_up and self.conversation_commands is not None:
            self.conversation_commands.log.start_conversation(user)
        turn.extra["utterance_s"] = round(heard.seconds, 2)
        turn.extra["follow_up"] = follow_up
        turn.extra["speaker_known"] = user is not None
        heard.audio = None  # done with the audio
        heard.text = strip_wake_phrase(heard.text, self.wake_phrases)

        if not heard.text:
            # Background noise in a follow-up window just closes it quietly.
            if not follow_up:
                self.say(f"{sorry(user)} I didn't catch that.")
            return not follow_up

        if getattr(self, "_heard_over_orion", False) and is_own_echo(heard.text,
                                                                     self.speaker.recent_text):
            # Barge-in fired on Orion's own voice from the speakers: drop it.
            log.info("Barge-in was Orion's own echo; ignoring %d words", len(heard.text.split()))
            turn.extra["barge_in_echo"] = True
            return True

        if is_dismissal(heard.text):
            self.say(f"Any time, {user.name}." if user else "Any time.")
            return False
        mode_reply = profanity.handle_command(heard.text, self.assistant.profanity)
        if mode_reply:
            self.say(mode_reply)
            return True
        # Before training-notes feedback: "that was wrong" right after a reminder corrects the rule.
        if self.rule_commands is not None and self.rule_commands.handle(heard, self, user):
            return True
        from core.web_search import handle_command as web_command

        web_reply = web_command(heard.text, self.assistant.web, show=lambda s: log.info("%s", s))
        if web_reply:
            self.say(web_reply)
            return True
        if feedback.handle(heard, self, user, self.notes, self.assistant.last_turn):
            return True
        if self.people is not None and self.people.handle(heard, self):
            return True
        if self.memory_commands is not None and self.memory_commands.handle(heard, self, user):
            return True
        if self.conversation_commands is not None and self.conversation_commands.handle(heard, self, user):
            return True

        decision = self._route(heard.text, user)
        if decision is not None and decision.handled:
            return True
        if self.barge_in is not None:
            self.barge_in.start()
        try:
            if decision is not None and decision.route == "cloud":
                self._chime("cloud")          # audible every time a request leaves the computer
                self.assistant.respond_cloud(decision.text, turn, user, decision,
                                             session=self.router.in_claude_mode(user.id if user else "guest"))
            else:
                self.assistant.respond(heard.text if decision is None else decision.text or heard.text,
                                       turn, self.identity.current)
        finally:
            interrupted = self.barge_in.stop() if self.barge_in is not None else None
        if not interrupted and self.rules is not None:
            for rule in self.rules.check(user, heard.text):
                turn.extra["rule_fired"] = True
                self.speaker.say(rule.spoken_action())
        if interrupted:
            turn.extra["barge_in"] = interrupted
            self.speaker.wait()
            self.speaker.resume()
            self._prefill = self.barge_in.prefill
            # A wake-word interrupt starts a fresh request; a voice interrupt
            # continues straight from what the user is saying.
            return True
        self._quiet()
        return True

    def _route(self, text: str, user):
        """Local or Claude for this request; mode switches are announced out loud."""
        if self.router is None:
            return None
        key = user.id if user else "guest"
        was_cloud = self.router.in_claude_mode(key)
        decision = self.router.decide(text, key)
        if self.audit is not None:
            self.audit.record("route", route=decision.route, reason=decision.reason,
                              person=user.id if user else None)
        if was_cloud and not self.router.in_claude_mode(key):
            self.assistant.end_cloud_session(user)
        if decision.announce:
            if decision.handled and decision.route == "cloud":
                self._chime("cloud")
            self.say(decision.announce)
        return decision

    def _sampling_echo(self):
        """While Orion talks at startup, sample how his voice sounds through this
        room's speakers and mic, so barge-in can tell his echo from a person."""
        import contextlib

        @contextlib.contextmanager
        def cm():
            bi = self.barge_in
            if bi is None or bi.mode != "voice" or bi.encoder is None:
                yield
                return
            frames: list[np.ndarray] = []
            tap = lambda f: frames.append(f) if self.speaker.speaking else None  # noqa: E731
            self.mic.taps.append(tap)
            try:
                yield
            finally:
                self.mic.taps.remove(tap)
            n = len(frames)
            # Embed one-second windows of echo; only Orion was talking when these were captured.
            step = max(1, bi.window)
            for i in range(0, n - step + 1, step):
                bi.own_voice.add(bi.encoder.embed(np.concatenate(frames[i:i + step])))
            log.info("Barge-in: sampled %.1fs of Orion's echo (%d voiceprints)",
                     n * self.endpointer.chunk_ms / 1000, len(bi.own_voice.prints))
        return cm()


# "Thanks, Orion, that's all" -> drop politeness fillers -> "that's all" -> dismissal.
_FILLERS = {"thanks", "thank", "you", "orion", "ok", "okay", "great", "cool", "perfect",
            "alright", "cheers", "no", "nope", "and", "so", "for", "now"}
_CLOSERS = {"", "that's all", "that is all", "thats all", "that's it", "thats it", "never mind",
            "nevermind", "nothing", "nothing else", "stop", "cancel", "goodbye", "bye",
            "i'm done", "im done", "all done", "i'm good", "im good", "that'll do"}


def is_own_echo(heard: str, recent: "Iterable[str]", min_overlap: float = 0.6) -> bool:
    """True if most words heard are words Orion just said ("your feeling" while he
    was saying "...you're feeling frustrated"): the mic picked up his own voice."""
    words = _echo_words(heard)
    if not words:
        return False
    said = set(_echo_words(" ".join(recent)))
    hits = sum(1 for w in words if w in said)
    return hits / len(words) >= min_overlap


def _echo_words(text: str) -> list[str]:
    """Words normalised the way Whisper tends to mishear them from the speakers:
    "you're" / "your" / "you" alike, contractions joined, plural "s" dropped."""
    text = re.sub(r"\byou'?re\b|\byour\b", "you", text.lower())
    words = re.findall(r"[a-z']+", text)
    return [w.replace("'", "").rstrip("s") or w for w in words]


def is_dismissal(text: str) -> bool:
    """A short 'I'm done' reply that should end the conversation, not go to the LLM."""
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    if not words or len(words) > 8:
        return False
    rest = " ".join(w for w in words if w not in _FILLERS)
    return rest in _CLOSERS
