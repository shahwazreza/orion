"""The hands-free loop: wake word -> listen -> who's speaking? -> transcribe -> reply -> speak."""

from __future__ import annotations

import logging
import re
from typing import Callable

from audio.devices import Microphone
from audio.stt import SpeechToText
from audio.tts import Chimes, Speaker
from audio.vad import Endpointer, Status
from audio.wakeword import WakeWordDetector
from brain import profanity
from core.assistant import Assistant, sorry
from core.dialog import Heard
from core.identity import Identity
from core.metrics import TurnMetrics
from core.people import People

log = logging.getLogger(__name__)


class VoiceLoop:
    hears_voice = True   # also serves as the Dialog for onboarding

    def __init__(self, mic: Microphone, wake: WakeWordDetector, endpointer: Endpointer,
                 stt: SpeechToText, assistant: Assistant, speaker: Speaker,
                 chimes: Chimes | None, identity: Identity, people: People | None,
                 follow_up_s: float = 6.0, show: Callable[[str], None] = print) -> None:
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
        self._chime("follow_up" if follow_up else "listening")
        self._quiet()                       # don't record the chime or Orion himself
        self.endpointer.reset(start_timeout_s=timeout_s)
        self.show("[listening for a follow-up]" if follow_up else "[listening]")

        status = Status.WAITING
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
        if self.people is not None:
            self.people.greet_on_start(self)
        else:
            self.say("Orion online.")
        self.show(f"Listening for '{wake_phrase}'. Ctrl+C to quit.")
        while True:
            frame = self.mic.read(timeout=1.0)
            if frame is None or not self.wake.detect(frame):
                continue
            log.debug("Wake word score %.2f", self.wake.last_score)
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

        user = self.identity.observe(heard.audio, heard.sample_rate, follow_up)
        turn.extra["utterance_s"] = round(heard.seconds, 2)
        turn.extra["follow_up"] = follow_up
        turn.extra["speaker_known"] = user is not None
        heard.audio = None  # done with the audio

        if not heard.text:
            # Background noise in a follow-up window just closes it quietly.
            if not follow_up:
                self.say(f"{sorry(user)} I didn't catch that.")
            return not follow_up

        if is_dismissal(heard.text):
            self.say(f"Any time, {user.name}." if user else "Any time.")
            return False
        mode_reply = profanity.handle_command(heard.text, self.assistant.profanity)
        if mode_reply:
            self.say(mode_reply)
            return True
        if self.people is not None and self.people.handle(heard, self):
            return True

        self.assistant.respond(heard.text, turn, self.identity.current)
        self._quiet()
        return True


# "Thanks, Orion, that's all" -> drop politeness fillers -> "that's all" -> dismissal.
_FILLERS = {"thanks", "thank", "you", "orion", "ok", "okay", "great", "cool", "perfect",
            "alright", "cheers", "no", "nope", "and", "so", "for", "now"}
_CLOSERS = {"", "that's all", "that is all", "thats all", "that's it", "thats it", "never mind",
            "nevermind", "nothing", "nothing else", "stop", "cancel", "goodbye", "bye",
            "i'm done", "im done", "all done", "i'm good", "im good", "that'll do"}


def is_dismissal(text: str) -> bool:
    """A short 'I'm done' reply that should end the conversation, not go to the LLM."""
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    if not words or len(words) > 8:
        return False
    rest = " ".join(w for w in words if w not in _FILLERS)
    return rest in _CLOSERS
