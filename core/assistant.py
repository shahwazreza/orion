"""One conversational turn: text in, spoken reply out, metrics recorded."""

from __future__ import annotations

import logging

from audio.tts import Speaker
from brain.llm import Conversation, LLMBackend, LLMError
from brain.persona import CASUAL_EXAMPLES, system_prompt
from brain.profanity import ProfanityPolicy, scrub
from brain.speech_text import SentenceChunker
from core.metrics import MetricsSink, TurnMetrics
from storage.users import User

log = logging.getLogger(__name__)


GUEST = "guest"


def sorry(user: User | None) -> str:
    return f"Sorry, {user.name}." if user else "Sorry."


class Assistant:
    """Answers one person at a time; each person gets their own chat history."""

    def __init__(self, llm: LLMBackend, speaker: Speaker, metrics: MetricsSink, profile: str,
                 assistant_name: str = "Orion", max_history_turns: int = 8,
                 profanity: ProfanityPolicy | None = None) -> None:
        self.llm = llm
        self.speaker = speaker
        self.metrics = metrics
        self.profile = profile
        self.assistant_name = assistant_name
        self.max_history_turns = max_history_turns
        self.profanity = profanity or ProfanityPolicy("on")
        self.conversations: dict[str, Conversation] = {}
        self.last_metrics: dict | None = None

    def may_swear(self, user: User | None) -> bool:
        """Only with people Orion knows, and only when no clean-mode reason applies."""
        return user is not None and self.profanity.allowed()

    def conversation_for(self, user: User | None) -> Conversation:
        key = user.id if user else GUEST
        convo = self.conversations.get(key)
        if convo is None:
            convo = self.conversations[key] = Conversation("", self.max_history_turns)
        # Refreshed every turn: picks up renames and the current time.
        casual = self.may_swear(user)
        convo.system_prompt = system_prompt(self.assistant_name, user.name if user else None,
                                            profanity=casual)
        convo.examples = list(CASUAL_EXAMPLES) if casual else []
        return convo

    def forget_guest(self) -> None:
        """Unknown speakers don't get to read each other's chat history."""
        self.conversations.pop(GUEST, None)

    def respond(self, user_text: str, turn: TurnMetrics, user: User | None = None) -> str | None:
        """Stream a reply to ``user_text``, speaking each sentence as it completes.

        Returns the reply text, or None if the model failed (Orion says why).
        """
        conversation = self.conversation_for(user)
        # Backstop: small models sometimes swear anyway (e.g. echoing earlier chat).
        clean = (lambda s: s) if self.may_swear(user) else scrub
        turn.mark("speech_end")  # no-op if the voice loop already marked it
        self.speaker.begin(on_first_audio=lambda: turn.mark("first_audio"))
        chunker = SentenceChunker()
        reply_parts: list[str] = []

        turn.mark("llm_start")
        try:
            for piece in self.llm.stream_chat(conversation.messages_with(user_text)):
                turn.mark("llm_first_token")
                reply_parts.append(piece)
                for sentence in chunker.feed(piece):
                    self._say(clean(sentence))
            for sentence in chunker.flush():
                self._say(clean(sentence))
            turn.mark("llm_done")
        except LLMError as exc:
            log.warning("LLM failure: %s", exc)
            self.speaker.say(f"{sorry(user)} {exc.spoken}")
            reply_parts = []
        finally:
            self.speaker.wait()
            turn.mark("done")

        reply = clean("".join(reply_parts).strip())
        if reply:
            conversation.commit(user_text, reply)
        turn.extra["ok"] = bool(reply)
        self.last_metrics = self.metrics.record(turn, self.profile)
        return reply or None

    def _say(self, text: str) -> None:
        if text:
            self.speaker.say(text)
