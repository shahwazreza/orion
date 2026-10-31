"""One conversational turn: text in, spoken reply out, metrics recorded."""

from __future__ import annotations

import logging
import re

from audio.tts import Speaker
from brain.llm import Conversation, LLMBackend, LLMError
from brain.capabilities import unsupported
from brain.web import SearchError
from core.web_search import couldnt_find, first_sentence_wants_search, needs_search, wants_another_round
from brain.persona import CASUAL_EXAMPLES, system_prompt
from brain.profanity import ProfanityPolicy, scrub
from brain.speech_text import SentenceChunker
from core.metrics import MetricsSink, TurnMetrics
from storage.users import User

log = logging.getLogger(__name__)


GUEST = "guest"


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def sorry(user: User | None) -> str:
    return f"Sorry, {user.name}." if user else "Sorry."


class Assistant:
    """Answers one person at a time; each person gets their own chat history."""

    def __init__(self, llm: LLMBackend, speaker: Speaker, metrics: MetricsSink, profile: str,
                 assistant_name: str = "Orion", max_history_turns: int = 8,
                 profanity: ProfanityPolicy | None = None, memory=None, conversations=None) -> None:
        self.llm = llm
        self.speaker = speaker
        self.metrics = metrics
        self.profile = profile
        self.assistant_name = assistant_name
        self.max_history_turns = max_history_turns
        self.profanity = profanity or ProfanityPolicy("on")
        self.conversations: dict[str, Conversation] = {}
        self.memory = memory            # core.memory.MemoryStore (Phase 2), optional
        self.cloud = None               # brain.claude.ClaudeBackend, set by main when available
        self.audit = None               # storage.audit.AuditLog: no audit log, no cloud
        self.cloud_sessions: dict[str, Conversation] = {}   # Claude-mode history, per person
        self.web = None                 # core.web_search.WebSearcher, set by main when enabled
        self.outlook = None             # core.outlook.OutlookCommands, set by main (Phase 3)
        # Yes/no question to the person (set by the voice loop / text mode).
        self.confirm = lambda prompt: False
        self.log = conversations        # core.conversations.ConversationLog (Phase 2), optional
        self.last_metrics: dict | None = None
        # Non-content facts about the last reply, for training notes. Never text.
        self.last_turn: dict | None = None

    def may_swear(self, user: User | None) -> bool:
        """Only with people Orion knows, and only when no clean-mode reason applies."""
        return user is not None and self.profanity.allowed()

    def conversation_for(self, user: User | None) -> Conversation:
        key = user.id if user else GUEST
        convo = self.conversations.get(key)
        if convo is None:
            convo = self.conversations[key] = Conversation("", self.max_history_turns)
            if user is not None and self.log is not None:
                # Continuity after a restart: pick up this person's recent turns.
                msgs = self.log.recent(user, turns=self.max_history_turns)
                for q, a in zip(msgs[::2], msgs[1::2]):
                    if q.role == "user" and a.role == "assistant":
                        convo.commit(q.text, a.text)
        # Refreshed every turn: picks up renames and the current time.
        casual = self.may_swear(user)
        convo.system_prompt = system_prompt(self.assistant_name, user.name if user else None,
                                            profanity=casual)
        convo.examples = list(CASUAL_EXAMPLES) if casual else []
        return convo

    def _context_for(self, user: User | None, text: str) -> str:
        """Memories and earlier conversations relevant to this request, for the LOCAL model only."""
        if user is None:
            return ""
        parts = []
        try:
            if self.memory is not None:
                parts.append(self.memory.context_for(user, text))
            if self.log is not None:
                parts.append(self.log.context_for(user, text))
        except Exception:
            log.exception("Context lookup failed; answering without it")
        return "".join(p for p in parts if p)

    def _remember_exchange(self, user: User | None, question: str, reply: str, route: str = "local") -> None:
        if self.log is not None:
            try:
                self.log.add_exchange(user, question, reply, route)
            except Exception:
                log.exception("Couldn't save the exchange")

    def clean_for(self, user: User | None):
        """The filter for everything spoken to ``user``: clean language unless swearing is allowed."""
        return (lambda s: s) if self.may_swear(user) else scrub

    def remember_tool_exchange(self, user: User | None, question: str, reply: str, logged: str,
                               route: str) -> None:
        """A command answered outside the model (e.g. Outlook): ``reply`` goes in this chat's
        history (memory only, for follow-ups), ``logged`` in the encrypted conversation log."""
        self.conversation_for(user).commit(question, reply)
        self._remember_exchange(user, question, logged, route=route)

    def forget_guest(self) -> None:
        """Unknown speakers don't get to read each other's chat history."""
        self.conversations.pop(GUEST, None)

    def respond(self, user_text: str, turn: TurnMetrics, user: User | None = None) -> str | None:
        """Stream a reply to ``user_text``, speaking each sentence as it completes.

        Returns the reply text, or None if the model failed (Orion says why).
        """
        conversation = self.conversation_for(user)
        conversation.context = self._context_for(user, user_text)
        refusal = unsupported(user_text, searchable=self.web is not None and self.web.enabled,
                              outlook=self.outlook is not None)
        if refusal is not None:
            # Needs live or personal data Orion doesn't have: never let the model guess.
            self.speaker.begin(on_first_audio=lambda: turn.mark("first_audio"))
            self._say(refusal)
            self.speaker.wait()
            conversation.commit(user_text, refusal)
            self._remember_exchange(user, user_text, refusal)
            turn.extra["ok"] = True
            turn.extra["capability_guard"] = True
            turn.mark("done")
            self.last_metrics = self.metrics.record(turn, self.profile)
            return refusal
        # Backstop: small models sometimes swear anyway (e.g. echoing earlier chat).
        clean = (lambda s: s) if self.may_swear(user) else scrub
        turn.mark("speech_end")  # no-op if the voice loop already marked it
        self.speaker.begin(on_first_audio=lambda: turn.mark("first_audio"))
        chunker = SentenceChunker()

        turn.mark("llm_start")
        if self.web is not None and self.web.enabled and needs_search(user_text):
            return self.respond_web(user_text, turn, user)
        reply, error, signal = self._speak_stream(
            self.llm.stream_chat(conversation.messages_with(user_text)), turn, clean, chunker,
            hold_first=first_sentence_wants_search if (self.web is not None and self.web.enabled) else None)
        if signal is not None:
            # The model doesn't know: search instead of saying "I'm not sure".
            return self.respond_web(user_text, turn, user, suggested=signal or None)
        if error is not None:
            log.warning("LLM failure: %s", error)
            self.speaker.say(f"{sorry(user)} {error.spoken}")
            reply = ""
        self.speaker.wait()
        turn.mark("done")
        if reply:
            conversation.commit(user_text, reply)
            self._remember_exchange(user, user_text, reply)
        turn.extra["ok"] = bool(reply)
        self.last_metrics = self.metrics.record(turn, self.profile)
        if reply:
            self.last_turn = {
                "mode": turn.mode,
                "profile": self.profile,
                "llm_model": getattr(self.llm, "model", None),
                "language": "casual" if self.may_swear(user) else "clean",
                "follow_up": bool(turn.extra.get("follow_up", False)),
                "reply_chars": len(reply),
                "reply_sentences": max(1, len(re.findall(r"[.!?](?:\s|$)", reply))),
                "first_audio_s": self.last_metrics.get("first_audio_s"),
                "llm_total_s": self.last_metrics.get("llm_total_s"),
            }
        return reply or None

    def _speak_stream(self, pieces, turn: TurnMetrics, clean, chunker=None,
                      hold_first=None) -> tuple[str, LLMError | None, str | None]:
        """Speak a streamed reply sentence by sentence; stop at once on barge-in.

        ``hold_first(text) -> signal | None`` checks the first sentence BEFORE it is
        spoken; a non-None signal (e.g. "search for this") stops the stream silently
        and is returned instead. Returns (reply, error, signal).
        """
        chunker = chunker or SentenceChunker()
        parts: list[str] = []
        held = hold_first is not None
        stream = iter(pieces)

        def check(sentence: str) -> str | None:
            signal = hold_first(sentence)
            if signal == "":
                # It admitted not knowing; read a little further, silently, for a suggested query.
                extra = "".join(parts)
                for more in stream:
                    extra += more
                    if len(extra) > len(sentence) + 160:
                        break
                signal = hold_first(extra) or ""
            return signal

        try:
            for piece in stream:
                if self.speaker.interrupted:
                    # Barge-in: stop generating. Leaving the loop closes the HTTP
                    # stream, which tells the model server to stop too.
                    turn.extra["interrupted"] = True
                    break
                turn.mark("llm_first_token")
                parts.append(piece)
                for sentence in chunker.feed(piece):
                    if held:
                        held = False
                        signal = check(sentence)
                        if signal is not None:
                            return "", None, signal
                    self._say(clean(sentence))
            if not self.speaker.interrupted:
                for sentence in chunker.flush():
                    if held:
                        held = False
                        signal = hold_first(sentence)
                        if signal is not None:
                            return "", None, signal
                    self._say(clean(sentence))
            turn.mark("llm_done")
        except LLMError as exc:
            return "", exc, None
        return clean("".join(parts).strip()), None, None

    # ---- cloud (Claude mode) ---------------------------------------------------
    def respond_cloud(self, user_text: str, turn: TurnMetrics, user: User | None, decision,
                      session: bool, tools: list[dict] | None = None) -> str | None:
        """Answer with Claude, under the privacy guard. Falls back to local on any failure.

        ``session``: Claude mode (keeps that mode's own history) vs a one-off "ask Claude".
        Nothing personal is added unless ``decision.include_personal`` (the person asked).
        """
        if self.cloud is None or self.audit is None:
            self.speaker.say("I can't use Claude right now, so I'll answer myself.")
            return self.respond(user_text, turn, user)
        key = (user.id if user else GUEST)
        convo = self.cloud_sessions.setdefault(key, Conversation("", self.max_history_turns)) if session \
            else Conversation("", self.max_history_turns)
        casual = self.may_swear(user)
        named = user.name if (user and decision.include_personal) else None
        convo.system_prompt = system_prompt(self.assistant_name, named, profanity=casual) + (
            "\nYou are answering through Orion's cloud mode. You have no access to this person's "
            "email, calendar, notes, memories, or past conversations; don't claim to.")
        convo.examples = []      # Claude follows the persona without few-shot turns; saves tokens
        convo.context = ""
        if decision.include_personal and user is not None and self.memory is not None:
            convo.context = self.memory.shared_context(user, decision.include_topic or user_text)
        messages = convo.messages_with(user_text)
        body = self.cloud.request_body(messages)
        if tools:
            body["tools"] = tools

        clean = (lambda s: s) if casual else scrub
        turn.extra["route"] = "cloud"
        turn.mark("speech_end")
        self.speaker.begin(on_first_audio=lambda: turn.mark("first_audio"))
        turn.mark("llm_start")
        reply, error, _ = self._speak_stream(self.cloud.stream_chat(messages, tools=tools), turn, clean)
        self.speaker.wait()
        self.audit.record("cloud_request", reason=decision.reason, body=body, usage=self.cloud.last_usage,
                          error=str(error) if error else None, included_personal=decision.include_personal)
        if error is not None:
            log.warning("Claude failure: %s", error)
            self.speaker.say(f"{error.spoken} I'll answer this one myself.")
            return self.respond(user_text, turn, user)
        turn.mark("done")
        if reply:
            if session:
                convo.commit(user_text, reply)
            self._remember_exchange(user, user_text, reply, route="cloud")
        turn.extra["ok"] = bool(reply)
        self.last_metrics = self.metrics.record(turn, self.profile)
        return reply or None

    # ---- web search -------------------------------------------------------------
    def respond_web(self, question: str, turn: TurnMetrics, user: User | None,
                    suggested: str | None = None) -> str | None:
        """Search, read, and answer in Orion's own words, naming the source."""
        web = self.web
        query, needs_ok = web.plan(question, user, suggested)
        if needs_ok:
            ok = self.confirm(f"To search that I'd have to send {_and(needs_ok)} to the search engine. "
                              "Is that okay?")
            if not ok:
                self._say("Okay, I won't search for that.")
                self.speaker.wait()
                return None
        if not query:
            self._say("I couldn't make a search out of that without personal details.")
            self.speaker.wait()
            return None
        clean = (lambda s: s) if self.may_swear(user) else scrub
        turn.extra["route"] = "web"
        turn.mark("speech_end")
        self.speaker.begin(on_first_audio=lambda: turn.mark("first_audio"))
        collected, reply = [], ""
        turn.mark("llm_start")
        for _ in range(web.max_rounds):
            try:
                collected += web.gather(query)
            except SearchError as exc:
                self._say(f"{sorry(user)} {exc}")
                self.speaker.wait()
                return None
            if not collected:
                continue
            reply, error, more = self._speak_stream(
                self.llm.stream_chat(web.messages_for(question, collected)), turn, clean,
                hold_first=wants_another_round)
            if error is not None:
                self._say(f"{sorry(user)} {error.spoken}")
                self.speaker.wait()
                return None
            if more is None:
                break
            query, _ = web.privacy.check(more, question, user)
            if not query:
                break
        self.speaker.wait()
        web.last_sources = [r for r, _ in collected]
        if not reply:
            reply = "I searched, but I couldn't find a reliable answer."
            self._say(reply)
            self.speaker.wait()
        if self.audit is not None:
            self.audit.record("web_answer", question=question, sources=[r.url for r, _ in collected])
        turn.mark("done")
        self._remember_exchange(user, question, reply, route="web")
        self.conversation_for(user).commit(question, reply)
        turn.extra["ok"] = True
        self.last_metrics = self.metrics.record(turn, self.profile)

        if (couldnt_find(reply) or web.is_big(question)) and self.cloud is not None:
            if self.confirm("This is a big one. Want me to ask Claude? It would search the web too."):
                from core.router import CLOUD, Decision

                return self.respond_cloud(question, TurnMetrics(turn.mode), user,
                                          Decision(CLOUD, "web handoff", question), session=False,
                                          tools=[{"type": "web_search_20260209", "name": "web_search",
                                                  "max_uses": 3}])
        return reply

    def end_cloud_session(self, user: User | None) -> None:
        self.cloud_sessions.pop(user.id if user else GUEST, None)

    def _say(self, text: str) -> None:
        if text:
            self.speaker.say(text)
