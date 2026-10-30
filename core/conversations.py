"""Conversation memory: every exchange, per person, encrypted, recallable.

What's kept: who said what and when (local or cloud), per recognised person.
Strangers' conversations are never saved. Each person's recent turns are
restored after a restart; older ones are found by date ("yesterday") or by
meaning ("what did I say about my exam?"), helped by a short daily summary
written by the local model. Past conversations go to the local model only,
never to the cloud.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from brain.embeddings import Embedder, EmbeddingError, rank, to_blob
from core.dialog import Dialog, Heard, yes_no
from storage.db import Database, now
from storage.users import User

log = logging.getLogger(__name__)

RECALL_MIN = 0.56          # same calibration as memories (core/memory.py)


@dataclass
class Message:
    id: int
    ts: float
    role: str
    text: str
    route: str


def day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts).date().isoformat()


def spoken_day(d: date, today: date) -> str:
    if d == today:
        return "today"
    if d == today - timedelta(days=1):
        return "yesterday"
    if today - d < timedelta(days=7):
        return f"on {d:%A}"
    return f"on {d:%A} the {_ordinal(d.day)} of {d:%B}"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


class ConversationLog:
    def __init__(self, db: Database, embedder: Embedder,
                 summarize: Callable[[str, list[Message]], str] | None = None,
                 background: bool = True) -> None:
        self.db = db
        self.embedder = embedder
        self.summarize = summarize          # (name, messages) -> short summary; local LLM
        self._jobs: queue.Queue = queue.Queue()
        self._paused: set[str] = set()      # users who said "don't save this conversation"
        self._started: dict[str, int] = {}  # user -> last message id before this conversation
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._worker, name="conversation-index", daemon=True)
            self._thread.start()
        # Anything left unindexed last time (e.g. Orion quit right after the last
        # exchange) is indexed now, so recall never silently misses it.
        for msg_id, text in db.query("SELECT id, text FROM messages WHERE role = 'user' AND embedding IS NULL"):
            self._submit(("embed", msg_id, text))

    # ---- writing --------------------------------------------------------------
    def start_conversation(self, user: User | None) -> None:
        if user is not None:
            self._started[user.id] = self.db.query("SELECT coalesce(max(id), 0) FROM messages")[0][0]
            self._paused.discard(user.id)

    def pause(self, user: User) -> int:
        """Don't save this conversation: stop saving and delete what was saved of it."""
        self._paused.add(user.id)
        since = self._started.get(user.id)
        if since is None:      # no conversation marker: only stop saving from now on
            return 0
        n = self.db.query("SELECT count(*) FROM messages WHERE user_id = ? AND id > ?", (user.id, since))[0][0]
        self.db.execute("DELETE FROM messages WHERE user_id = ? AND id > ?", (user.id, since))
        return n

    def saving(self, user: User | None) -> bool:
        return user is not None and user.id not in self._paused

    def add_exchange(self, user: User | None, question: str, reply: str, route: str = "local") -> None:
        if not self.saving(user) or not question.strip() or not reply.strip():
            return
        t = now()
        q_id = self.db.execute("INSERT INTO messages (user_id, ts, role, text, route) VALUES (?,?,?,?,?)",
                               (user.id, t, "user", question, route))
        self.db.execute("INSERT INTO messages (user_id, ts, role, text, route) VALUES (?,?,?,?,?)",
                        (user.id, t, "assistant", reply, route))
        # Index what the person said: recall is "what did *I* say about ...".
        self._submit(("embed", q_id, question))

    def _submit(self, job) -> None:
        if self._thread is None:
            self._run(job)
        else:
            self._jobs.put(job)

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._run(job)
            except Exception:
                log.exception("Conversation indexing failed")
            finally:
                self._jobs.task_done()

    def close(self, timeout: float = 10.0) -> None:
        """Finish pending indexing (called on shutdown)."""
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join(timeout)

    def _run(self, job) -> None:
        kind, *args = job
        if kind == "embed":
            msg_id, text = args
            try:
                emb = to_blob(self.embedder.embed_documents([text])[0])
            except EmbeddingError:
                return
            self.db.execute("UPDATE messages SET embedding = ? WHERE id = ?", (emb, msg_id))
        elif kind == "summaries":
            self._summarize_past_days(*args)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for background indexing (tests)."""
        end = time.monotonic() + timeout
        while not self._jobs.empty() and time.monotonic() < end:
            time.sleep(0.01)
        time.sleep(0.05)

    # ---- reading ----------------------------------------------------------------
    def recent(self, user: User, turns: int = 8, within_hours: float = 24.0) -> list[Message]:
        """The last ``turns`` exchanges from the past day, oldest first (restored on restart)."""
        rows = self.db.query("SELECT id, ts, role, text, route FROM messages WHERE user_id = ? AND ts >= ? "
                             "ORDER BY id DESC LIMIT ?", (user.id, now() - within_hours * 3600, turns * 2))
        msgs = [Message(*r) for r in reversed(rows)]
        while msgs and msgs[0].role != "user":      # start on a question
            msgs.pop(0)
        return msgs

    def on_day(self, user: User, day: str) -> list[Message]:
        start = datetime.fromisoformat(day).timestamp()
        return [Message(*r) for r in self.db.query(
            "SELECT id, ts, role, text, route FROM messages WHERE user_id = ? AND ts >= ? AND ts < ? ORDER BY id",
            (user.id, start, start + 86400))]

    def search(self, user: User, query: str, k: int = 3, min_score: float = RECALL_MIN,
               before: float | None = None) -> list[tuple[float, tuple[Message, Message | None]]]:
        """Past exchanges (question, reply) matching ``query`` by meaning."""
        try:
            q = self.embedder.embed_query(query)
        except EmbeddingError:
            return []
        rows = self.db.query("SELECT id, ts, role, text, route, embedding FROM messages "
                             "WHERE user_id = ? AND role = 'user' AND embedding IS NOT NULL"
                             + (" AND ts < ?" if before else ""),
                             (user.id, before) if before else (user.id,))
        hits = rank(q, [(Message(*r[:5]), r[5]) for r in rows], k=k, min_score=min_score)
        out = []
        for score, question in hits:
            reply = self.db.query("SELECT id, ts, role, text, route FROM messages WHERE user_id = ? AND "
                                  "role = 'assistant' AND id > ? ORDER BY id LIMIT 1", (user.id, question.id))
            out.append((score, (question, Message(*reply[0]) if reply else None)))
        return out

    def context_for(self, user: User, request: str, k: int = 2) -> str:
        """Earlier exchanges (not from the last hour) relevant to ``request``, for the LOCAL model."""
        # Measured with nomic-embed-text: related past requests 0.60-0.77, unrelated <= 0.52.
        hits = self.search(user, request, k=k, min_score=0.57, before=now() - 3600)
        if not hits:
            return ""
        lines = "\n".join(f'- {spoken_day(datetime.fromtimestamp(q.ts).date(), date.today())}, '
                          f'{user.name} said "{q.text}" and you replied "{(r.text if r else "")[:200]}"'
                          for _, (q, r) in hits)
        return f"Earlier conversations with {user.name} that may be relevant:\n{lines}\n"

    # ---- summaries -----------------------------------------------------------------
    def summary(self, user: User, day: str) -> str | None:
        rows = self.db.query("SELECT text FROM summaries WHERE user_id = ? AND day = ?", (user.id, day))
        return rows[0][0] if rows else None

    def summarize_day(self, user: User, day: str, store: bool = True) -> str | None:
        msgs = self.on_day(user, day)
        if not msgs or self.summarize is None:
            return None
        text = self.summarize(user.name, msgs)
        if store and text:
            try:
                emb = to_blob(self.embedder.embed_documents([text])[0])
            except EmbeddingError:
                emb = None
            self.db.execute("INSERT OR REPLACE INTO summaries (user_id, day, text, embedding) VALUES (?,?,?,?)",
                            (user.id, day, text, emb))
        return text

    def summarize_past_days(self, users: list[User]) -> None:
        """Write missing summaries for days before today, in the background."""
        self._submit(("summaries", users))

    def _summarize_past_days(self, users: list[User]) -> None:
        today = date.today().isoformat()
        for user in users:
            days = {day_of(r[0]) for r in self.db.query("SELECT ts FROM messages WHERE user_id = ?", (user.id,))}
            done = {r[0] for r in self.db.query("SELECT day FROM summaries WHERE user_id = ?", (user.id,))}
            for day in sorted(days - done):
                if day < today:
                    self.summarize_day(user, day)

    # ---- deleting / retention --------------------------------------------------------
    def delete_day(self, user: User, day: str) -> int:
        start = datetime.fromisoformat(day).timestamp()
        n = self.db.query("SELECT count(*) FROM messages WHERE user_id = ? AND ts >= ? AND ts < ?",
                          (user.id, start, start + 86400))[0][0]
        self.db.execute("DELETE FROM messages WHERE user_id = ? AND ts >= ? AND ts < ?", (user.id, start, start + 86400))
        self.db.execute("DELETE FROM summaries WHERE user_id = ? AND day = ?", (user.id, day))
        return n

    def delete_all(self, user: User) -> int:
        n = self.db.query("SELECT count(*) FROM messages WHERE user_id = ?", (user.id,))[0][0]
        self.db.execute("DELETE FROM messages WHERE user_id = ?", (user.id,))
        self.db.execute("DELETE FROM summaries WHERE user_id = ?", (user.id,))
        return n

    def apply_retention(self, keep_days: int | None) -> int:
        if not keep_days:
            return 0
        cutoff = now() - keep_days * 86400
        n = self.db.query("SELECT count(*) FROM messages WHERE ts < ?", (cutoff,))[0][0]
        self.db.execute("DELETE FROM messages WHERE ts < ?", (cutoff,))
        self.db.execute("DELETE FROM summaries WHERE day < ?", (datetime.fromtimestamp(cutoff).date().isoformat(),))
        return n


# ---- commands -------------------------------------------------------------------------
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WHAT_TALKED = re.compile(r"\bwhat (?:did|have) (?:we|i) (?:talk(?:ed)? about|discuss(?:ed)?|chat(?:ted)? about|say)"
                          r"(?P<when>.*?)\??$", re.I)
_SEARCH = re.compile(r"\b(?:what did i (?:say|tell you|mention) about|did i (?:ever )?(?:mention|tell you about|"
                     r"talk about)|when did (?:i|we) (?:last )?(?:talk|speak) about|have we talked about)\s+(.+?)\??$",
                     re.I)
_FORGET_DAY = re.compile(r"\b(?:forget|delete|erase|remove) (?:today'?s|this|our) conversation(?:s)?(?: today)?\b|"
                         r"\b(?:forget|delete|erase) (?:what we (?:talked about|said) )?today\b", re.I)
_FORGET_ALL = re.compile(r"\b(?:forget|delete|erase|clear|remove) (?:all )?(?:of )?(?:my|our) conversation"
                         r"(?:s| history)?\b|\bdelete my (?:chat )?history\b", re.I)
_DONT_SAVE = re.compile(r"\b(?:don'?t|do not|stop) (?:save|saving|record|recording|keep|keeping) "
                        r"(?:this|our|the) conversation\b|\boff the record\b", re.I)


def parse_when(text: str, today: date) -> list[date] | None:
    t = text.lower()
    if not t.strip() or "today" in t or "this morning" in t or "earlier" in t:
        return [today]
    if "yesterday" in t or "last night" in t:
        return [today - timedelta(days=1)]
    if "last week" in t or "this week" in t or "past week" in t:
        return [today - timedelta(days=i) for i in range(7, 0, -1)]
    for i, name in enumerate(_WEEKDAYS):
        if name in t:
            back = (today.weekday() - i) % 7 or 7
            return [today - timedelta(days=back)]
    return None


class ConversationCommands:
    def __init__(self, log: ConversationLog, verified: Callable[[Dialog, User], bool]) -> None:
        self.log = log
        self.verified = verified

    def recall_topic(self, user: User, topic: str) -> str | None:
        """Spoken answer to "what did I tell you about X" from past conversations, or None."""
        hits = self.log.search(user, topic, k=1)
        if not hits:
            return None
        _, (q, r) = hits[0]
        when = spoken_day(datetime.fromtimestamp(q.ts).date(), date.today())
        return f"{when.capitalize()}, you said: {q.text.rstrip('.?!')}." + (
            f" And I said: {r.text[:240].rstrip('.?!')}." if r else "")

    def handle(self, heard: Heard, dialog: Dialog, user: User | None, today: date | None = None) -> bool:
        t = heard.text.strip()
        today = today or date.today()
        dont, forget_day, forget_all = _DONT_SAVE.search(t), _FORGET_DAY.search(t), _FORGET_ALL.search(t)
        talked, search = _WHAT_TALKED.search(t), _SEARCH.search(t)
        if not any((dont, forget_day, forget_all, talked, search)):
            return False
        if user is None:
            dialog.say("I don't keep conversations with people I don't recognise.")
            return True

        if dont:
            n = self.log.pause(user)
            dialog.say("Okay, this conversation stays off the record."
                       + (" I've also dropped what we said so far." if n else ""))
            return True

        if forget_all:
            if not self.verified(dialog, user):
                return True
            answer = dialog.ask(f"Delete all our past conversations, {user.name}? Your saved memories "
                                "and rules stay. Are you sure?")
            if answer is not None and yes_no(answer.text) is True:
                self.log.delete_all(user)
                dialog.say("Done. Our conversation history is gone.")
            else:
                dialog.say("Okay, I've kept it.")
            return True

        if forget_day:
            if not self.verified(dialog, user):
                return True
            answer = dialog.ask("Forget everything we talked about today?")
            if answer is not None and yes_no(answer.text) is True:
                self.log.delete_day(user, today.isoformat())
                self.log.start_conversation(user)
                dialog.say("Done. Today's conversation is forgotten.")
            else:
                dialog.say("Okay, I'll keep it.")
            return True

        if search:
            answer = self.recall_topic(user, search.group(1))
            dialog.say(answer or f"I don't remember us talking about {search.group(1).rstrip('.?!')}.")
            return True

        days = parse_when(talked.group("when"), today)
        if days is None:
            return False
        parts = []
        for d in days:
            day = d.isoformat()
            text = self.log.summary(user, day) or (self.log.summarize_day(user, day, store=d != today)
                                                   if self.log.on_day(user, day) else None)
            if text:
                parts.append(f"{spoken_day(d, today).capitalize()}: {text}" if len(days) > 1 else text)
        if not parts:
            dialog.say(f"We didn't talk {spoken_day(days[0], today) if len(days) == 1 else 'that week'}"
                       + (", or you asked me not to keep it." if days[0] != today else " yet."))
        else:
            dialog.say(" ".join(parts))
        return True


def make_summarizer(llm) -> Callable[[str, list[Message]], str]:
    """A local-model summarizer: 2 to 3 spoken-style sentences about one day."""
    def summarize(name: str, msgs: list[Message]) -> str:
        transcript = "\n".join(f"{name if m.role == 'user' else 'Orion'}: {m.text}" for m in msgs)[-6000:]
        prompt = [{"role": "system", "content":
                   "You summarise a day's conversation between a person and their voice assistant, Orion. "
                   "Write 2 or 3 short plain sentences in the second person, addressed to the person, "
                   "and refer to Orion as 'I' ('You asked about...', 'I told you...', 'We talked "
                   "about...'). Only include what is in the transcript. No lists, no markdown."},
                  {"role": "user", "content": transcript}]
        return "".join(llm.stream_chat(prompt)).strip()
    return summarize
