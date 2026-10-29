"""Permanent memory: facts people ask Orion to remember, recalled by meaning.

Memories belong to the person who said them and never expire; only that person
can delete them. They're written ONLY from that person's own utterances (the
command parser below), never from model output, web pages, Claude replies, or
documents. Relevant memories are given to the local model on every request;
they never go to the cloud.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from brain.embeddings import Embedder, EmbeddingError, rank, to_blob
from core.dialog import Dialog, Heard, yes_no
from storage.db import Database, now
from storage.users import User

log = logging.getLogger(__name__)


@dataclass
class Memory:
    id: int
    text: str            # as the person said it ("my sister's name is Sara")
    created: float


# ---- wording -------------------------------------------------------------
_FLIP = [(r"\bI am\b", "you are"), (r"\bI'm\b", "you're"), (r"\bI've\b", "you've"),
         (r"\bI'll\b", "you'll"), (r"\bI'd\b", "you'd"), (r"\bI\b", "you"), (r"\bmyself\b", "yourself"),
         (r"\bmine\b", "yours"), (r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bam\b", "are"),
         (r"\bwas\b(?= )", "were")]


def to_second_person(text: str) -> str:
    """How Orion says a memory back: "my sister is Sara" -> "your sister is Sara"."""
    out = text
    for pat, rep in _FLIP:
        out = re.sub(pat, lambda m, r=rep: _case(m.group(0), r), out, flags=re.I)
    return out


def _case(original: str, replacement: str) -> str:
    # "I" is always capitalised, so it says nothing about sentence position.
    if original[:1] == "I" and original.lower().startswith("i"):
        return replacement
    return replacement.capitalize() if original[:1].isupper() else replacement


_LOWER_FIRST = {"my", "the", "a", "an", "our", "his", "her", "their", "it", "this", "that", "there"}


def _clean(text: str) -> str:
    """Tidy a memory; lowercase only ordinary first words, never names ("Sara is ...")."""
    text = re.sub(r"\s+", " ", text).strip().rstrip(".!?")
    first = text.split(" ", 1)[0]
    return text[:1].lower() + text[1:] if first.lower() in _LOWER_FIRST else text


_SAVE = re.compile(r"^(?:hey,? |okay,? |so,? |also,? |and )*(?:please )?(?:remember|don't forget|do not forget|"
                   r"make a note|note)(?: that| this)?[,:]?\s+(.+)$", re.I)
_SAVE_TO = re.compile(r"^to\s+(.+)$", re.I)       # "remember to call mum" -> a need, not a reminder
# Lasting facts said in passing are saved too (spec: "lasting facts about me").
_LASTING = re.compile(
    r"^(?:by the way,? |fyi,? |just so you know,? )?(?P<fact>"
    r"my (?:birthday|name|middle name|surname|last name|phone number|address|major|minor|"
    r"favou?rite \w+|(?:mum|mom|dad|father|mother|sister|brother|wife|husband|partner|girlfriend|"
    r"boyfriend|roommate|best friend|son|daughter)(?:'s name)?) (?:is|are) .+|"
    r"i(?:'m| am) allergic to .+|i(?:'m| am) (?:vegetarian|vegan|lactose intolerant|diabetic)\b.*|"
    r"i (?:live|study|work) (?:in|at) .+|i(?:'m| am) (?:studying|majoring in) .+|"
    r"i (?:can't|cannot|don't) (?:eat|drink) .+)$", re.I)
_ABOUT_ME = re.compile(r"\bwhat (?:do you|else do you) (?:remember|know) about me\b|"
                       r"\bwhat have i told you about myself\b|\bwhat do you know of me\b", re.I)
_ABOUT_X = re.compile(r"\bwhat (?:do you|else do you) (?:remember|know) about (?!me\b)(.+?)\??$|"
                      r"\bwhat did i tell you about (.+?)\??$|\bdo you remember (?:what|anything) about (.+?)\??$",
                      re.I)
_FORGET_ALL = re.compile(r"\bforget (?:everything|all) (?:you know |you remember )?(?:about me)?\b", re.I)
_FORGET = re.compile(r"^(?:please )?(?:forget|delete the memory|remove the memory|delete what you know)"
                     r"(?: about| that| that i said)?\s+(.+)$", re.I)


# Thresholds measured with nomic-embed-text on 8 sample memories (2026-10-29):
# short topic phrases ("my sister", "nuts") scored >= 0.59 against the right
# memory and <= 0.53 against unrelated ones; full questions overlap more
# (0.52 related vs 0.53 unrelated), so context injection is looser and tells
# the model to use memories only if relevant.
RECALL_MIN = 0.56
CONTEXT_MIN = 0.52


class MemoryStore:
    def __init__(self, db: Database, embedder: Embedder) -> None:
        self.db = db
        self.embedder = embedder

    def add(self, user: User, text: str) -> int:
        try:
            emb = to_blob(self.embedder.embed_documents([text])[0])
        except EmbeddingError:
            log.warning("Memory saved without embedding (Ollama unavailable); recall will skip it")
            emb = None
        return self.db.execute("INSERT INTO memories (user_id, text, created, embedding) VALUES (?,?,?,?)",
                               (user.id, text, now(), emb))

    def all(self, user: User) -> list[Memory]:
        return [Memory(*r) for r in self.db.query(
            "SELECT id, text, created FROM memories WHERE user_id = ? ORDER BY created", (user.id,))]

    def search(self, user: User, query: str, k: int = 3, min_score: float = 0.5) -> list[tuple[float, Memory]]:
        try:
            q = self.embedder.embed_query(query)
        except EmbeddingError:
            return []
        rows = self.db.query("SELECT id, text, created, embedding FROM memories WHERE user_id = ?", (user.id,))
        return rank(q, [(Memory(r[0], r[1], r[2]), r[3]) for r in rows], k=k, min_score=min_score)

    def delete(self, user: User, memory_id: int) -> None:
        self.db.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user.id))

    def delete_all(self, user: User) -> int:
        n = len(self.all(user))
        self.db.execute("DELETE FROM memories WHERE user_id = ?", (user.id,))
        return n

    def shared_context(self, user: User, topic: str, k: int = 5) -> str:
        """Memories about a topic the person explicitly chose to share with Claude, for that
        request only. Uses the stricter topic cutoff: only what matches the named topic."""
        hits = self.search(user, topic, k=k, min_score=RECALL_MIN)
        if not hits:
            return ""
        lines = "\n".join(f'- "{m.text}"' for _, m in hits)
        return f"{user.name} chose to share these notes with you for this request:\n{lines}\n"

    def context_for(self, user: User, request: str, k: int = 4) -> str:
        """Memories relevant to ``request``, formatted for the LOCAL model's system prompt."""
        hits = self.search(user, request, k=k, min_score=CONTEXT_MIN)
        if not hits:
            return ""
        lines = "\n".join(f'- {user.name} said: "{m.text}"' for _, m in hits)
        return (f"Things {user.name} asked you to remember (use them only if relevant; "
                f"never claim to remember anything not listed):\n{lines}\n")


class MemoryCommands:
    """Saving, recalling, and forgetting by voice or text. Returns True if handled."""

    def __init__(self, store: MemoryStore, verified: Callable[[Dialog, User], bool],
                 recall_conversation: Callable[[User, str], str | None] | None = None) -> None:
        self.store = store
        self.verified = verified      # strong voice match for deletions (People._verified)
        # When no saved fact matches "what did I tell you about X", look in past
        # conversations instead (core/conversations.py).
        self.recall_conversation = recall_conversation

    def handle(self, heard: Heard, dialog: Dialog, user: User | None) -> bool:
        t = heard.text.strip()
        if not t:
            return False
        if re.match(r"^\W*do you remember\b", t, re.I) and not _ABOUT_X.search(t):
            return False   # "do you remember when..." is a question for the model
        about_me, about_x = _ABOUT_ME.search(t), _ABOUT_X.search(t)
        save, lasting = _SAVE.match(t), _LASTING.match(t.rstrip(".!"))
        forget_all, forget = _FORGET_ALL.search(t), _FORGET.match(t)
        if not any((about_me, about_x, save, lasting, forget_all, forget)):
            return False
        if forget and re.match(r"^(?:me|my (?:profile|voice|voiceprint|account)|who i am|"
                               r"(?:today'?s|this|our|the) conversation|my conversation)", forget.group(1), re.I):
            return False   # people / conversation commands, handled elsewhere
        if user is None:
            dialog.say("I only keep memories for people I recognise. Ask the owner to add you first.")
            return True

        if about_me:
            mems = self.store.all(user)
            if not mems:
                dialog.say(f"You haven't asked me to remember anything yet, {user.name}.")
            else:
                recent = [to_second_person(m.text) for m in mems[-5:]]
                more = f" That's the latest five of {len(mems)}." if len(mems) > 5 else ""
                dialog.say("You told me: " + "; ".join(recent) + "." + more)
            return True

        if about_x:
            topic = next(g for g in about_x.groups() if g)
            hits = self.store.search(user, topic, k=3, min_score=RECALL_MIN)
            from_talk = self.recall_conversation(user, topic) if not hits and self.recall_conversation else None
            if from_talk:
                dialog.say(from_talk)
            elif not hits:
                dialog.say(f"You haven't told me anything about {to_second_person(topic)}.")
            else:
                dialog.say("You told me: " + "; ".join(to_second_person(m.text) for _, m in hits) + ".")
            return True

        if forget_all:
            if not self.verified(dialog, user):
                return True
            n = len(self.store.all(user))
            if not n:
                dialog.say("There's nothing to forget.")
                return True
            answer = dialog.ask(f"Forget all {n} things you've asked me to remember? Are you sure?")
            if answer is not None and yes_no(answer.text) is True:
                self.store.delete_all(user)
                dialog.say("Done. I've forgotten all of them.")
            else:
                dialog.say("Okay, I've kept them.")
            return True

        if forget:
            hits = self.store.search(user, forget.group(1), k=1, min_score=RECALL_MIN)
            if not hits:
                dialog.say("I don't have a memory like that.")
                return True
            if not self.verified(dialog, user):
                return True
            _, mem = hits[0]
            answer = dialog.ask(f"Forget that {to_second_person(mem.text)}?")
            if answer is not None and yes_no(answer.text) is True:
                self.store.delete(user, mem.id)
                dialog.say("Forgotten.")
            else:
                dialog.say("Okay, I'll keep it.")
            return True

        text = _clean(save.group(1) if save else lasting.group("fact"))
        need = _SAVE_TO.match(text)
        if need:
            text = f"I need to {need.group(1)}"
        if len(text) < 3:
            return False
        self.store.add(user, text)
        dialog.say(f"Got it. I'll remember that {to_second_person(text)}.")
        return True
