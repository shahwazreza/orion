"""Standing instructions: "whenever X, do Y", matched by meaning.

How a rule fires (measured on llama3.2:3b + nomic-embed-text, see docs/benchmarks.md):
embeddings alone fired on look-alikes ("my roommate left the room" scored
higher than "leaving now"), and the 3B model as a yes/no judge missed most real
triggers. What works: when a rule is created the model writes casual
variations of the trigger ("I'm heading out", "off I go"); variations that
drift in meaning are dropped; each utterance is matched against the closest
variation; and simple guards reject questions, other people, and past or
future plans. People correct it: "that was wrong" right after a reminder adds
a "not this" example, and "also remind me when I say ..." adds a phrase.

Rules are created and changed only from the person's own commands, never from
model output or external content.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from brain.embeddings import Embedder, EmbeddingError, from_blob, to_blob
from core.dialog import Dialog, Heard, yes_no
from core.memory import to_second_person
from storage.db import Database, now
from storage.users import User

log = logging.getLogger(__name__)

FIRE_MIN = 0.62          # closest variation must be at least this similar
DRIFT_MIN = 0.60         # generated variations must stay this close to the trigger
NEGATIVE_MARGIN = 0.02   # closer to a "not this" example than to the trigger: don't fire
FEEDBACK_WINDOW_S = 90   # "that was wrong" this soon after a reminder corrects the rule

RULE_PHRASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS rule_phrases (
    id INTEGER PRIMARY KEY,
    rule_id INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('match', 'not')),
    text TEXT NOT NULL,
    source TEXT NOT NULL,           -- 'trigger' | 'generated' | 'user'
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS rule_phrases_rule ON rule_phrases(rule_id);
"""

# Utterances that can't be "this is happening to me now".
_GUARD = re.compile(
    r"\?$|^(?:how|what|when|where|why|who|which|should|can|could|do|does|did|is|are|will|would)\b|"
    r"\b(?:yesterday|tomorrow|last (?:night|week)|next|later today|in an? (?:hour|minute)|"
    r"i'll|i will|i'm going to (?!bed|sleep)|thinking (?:about|of)|i was|used to|"
    r"starts? at|begins? at|at \d|on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b|"
    r"^(?:my|his|her|their|our) (?!class\b)\w+ (?:left|is|was|went|starts)|^(?:he|she|they|someone)\b",
    re.I)


@dataclass
class Rule:
    id: int
    user_id: str
    trigger: str        # as said ("I'm leaving my room")
    action: str         # as said ("remind me to take my keys")
    match: list[np.ndarray] = field(default_factory=list)
    not_: list[np.ndarray] = field(default_factory=list)

    def spoken_trigger(self) -> str:
        return to_second_person(self.trigger)

    def spoken_action(self) -> str:
        """What Orion says when the rule fires."""
        a = self.action.strip().rstrip(".!")
        m = re.match(r"^(?:please )?remind me (?:to )(.+)$", a, re.I)
        if m:
            return f"Don't forget to {to_second_person(m.group(1))}."
        m = re.match(r"^(?:please )?remind me (?:about |of )?(.+)$", a, re.I)
        if m:
            return f"Don't forget {to_second_person(m.group(1))}."
        m = re.match(r"^(?:please )?(?:tell|warn) me (?:to )(.+)$", a, re.I)
        if m:
            return f"{to_second_person(m.group(1)).capitalize()}."
        m = re.match(r"^(?:please )?(?:tell|warn) me (?:that |about )?(.+)$", a, re.I)
        if m:
            return f"{to_second_person(m.group(1)).capitalize()}."
        m = re.match(r"^(?:please )?(?:say|ask me) (.+)$", a, re.I)
        if m:
            return f"{m.group(1).strip(chr(34)).capitalize()}" + ("" if m.group(1).endswith("?") else ".")
        return f"Reminder: {to_second_person(a)}."

    def spoken_summary(self) -> str:
        return f"whenever {self.spoken_trigger()}, I'll {_promise(self.action)}"


def _promise(action: str) -> str:
    """ "remind me to take my keys" -> "remind you to take your keys" """
    return to_second_person(action.strip().rstrip(".!"))


class RuleBook:
    def __init__(self, db: Database, embedder: Embedder,
                 paraphrase: Callable[[str], list[str]] | None = None, background: bool = True) -> None:
        self.db = db
        self.embedder = embedder
        self.paraphrase = paraphrase       # trigger -> casual variations (local LLM)
        self.background = background
        with db._lock:
            db._conn.executescript(RULE_PHRASES_SCHEMA)
            db._conn.commit()
        self.last_fired: tuple[int, str, float] | None = None   # (rule id, utterance, time)

    # ---- storage --------------------------------------------------------------
    def add(self, user: User, trigger: str, action: str) -> int:
        t = now()
        rid = self.db.execute("INSERT INTO rules (user_id, trigger, action, created, updated) VALUES (?,?,?,?,?)",
                              (user.id, trigger, action, t, t))
        self._add_phrases(rid, "match", [trigger], "trigger")
        if self.paraphrase is not None:
            job = lambda: self._expand(rid, trigger)       # noqa: E731
            threading.Thread(target=job, daemon=True).start() if self.background else job()
        return rid

    def _expand(self, rule_id: int, trigger: str) -> None:
        try:
            variations = [v for v in self.paraphrase(trigger) if v.lower() != trigger.lower()]
            if not variations:
                return
            base = self.embedder.embed_query(trigger)
            vecs = self.embedder.embed_documents(variations)
        except Exception:
            log.exception("Couldn't generate trigger variations; matching on the trigger only")
            return
        keep = [(v, e) for v, e in zip(variations, vecs) if float(base @ e) >= DRIFT_MIN]
        self.db.executemany("INSERT INTO rule_phrases (rule_id, kind, text, source, embedding) VALUES (?,?,?,?,?)",
                            [(rule_id, "match", v, "generated", to_blob(e)) for v, e in keep])
        log.info("Rule %d: %d trigger variations kept, %d dropped for drifting", rule_id, len(keep),
                 len(variations) - len(keep))

    def _add_phrases(self, rule_id: int, kind: str, texts: list[str], source: str) -> None:
        try:
            vecs = self.embedder.embed_documents(texts)
        except EmbeddingError:
            vecs = [None] * len(texts)
        self.db.executemany("INSERT INTO rule_phrases (rule_id, kind, text, source, embedding) VALUES (?,?,?,?,?)",
                            [(rule_id, kind, t, source, to_blob(v) if v is not None else None)
                             for t, v in zip(texts, vecs)])

    def all(self, user: User) -> list[Rule]:
        rules = [Rule(*r) for r in self.db.query(
            "SELECT id, user_id, trigger, action FROM rules WHERE user_id = ? ORDER BY created", (user.id,))]
        for rule in rules:
            for kind, blob in self.db.query("SELECT kind, embedding FROM rule_phrases WHERE rule_id = ?", (rule.id,)):
                v = from_blob(blob)
                if v is not None:
                    (rule.match if kind == "match" else rule.not_).append(v)
        return rules

    def delete(self, user: User, rule_id: int) -> None:
        self.db.execute("DELETE FROM rule_phrases WHERE rule_id = ?", (rule_id,))
        self.db.execute("DELETE FROM rules WHERE id = ? AND user_id = ?", (rule_id, user.id))

    def update_action(self, user: User, rule_id: int, action: str) -> None:
        self.db.execute("UPDATE rules SET action = ?, updated = ? WHERE id = ? AND user_id = ?",
                        (action, now(), rule_id, user.id))

    def add_example(self, rule_id: int, text: str, kind: str) -> None:
        self._add_phrases(rule_id, kind, [text], "user")

    # ---- matching -------------------------------------------------------------
    def find(self, user: User, description: str) -> tuple[float, Rule] | None:
        """The rule a person is referring to ("the keys rule", "about my keys")."""
        rules = self.all(user)
        if not rules:
            return None
        try:
            q = self.embedder.embed_query(description)
            vecs = self.embedder.embed_documents([f"{r.trigger}. {r.action}" for r in rules])
        except EmbeddingError:
            return None
        score, rule = max(((float(q @ v), r) for v, r in zip(vecs, rules)), key=lambda x: x[0])
        return (score, rule) if score >= 0.5 or len(rules) == 1 else None

    def check(self, user: User | None, utterance: str) -> list[Rule]:
        """Rules that this utterance triggers (at most one: the closest)."""
        if user is None or not utterance.strip() or _GUARD.search(utterance.lower().strip()):
            return []
        rules = [r for r in self.all(user) if r.match]
        if not rules:
            return []
        try:
            q = self.embedder.embed_query(utterance)
        except EmbeddingError:
            return []
        scored = []
        for r in rules:
            pos = max(float(q @ v) for v in r.match)
            neg = max((float(q @ v) for v in r.not_), default=-1.0)
            if pos >= FIRE_MIN and neg < pos - NEGATIVE_MARGIN:
                scored.append((pos, r))
        if not scored:
            return []
        score, rule = max(scored, key=lambda x: x[0])
        log.debug("Rule %d fired (similarity %.2f)", rule.id, score)
        self.last_fired = (rule.id, utterance, time.monotonic())
        return [rule]


# ---- commands -----------------------------------------------------------------
_TRIGGER_LEAD = re.compile(r"^(?:i (?:say|tell you|mention)(?: that)?|you hear me say|i'm saying)\s+", re.I)
_CREATE = [
    # "whenever X, Y": the comma is the most reliable split ("whenever I say I'm leaving, remind me")
    re.compile(r"^(?:please |orion,? )?(?:whenever|every time|each time|any ?time)\s+(?P<trig>[^,]+),\s*"
               r"(?:then\s+|please\s+|always\s+)?(?P<act>(?:remind|tell|warn|ask|say|make sure)\b.+)$", re.I),
    # no comma (common in speech-to-text): split at the action verb; "say" is excluded
    # because triggers often start "I say ..."
    re.compile(r"^(?:please |orion,? )?(?:whenever|every time|each time|any ?time)\s+(?P<trig>.+?)\s+"
               r"(?:then\s+|please\s+|always\s+)?(?P<act>(?:remind|tell|warn|make sure)\b.+)$", re.I),
    re.compile(r"^(?:please |orion,? )?always\s+(?P<act>(?:remind|tell|warn|ask|say|make sure)\b.+?),?\s+"
               r"(?:when|whenever|every time|if)\s+(?P<trig>.+)$", re.I),
    re.compile(r"^(?:please |orion,? )?(?:if|when)\s+(?P<trig>.+?),\s*always\s+(?P<act>(?:remind|tell|warn|say)\b.+)$",
               re.I),
]
_LIST = re.compile(r"\b(?:what (?:rules|standing instructions) do i have|what are my (?:rules|standing instructions)|"
                   r"list (?:my|the) rules|(?:tell|read) me my rules|what (?:are you|do you) (?:supposed|meant) to "
                   r"remind me)\b", re.I)
_DELETE = re.compile(r"^(?:please )?(?:stop reminding me (?:about |to |of )?(?P<a>.+)|"
                     r"(?:delete|remove|forget|cancel|drop) the (?P<b>.+?) rule|"
                     r"(?:delete|remove|forget|cancel) (?:the |my )?rule (?:about|for) (?P<c>.+))$", re.I)
_DELETE_ALL = re.compile(r"\b(?:delete|remove|forget|clear|cancel) (?:all )?(?:of )?my rules\b", re.I)
_EDIT = re.compile(r"^(?:please )?(?:change|update|edit) the (?P<which>.+?) rule (?:so (?:that )?it |to )"
                   r"(?P<also>also )?(?:include |remind me (?:about |to )?|say |)(?P<new>.+)$", re.I)
_ADD_EXAMPLE = re.compile(r"^(?:and |also |please )*(?:remind me|do that|that counts|trigger (?:it|that|the "
                          r"(?P<which>.+?) rule))?(?: also)? (?:when(?:ever)?|if) i say (?P<phrase>.+)$", re.I)
_WRONG = re.compile(r"^(?:no,? |nope,? )*(?:that was wrong|wrong reminder|not now|i'm not|i am not|"
                    r"that's not what i meant|you misunderstood|false alarm)\b", re.I)


class RuleCommands:
    def __init__(self, book: RuleBook, verified: Callable[[Dialog, User], bool]) -> None:
        self.book = book
        self.verified = verified

    def handle(self, heard: Heard, dialog: Dialog, user: User | None) -> bool:
        t = heard.text.strip().rstrip(".!")
        if not t:
            return False

        if user is not None and self.book.last_fired and _WRONG.match(t):
            rule_id, utterance, at = self.book.last_fired
            if time.monotonic() - at <= FEEDBACK_WINDOW_S:
                self.book.add_example(rule_id, utterance, "not")
                self.book.last_fired = None
                dialog.say("Sorry. I won't remind you for that again.")
                return True

        m = _ADD_EXAMPLE.match(t)
        if m and user is not None and (m.group("which") or self.book.last_fired or len(self.book.all(user)) == 1):
            target = None
            if m.group("which"):
                found = self.book.find(user, m.group("which"))
                target = found[1] if found else None
            elif self.book.last_fired:
                target = next((r for r in self.book.all(user) if r.id == self.book.last_fired[0]), None)
            else:
                target = self.book.all(user)[0]
            if target is None:
                dialog.say("I'm not sure which rule you mean.")
                return True
            phrase = m.group("phrase").strip(" \"'")
            self.book.add_example(target.id, phrase, "match")
            dialog.say(f'Got it. "{phrase}" counts too.')
            return True

        create = next((p.match(t) for p in _CREATE if p.match(t)), None)
        listing, delete_all, delete, edit = _LIST.search(t), _DELETE_ALL.search(t), _DELETE.match(t), _EDIT.match(t)
        if not any((create, listing, delete_all, delete, edit)):
            return False
        if user is None:
            dialog.say("I only keep standing instructions for people I recognise.")
            return True

        if create:
            trigger = _TRIGGER_LEAD.sub("", create.group("trig").strip(" ,"))
            action = create.group("act").strip(" ,")
            rule = Rule(0, user.id, trigger, action)
            self.book.add(user, trigger, action)
            dialog.say(f"Got it. {rule.spoken_summary()[0].upper()}{rule.spoken_summary()[1:]}.")
            return True

        if listing:
            rules = self.book.all(user)
            if not rules:
                dialog.say("You don't have any standing instructions yet.")
            else:
                parts = [r.spoken_summary() for r in rules]
                dialog.say(f"You have {len(rules)}: " + "; ".join(parts) + ".")
            return True

        if delete_all:
            rules = self.book.all(user)
            if not rules:
                dialog.say("You don't have any rules.")
                return True
            if not self.verified(dialog, user):
                return True
            answer = dialog.ask(f"Delete all {len(rules)} of your rules?")
            if answer is not None and yes_no(answer.text) is True:
                for r in rules:
                    self.book.delete(user, r.id)
                dialog.say("Done. No more standing instructions.")
            else:
                dialog.say("Okay, I've kept them.")
            return True

        if delete:
            found = self.book.find(user, next(g for g in delete.groups() if g))
            if found is None:
                dialog.say("I don't have a rule like that.")
                return True
            if not self.verified(dialog, user):
                return True
            _, rule = found
            answer = dialog.ask(f"Stop the rule: {rule.spoken_summary()}?")
            if answer is not None and yes_no(answer.text) is True:
                self.book.delete(user, rule.id)
                dialog.say("Done.")
            else:
                dialog.say("Okay, I'll keep it.")
            return True

        found = self.book.find(user, edit.group("which"))
        if found is None:
            dialog.say("I don't have a rule like that.")
            return True
        if not self.verified(dialog, user):
            return True
        _, rule = found
        addition = edit.group("new").strip(" ,")
        if edit.group("also"):
            new_action = f"{rule.action.rstrip('. ')} and {addition}"
        else:
            new_action = addition if re.match(r"^(?:remind|tell|warn|say|ask)\b", addition, re.I) \
                else f"remind me {addition}" if addition.lower().startswith("to ") else f"remind me about {addition}"
        updated = Rule(rule.id, rule.user_id, rule.trigger, new_action)
        answer = dialog.ask(f"So: {updated.spoken_summary()}. Is that right?")
        if answer is not None and yes_no(answer.text) is True:
            self.book.update_action(user, rule.id, new_action)
            dialog.say("Updated.")
        else:
            dialog.say("Okay, I've left it as it was.")
        return True


def make_paraphraser(llm_host: str, model: str) -> Callable[[str], list[str]]:
    """Casual variations of a trigger, from the local model (meaning-preserving prompt)."""
    import httpx

    def paraphrase(trigger: str) -> list[str]:
        prompt = (f'The sentence is: "{trigger}"\nWrite 12 short, casual ways a person might say exactly this '
                  "about themselves, as it is happening. Keep the same meaning; don't add new details. "
                  "One per line, no numbering, no quotes.")
        r = httpx.post(f"{llm_host.rstrip('/')}/api/chat", timeout=120, json={
            "model": model, "stream": False, "keep_alive": "30m", "options": {"temperature": 0.7},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        lines = [re.sub(r"^\W*\d*[.)]?\s*", "", ln).strip(' "') for ln in r.json()["message"]["content"].splitlines()]
        return [ln for ln in lines if 1 <= len(ln.split()) <= 10]

    return paraphrase
