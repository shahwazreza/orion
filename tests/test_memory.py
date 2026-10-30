"""Permanent memory: saving, recall by meaning, forgetting, per-person isolation."""

import hashlib
import re
import secrets

import numpy as np
import pytest

from brain.embeddings import EmbeddingError, unit
from core.dialog import Heard
from core.memory import MemoryCommands, MemoryStore, to_second_person
from storage.db import Database, derive_db_key
from storage.users import User

SYNONYMS = {"sibling": "sister", "sis": "sister", "nut": "peanut", "nuts": "peanut", "peanuts": "peanut",
            "born": "birthday", "bday": "birthday", "uni": "university", "college": "university"}


class FakeEmbedder:
    """Bag-of-words embeddings, with a few synonyms, so 'meaning' tests are deterministic."""

    def __init__(self):
        self.down = False

    def _vec(self, text):
        if self.down:
            raise EmbeddingError("ollama down")
        v = np.zeros(512, dtype=np.float32)
        for w in re.findall(r"[a-z]{2,}", text.lower()):
            w = SYNONYMS.get(w, w)
            if w in {"my", "the", "a", "is", "i", "what", "do", "you", "about", "to", "that", "of", "me",
                     "on", "and", "im", "it", "for", "in", "be", "being", "any", "did", "say", "ll"}:
                continue
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        return unit(v) if v.any() else v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class Dialog:
    hears_voice = True

    def __init__(self, replies=()):
        self.replies, self.said = list(replies), []

    def say(self, t):
        self.said.append(t)

    def ask(self, t, timeout_s=None):
        self.said.append(t)
        return Heard(self.replies.pop(0)) if self.replies else None


REZA, MALI = User(id="r", name="Reza", role="owner"), User(id="m", name="Malihah")


@pytest.fixture
def mem():
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    emb = FakeEmbedder()
    store = MemoryStore(db, emb)
    return store, MemoryCommands(store, verified=lambda d, u: True), emb


def say(cmds, text, user=REZA, replies=()):
    d = Dialog(replies)
    handled = cmds.handle(Heard(text), d, user)
    return handled, d.said


@pytest.mark.parametrize("text,stored", [
    ("Remember that my sister's name is Sara.", "my sister's name is Sara"),
    ("Don't forget I'm allergic to peanuts", "I'm allergic to peanuts"),
    ("remember Sara is my sister", "Sara is my sister"),
    ("Remember to call mum on Sunday", "I need to call mum on Sunday"),
    ("By the way, my birthday is March 3rd.", "my birthday is March 3rd"),
    ("I live in Tampa", "I live in Tampa"),
])
def test_saving(mem, text, stored):
    store, cmds, _ = mem
    handled, said = say(cmds, text)
    assert handled and said[0].startswith("Got it. I'll remember that")
    assert [m.text for m in store.all(REZA)] == [stored]


@pytest.mark.parametrize("text", ["What's the capital of France?", "Do you remember when Rome fell?",
                                  "I'm tired", "my code doesn't compile", "remember"])
def test_ordinary_sentences_are_not_memory_commands(mem, text):
    _, cmds, _ = mem
    assert say(cmds, text)[0] is False


def test_recall_by_meaning_and_spoken_in_second_person(mem):
    store, cmds, _ = mem
    say(cmds, "Remember that my sister's name is Sara")
    say(cmds, "Remember that I'm allergic to peanuts")
    _, said = say(cmds, "What do you know about my sibling?")
    assert said == ["You told me: your sister's name is Sara."]
    _, said = say(cmds, "what did I tell you about nuts")
    assert "allergic to peanuts" in said[0]
    _, said = say(cmds, "What do you remember about me?")
    assert said[0].startswith("You told me: your sister's name is Sara; you're allergic to peanuts")
    _, said = say(cmds, "what do you know about Jupiter")
    assert said[0].startswith("You haven't told me anything")


def test_memories_are_per_person(mem):
    store, cmds, _ = mem
    say(cmds, "Remember that my sister's name is Sara", REZA)
    _, said = say(cmds, "What do you know about my sister?", MALI)
    assert said[0].startswith("You haven't told me anything")
    assert store.all(MALI) == [] and len(store.all(REZA)) == 1


def test_strangers_cannot_save_memories(mem):
    store, cmds, _ = mem
    handled, said = say(cmds, "Remember that my PIN is 1234", None)
    assert handled and "only keep memories for people I recognise" in said[0]


def test_forget_needs_confirmation_and_verification(mem):
    store, cmds, _ = mem
    say(cmds, "Remember that my sister's name is Sara")
    say(cmds, "Forget about my sister", replies=["no"])
    assert len(store.all(REZA)) == 1
    _, said = say(cmds, "Forget about my sister", replies=["yes"])
    assert said[0] == "Forget that your sister's name is Sara?" and said[-1] == "Forgotten."
    assert store.all(REZA) == []
    blocked = MemoryCommands(store, verified=lambda d, u: False)
    say(cmds, "Remember that my birthday is March 3rd")
    assert blocked.handle(Heard("forget my birthday"), Dialog(["yes"]), REZA)
    assert len(store.all(REZA)) == 1                    # weak voice match: nothing deleted


def test_forget_everything(mem):
    store, cmds, _ = mem
    for f in ("my birthday is March 3rd", "I live in Tampa"):
        say(cmds, f"Remember that {f}")
    say(cmds, "Forget everything you know about me", replies=["yes"])
    assert store.all(REZA) == []


def test_people_and_conversation_forget_commands_are_left_alone(mem):
    _, cmds, _ = mem
    for text in ("forget me", "forget my voice", "forget today's conversation"):
        assert say(cmds, text)[0] is False


def test_context_for_local_model_only_includes_relevant_memories(mem):
    store, cmds, _ = mem
    say(cmds, "Remember that my sister's name is Sara")
    say(cmds, "Remember that I'm allergic to peanuts")
    ctx = store.context_for(REZA, "Am I allergic to peanuts?")
    assert "allergic to peanuts" in ctx and "Sara" not in ctx
    assert store.context_for(REZA, "what's the capital of Peru") == ""


def test_saving_still_works_when_embeddings_are_down(mem):
    store, cmds, emb = mem
    emb.down = True
    say(cmds, "Remember that I live in Tampa")
    assert [m.text for m in store.all(REZA)] == ["I live in Tampa"]
    assert store.search(REZA, "where do I live") == []


@pytest.mark.parametrize("text,expected", [
    ("my sister's name is Sara", "your sister's name is Sara"),
    ("I'm allergic to peanuts", "you're allergic to peanuts"),
    ("I am a student and I live in Tampa", "you are a student and you live in Tampa"),
    ("Sara gave me her keys", "Sara gave you her keys"),
])
def test_second_person(text, expected):
    assert to_second_person(text) == expected


@pytest.mark.parametrize("text", ["I want you to remember that my favorite food is biryani.",
                                  "Can you remember that my favorite food is biryani",
                                  "Make sure you remember my favorite food is biryani"])
def test_natural_ways_of_asking_to_remember(mem, text):
    # Live log: "I want you to remember that..." got "I'll make a note" from the model, and nothing was saved.
    store, cmds, _ = mem
    handled, said = say(cmds, text)
    assert handled and [m.text for m in store.all(REZA)] == ["my favorite food is biryani"]
