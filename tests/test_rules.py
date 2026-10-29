"""Standing rules: creation from natural phrasing, firing by meaning, guards, corrections."""

import secrets
import time

import pytest

from core.dialog import Heard
from core.rules import Rule, RuleBook, RuleCommands
from storage.db import Database, derive_db_key
from storage.users import User
from tests.test_memory import Dialog, FakeEmbedder

REZA, MALI = User(id="r", name="Reza", role="owner"), User(id="m", name="Malihah")
# Word-overlap fake embeddings can't do synonyms, so variations share words with the trigger.
VARIATIONS = {"I'm leaving my room": ["leaving my room now", "leaving room heading out"],
              "I'm going to bed": ["going to bed now"]}


@pytest.fixture
def rules():
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    book = RuleBook(db, FakeEmbedder(), paraphrase=lambda t: VARIATIONS.get(t, []), background=False)
    return book, RuleCommands(book, verified=lambda d, u: True)


def say(cmds, text, user=REZA, replies=()):
    d = Dialog(replies)
    return cmds.handle(Heard(text), d, user), d.said


@pytest.mark.parametrize("text,trigger,action", [
    ("Whenever I say I'm leaving my room, remind me to take my keys.", "I'm leaving my room",
     "remind me to take my keys"),
    ("Every time I'm going to bed, remind me to drink water", "I'm going to bed", "remind me to drink water"),
    ("Always remind me to stretch when I finish class", "I finish class", "remind me to stretch"),
    ("If I say I'm hungry, always tell me to drink some water first", "I'm hungry",
     "tell me to drink some water first"),
])
def test_creating_rules(rules, text, trigger, action):
    book, cmds = rules
    handled, said = say(cmds, text)
    assert handled and said[0].startswith("Got it. Whenever")
    [r] = book.all(REZA)
    assert (r.trigger, r.action) == (trigger, action)


def test_confirmation_speaks_it_back_in_second_person(rules):
    _, cmds = rules
    _, said = say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    assert said == ["Got it. Whenever you're leaving your room, I'll remind you to take your keys."]


@pytest.mark.parametrize("action,spoken", [
    ("remind me to take my keys", "Don't forget to take your keys."),
    ("remind me about my umbrella", "Don't forget your umbrella."),
    ("tell me to drink water", "Drink water."),
    ("say good luck", "Good luck."),
    ("make sure I lock the door", "Reminder: make sure you lock the door."),
])
def test_spoken_action(action, spoken):
    assert Rule(1, "r", "x", action).spoken_action() == spoken


def test_fires_on_variations_not_on_unrelated_or_guarded(rules):
    book, cmds = rules
    say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    say(cmds, "Every time I'm going to bed, remind me to drink water")
    assert [r.trigger for r in book.check(REZA, "leaving now")] == ["I'm leaving my room"]
    assert [r.trigger for r in book.check(REZA, "going to bed now")] == ["I'm going to bed"]
    for text in ("what's the capital of France", "how do I leave a group chat?", "my roommate left the room",
                 "I'll leave in an hour", "I left my room yesterday"):
        assert book.check(REZA, text) == [], text
    assert book.check(MALI, "leaving now") == []          # rules are per person
    assert book.check(None, "leaving now") == []


def test_drifted_variations_are_dropped():
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    book = RuleBook(db, FakeEmbedder(), background=False,
                    paraphrase=lambda t: ["leaving my room now", "I've had enough for now"])
    book.add(REZA, "I'm leaving my room", "remind me to take my keys")
    texts = [r[0] for r in db.query("SELECT text FROM rule_phrases WHERE source = 'generated'")]
    assert texts == ["leaving my room now"]


def test_that_was_wrong_teaches_a_not_this_example(rules):
    book, cmds = rules
    say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    assert book.check(REZA, "leaving now")
    handled, said = say(cmds, "No, that was wrong")
    assert handled and "won't remind you for that again" in said[0]
    assert book.check(REZA, "leaving now") == []
    assert book.check(REZA, "leaving room heading out")        # other phrasings still work


def test_wrong_outside_the_feedback_window_is_not_a_rule_correction(rules):
    book, cmds = rules
    say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    book.check(REZA, "leaving now")
    rid, utt, _ = book.last_fired
    book.last_fired = (rid, utt, time.monotonic() - 3600)
    assert say(cmds, "that was wrong")[0] is False


def test_adding_your_own_phrase(rules):
    book, cmds = rules
    say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    assert book.check(REZA, "ciao bella") == []
    handled, said = say(cmds, "Also remind me when I say ciao bella")
    assert handled and said[0] == 'Got it. "ciao bella" counts too.'
    assert book.check(REZA, "ciao bella")


def test_list_delete_and_edit(rules):
    book, cmds = rules
    say(cmds, "Whenever I say I'm leaving my room, remind me to take my keys")
    _, said = say(cmds, "What rules do I have?")
    assert said == ["You have 1: whenever you're leaving your room, I'll remind you to take your keys."]
    _, said = say(cmds, "Change the keys rule to also include my wallet", replies=["yes"])
    assert said[0] == ("So: whenever you're leaving your room, I'll remind you to take your keys and "
                       "your wallet. Is that right?")
    assert book.all(REZA)[0].spoken_action() == "Don't forget to take your keys and your wallet."
    say(cmds, "Stop reminding me about my keys", replies=["no"])
    assert book.all(REZA)
    say(cmds, "Stop reminding me about my keys", replies=["yes"])
    assert book.all(REZA) == []


def test_deleting_needs_a_verified_speaker(rules):
    book, _ = rules
    blocked = RuleCommands(book, verified=lambda d, u: False)
    blocked.handle(Heard("Whenever I say I'm leaving my room, remind me to take my keys"), Dialog(), REZA)
    blocked.handle(Heard("stop reminding me about my keys"), Dialog(["yes"]), REZA)
    assert book.all(REZA)


def test_strangers_cannot_make_rules(rules):
    book, cmds = rules
    handled, said = say(cmds, "Whenever I say hello, tell me a joke", user=None)
    assert handled and "people I recognise" in said[0]


@pytest.mark.parametrize("text", ["What's the weather?", "Remind me to call mum", "when is my exam",
                                  "always be nice", "I always forget my keys"])
def test_ordinary_sentences_are_not_rule_commands(rules, text):
    _, cmds = rules
    assert say(cmds, text)[0] is False
