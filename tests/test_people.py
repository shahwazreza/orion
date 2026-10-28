"""Onboarding and people commands, driven by a scripted dialog."""

import numpy as np
import pytest

from core.dialog import Heard, yes_no
from core.identity import Identity
from core.people import People, extract_name
from storage.users import UserStore


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class ScriptedDialog:
    """Answers each ask() with the next scripted reply. Replies are (text, seconds)."""

    def __init__(self, replies, hears_voice=True):
        self.replies = list(replies)
        self.hears_voice = hears_voice
        self.said = []

    def say(self, text):
        self.said.append(text)

    def ask(self, prompt, timeout_s=None):
        self.said.append(prompt)
        if not self.replies:
            return None
        text, secs = self.replies.pop(0) if isinstance(self.replies[0], tuple) else (self.replies.pop(0), 1.0)
        if text is None:
            return None
        audio = np.zeros(int(secs * 16000), dtype=np.int16) if self.hears_voice else None
        return Heard(text, audio)


class Encoder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, audio, sample_rate=16000):
        return self.vec


def setup(vec=unit(1, 0, 0)):
    store = UserStore(None)
    identity = Identity(store, Encoder(vec), threshold=0.6)
    return store, identity, People(store, identity, enroll_target_s=6.0)


@pytest.mark.parametrize("text,name", [
    ("My name is reza.", "Reza"), ("I'm Sam", "Sam"), ("Hi, I'm Mary Jane.", "Mary Jane"),
    ("Reza", "Reza"), ("um, it's Ali, nice to meet you", "Ali"), ("call me Jo", "Jo"),
    ("Hey Ryan, my name is Reza", "Reza"), ("Hey Orion, I'm Reza.", "Reza"),
    ("Hello there Orion, my name's Sam Lee", "Sam Lee"),
    ("Yes", None), ("What's the weather like in London today", None), ("", None),
])
def test_extract_name(text, name):
    assert extract_name(text) == name


@pytest.mark.parametrize("text,expected", [
    ("Yes", True), ("yeah, go ahead", True), ("That's right.", True), ("Sure!", True),
    ("No", False), ("Nope, it's Sam", False), ("yes, no wait", False), ("Hmm", None),
    ("I reckon", True), ("you got it right the first time", True), ("spot on", True),
    ("that's not right", False),
])
def test_yes_no(text, expected):
    assert yes_no(text) is expected


def test_first_run_learns_name_and_voice():
    store, identity, people = setup()
    script = ["I'm Reza", "yes", "sure",
              ("I went to class and then the gym", 3.0), ("Pizza, obviously", 2.0),
              ("Japan, for the food", 2.5)]
    d = ScriptedDialog(script)
    people.greet_on_start(d)
    u = store.owner()
    assert u.name == "Reza" and u.has_voice and len(u.enrollment) == 3
    assert identity.current.id == u.id
    assert d.said[-1].startswith("Got it, Reza.")


def test_name_correction_and_declining_voice():
    store, _, people = setup()
    d = ScriptedDialog(["I'm Risa", "no", "Reza", "yes", "no thanks"])
    people.greet_on_start(d)
    u = store.owner()
    assert u.name == "Reza" and not u.has_voice and u.voice_opt_out
    # declined voice is respected on the next start
    d2 = ScriptedDialog([])
    people.greet_on_start(d2)
    assert d2.said == ["Orion online."]


def test_text_mode_first_run_skips_voice():
    store, _, people = setup()
    people.greet_on_start(ScriptedDialog(["Reza", "yes"], hears_voice=False))
    assert store.owner().name == "Reza" and not store.owner().has_voice


def test_too_little_speech_does_not_save_a_voiceprint():
    store, _, people = setup()
    d = ScriptedDialog(["Reza", "yes", "yes", ("hi", 0.5), ("ok", 0.6), None, None, None])
    people.greet_on_start(d)
    assert not store.owner().has_voice


def test_only_owner_can_add_people():
    store, identity, people = setup(unit(0, 1, 0))
    owner = store.add("Reza", [unit(1, 0, 0)])
    identity.set_current(None)
    d = ScriptedDialog([])
    assert people.handle(Heard("add a new person"), d)
    assert "Only the owner" in d.said[0] and len(store.all()) == 1

    identity.set_current(owner)
    d = ScriptedDialog(["My name is Sam", "yes", "yes", ("one two three", 3.0), ("four five six", 3.0)])
    assert people.handle(Heard("Orion, add a new person"), d)
    sam = store.find("Sam")
    assert sam is not None and sam.role == "member" and sam.has_voice
    assert identity.current.id == sam.id


@pytest.mark.parametrize("text", ["Orion, add another voice", "add a new person", "add someone",
                                  "learn a new voice", "add another person please", "add a new member"])
def test_add_person_phrasings(text):
    store, identity, people = setup()
    identity.set_current(None)
    d = ScriptedDialog([])
    assert people.handle(Heard(text), d) and "Only the owner" in d.said[0]


def test_new_voice_that_matches_someone_else_needs_confirmation():
    store, identity, people = setup(unit(1, 0, 0))     # new person sounds exactly like Reza
    owner = store.add("Reza", [unit(1, 0, 0)])
    identity.set_current(owner)
    d = ScriptedDialog(["Sam", "yes", "yes", ("a b c", 3.0), ("d e f", 3.0), "no"])
    people.handle(Heard("add a new person"), d)
    assert any("sounds a lot like Reza" in s for s in d.said)
    assert not store.find("Sam").has_voice


def test_who_am_i_and_who_do_you_know():
    store, identity, people = setup()
    reza = store.add("Reza")
    store.add("Sam")
    d = ScriptedDialog([])
    identity.set_current(reza)
    people.handle(Heard("Who am I?"), d)
    people.handle(Heard("who do you know"), d)
    assert d.said == ["You're Reza.", "I know Reza and Sam."]
    identity.set_current(None)            # strangers don't get the list of names
    people.handle(Heard("who do you know"), d)
    assert d.said[-1] == "I know 2 people."


def test_rename_and_delete_need_confirmation():
    store, identity, people = setup()
    u = store.add("Reza")
    identity.set_current(u)
    people.handle(Heard("call me Rez"), ScriptedDialog(["no"]))
    assert store.get(u.id).name == "Reza"
    people.handle(Heard("call me Rez"), ScriptedDialog(["yes"]))
    assert store.get(u.id).name == "Rez"
    people.handle(Heard("delete my profile"), ScriptedDialog(["no"]))
    assert store.get(u.id) is not None
    people.handle(Heard("delete my profile"), ScriptedDialog(["yes"]))
    assert store.get(u.id) is None and identity.current is None


def test_ordinary_questions_are_not_commands():
    _, _, people = setup()
    for text in ("what does my name mean", "who is the prime minister", "tell me about voices"):
        assert not people.handle(Heard(text), ScriptedDialog([]))


def test_one_silence_gets_a_second_chance_two_give_up():
    store, _, people = setup()
    d = ScriptedDialog([None, "Reza", "yes", "no"])      # silent once, then answers
    people.greet_on_start(d)
    assert store.owner().name == "Reza"
    assert "didn't hear anything" in " ".join(d.said)
    store2, _, people2 = setup()
    people2.greet_on_start(ScriptedDialog([None, None]))
    assert store2.is_empty()


def test_my_name_is_with_nobody_set_up_starts_introductions():
    store, identity, people = setup()
    d = ScriptedDialog(["yes", "no"])                    # confirm name, decline voice
    assert people.handle(Heard("My name is Reza."), d)
    assert store.owner().name == "Reza" and identity.current.name == "Reza"
    assert not any("ask the owner" in s for s in d.said)


def test_reply_that_repeats_the_suggested_name_confirms_it():
    # The live session that failed: "I reckon" then "Jimble, you got it right the first time".
    store, _, people = setup()
    store.add("Reza")
    d = ScriptedDialog(["I tumbled ya on", "Jimble", "hmm", "Jimble, you got it right the first time", "no"])
    people.meet(d)
    assert store.find("Jimble") is not None


def test_short_answers_get_one_nudge_then_count():
    store, _, people = setup()
    u = store.add("Reza")
    # "Biryani." (1 s) -> nudge -> longer answer; total 1 + 4 + 4 = 9 s >= 6 s target
    d = ScriptedDialog([("Biryani.", 1.0), ("Chicken biryani with raita, every week", 4.0),
                        ("Japan, for the trains and the food", 4.0)])
    assert people.enroll_voice(d, u)
    assert "Could you say a bit more about that?" in d.said
    assert len(store.get(u.id).enrollment) == 3


def test_owner_can_teach_someone_elses_voice():
    store, identity, people = setup(unit(0, 1, 0))
    owner = store.add("Reza", [unit(1, 0, 0)])
    joe = store.add("Joe")
    identity.set_current(owner)
    d = ScriptedDialog([("I like grapes and crab and shrimp", 3.5), ("My room is a filthy mess", 3.5)])
    assert people.handle(Heard("learn Joe's voice"), d)
    assert store.get(joe.id).has_voice and identity.current.id == joe.id


def test_only_owner_can_teach_someone_elses_voice_and_unknown_name():
    store, identity, people = setup()
    owner = store.add("Reza")
    joe = store.add("Joe")
    identity.set_current(joe)
    d = ScriptedDialog([])
    people.handle(Heard("learn Reza's voice"), d)
    assert d.said == ["Only the owner can do that."]
    identity.set_current(owner)
    people.handle(Heard("learn Sam's voice"), d)
    assert d.said[-1] == "I don't know anyone called Sam."
