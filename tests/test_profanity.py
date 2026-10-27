"""Profanity: on by default, clean mode reasons, the scrub filter, and the prompt."""

import pytest

from audio.tts import Speaker
from brain.persona import system_prompt
from brain.profanity import (DEMO, MEETING, ProfanityPolicy, contains_profanity, handle_command,
                             scrub)
from core.assistant import Assistant
from core.config import load_config
from core.hardware import Probe
from core.metrics import MetricsSink, TurnMetrics
from storage.users import User


@pytest.mark.parametrize("text,expected", [
    ("That's fucking brilliant, Reza.", "That's brilliant, Reza."),
    ("Holy shit, well done!", "Wow, well done!"),
    ("Shit, I forgot my keys.", "Oh no, I forgot my keys."),
    ("What the hell is that?", "What on earth is that?"),
    ("My roommate is being a dick about the dishes.", "My roommate is being a pain about the dishes."),
    ("This f*** code.", "This code."),
    ("That was a shitty day.", "That was a rough day."),
])
def test_scrub(text, expected):
    assert scrub(text) == expected
    assert not contains_profanity(scrub(text))


@pytest.mark.parametrize("text", [
    "Hello there, the prickly pear is lovely.", "Assess the class schedule.", "Scrap that idea.",
    "Shell scripts and cocktails.", "Use **bold** sparingly.", "Dickens wrote Hard Times.",
])
def test_scrub_leaves_innocent_words_alone(text):
    assert not contains_profanity(text)
    assert scrub(text) == text


def test_policy_reasons_and_commands():
    p = ProfanityPolicy("on")
    assert p.allowed()
    assert handle_command("Orion, keep it clean please", p).startswith("Understood")
    assert not p.allowed()
    assert handle_command("okay, back to normal", p) == "Back to normal."
    assert p.allowed()
    assert handle_command("what's the weather", p) is None


def test_back_to_normal_cannot_override_demo_meeting_or_config():
    p = ProfanityPolicy("on")
    p.keep_clean(DEMO)
    handle_command("keep it clean", p)
    assert "Demo mode" in handle_command("back to normal", p) and not p.allowed()
    p = ProfanityPolicy("on")
    p.keep_clean(MEETING)
    assert "meeting notes" in handle_command("back to normal", p) and not p.allowed()
    p.release(MEETING)
    assert p.allowed()
    p = ProfanityPolicy(False)          # YAML `profanity: off` arrives as a boolean
    assert not p.allowed()
    assert "settings" in handle_command("you can swear again", p)


def test_prompt_switches_between_casual_and_clean():
    casual, clean = system_prompt(user_name="Reza", profanity=True), system_prompt(user_name="Reza")
    assert "swear back" in casual and "never offended" in casual
    assert "no swearing" in clean and "lecture" in clean      # clean, but still no lecturing


class LLM:
    name = "fake"

    def __init__(self, pieces):
        self.pieces = pieces
        self.seen = None

    def stream_chat(self, messages):
        self.seen = messages
        yield from self.pieces

    def warm_up(self):
        pass


@pytest.fixture
def run_turn(tmp_path):
    speakers = []

    def _run(pieces, user, policy):
        said = []
        spk = Speaker(None, None, echo=said.append)
        speakers.append(spk)
        llm = LLM(pieces)
        a = Assistant(llm, spk, MetricsSink(None, False), "test", profanity=policy)
        reply = a.respond("hi", TurnMetrics("text"), user)
        _run.messages = llm.seen
        return said, reply, llm.seen[0]["content"]

    yield _run
    for s in speakers:
        s.close()


REZA = User(id="r", name="Reza")


def test_swearing_passes_through_for_known_user(run_turn):
    said, reply, prompt = run_turn(["Holy shit, well done!"], REZA, ProfanityPolicy("on"))
    assert said == ["Holy shit, well done!"] and "swear back" in prompt
    assert any("fucking wifi" in m["content"] for m in run_turn.messages)   # few-shot examples


def test_clean_mode_scrubs_speech_and_history(run_turn):
    p = ProfanityPolicy("on")
    p.keep_clean(DEMO)
    said, reply, prompt = run_turn(["Holy shit, well done!"], REZA, p)
    assert said == ["Wow, well done!"] and reply == "Wow, well done!"
    assert "no swearing" in prompt
    assert not any("fucking" in m["content"] for m in run_turn.messages[1:])  # no examples


def test_strangers_always_get_clean_language(run_turn):
    said, _, prompt = run_turn(["Holy shit, well done!"], None, ProfanityPolicy("on"))
    assert said == ["Wow, well done!"] and "no swearing" in prompt


def test_config_default_on_and_demo_overlay():
    laptop = Probe("x86_64", "")
    cfg = load_config(probe=laptop)
    assert ProfanityPolicy(cfg.get("assistant.profanity")).allowed()
    assert not cfg.demo
    demo = load_config(probe=laptop, demo=True)
    assert demo.demo and demo.get("conversation.follow_up_s") == 8.0
