"""Router, Claude mode, and the privacy guard, with a mocked Anthropic API (nothing billed)."""

import json
import secrets
from types import SimpleNamespace

import pytest

from audio.tts import Speaker
from brain.claude import ClaudeBackend
from brain.llm import LLMError
from core.assistant import Assistant
from core.memory import MemoryStore
from core.metrics import MetricsSink, TurnMetrics
from core.router import CLOUD, LOCAL, Router
from storage.audit import AuditLog
from storage.db import Database, derive_db_key
from storage.users import User
from tests.test_memory import FakeEmbedder

REZA = User(id="r", name="Reza", role="owner")


# ---- router decisions ---------------------------------------------------------------
@pytest.mark.parametrize("text,route,q", [
    ("Ask Claude what the tallest mountain on Mars is", CLOUD, "what the tallest mountain on Mars is"),
    ("Ask Claude whether octopuses dream", CLOUD, "octopuses dream"),
    ("Ask Claude why the sky is blue", CLOUD, "why the sky is blue"),
    ("ask claude: explain quantum tunnelling", CLOUD, "explain quantum tunnelling"),
    ("Can you ask Claude to write a haiku about rain?", CLOUD, "write a haiku about rain?"),
    ("What's the tallest mountain on Mars?", LOCAL, "What's the tallest mountain on Mars?"),
    ("Tell me about Claude Monet", LOCAL, "Tell me about Claude Monet"),
])
def test_one_off_ask(text, route, q):
    d = Router().decide(text, "r")
    assert d.route == route and d.text == q


def test_claude_mode_on_and_off_per_person():
    r = Router()
    on = r.decide("Switch to Claude mode", "r")
    assert on.handled and "Claude mode" in on.announce and r.in_claude_mode("r")
    assert r.decide("what is entropy", "r").route == CLOUD
    assert r.decide("what is entropy", "m").route == LOCAL          # Malihah isn't in Claude mode
    off = r.decide("back to local", "r")
    assert off.handled and off.announce.startswith("Back to local") and not r.in_claude_mode("r")
    assert r.decide("what is entropy", "r").route == LOCAL


def test_no_key_or_audit_means_no_cloud():
    r = Router(cloud_available=False)
    assert r.decide("switch to Claude mode", "r").route == LOCAL
    d = r.decide("ask Claude what entropy is", "r")
    assert d.route == LOCAL and "answer myself" in d.announce


def test_include_is_explicit_consent_for_that_request_only():
    d = Router().decide("Ask Claude to plan my week, and include what you know about my schedule", "r")
    assert d.route == CLOUD and d.include_personal and d.include_topic == "my schedule"
    assert d.text == "plan my week"
    assert not Router().decide("Ask Claude to plan my week", "r").include_personal


@pytest.mark.parametrize("text,route", [
    ("What causes the northern lights?", CLOUD), ("Explain how vaccines work", CLOUD),
    ("What did I eat yesterday?", LOCAL), ("What's my sister's name?", LOCAL),
    ("Remind me to call mum", LOCAL), ("How are you?", LOCAL), ("What should I do today?", LOCAL),
])
def test_auto_router_only_sends_impersonal_general_knowledge(text, route):
    assert Router(auto=True).decide(text, "r").route == route
    assert Router(auto=False).decide(text, "r").route == LOCAL      # off by default


# ---- Claude backend + assistant with a fake Anthropic client ---------------------------
class FakeStream:
    def __init__(self, parts, stop="end_turn"):
        self.parts, self.stop = parts, stop

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        yield from self.parts

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop, model="claude-opus-5-5", content=[],
                               usage=SimpleNamespace(input_tokens=420, output_tokens=60))


class FakeClient:
    def __init__(self, parts=("Olympus Mons, ", "about 22 kilometres high."), error=None, stop="end_turn"):
        self.calls, self.parts, self.error, self.stop = [], parts, error, stop
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **body):
        self.calls.append(body)
        if self.error:
            raise self.error
        return FakeStream(self.parts, self.stop)


class LocalLLM:
    model = "llama3.2:3b"

    def __init__(self):
        self.calls = 0

    def stream_chat(self, messages):
        self.calls += 1
        yield "Local answer."


@pytest.fixture
def setup(tmp_path):
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    memory = MemoryStore(db, FakeEmbedder())
    memory.add(REZA, "my schedule: lectures Monday")
    memory.add(REZA, "my sister's name is Sara")
    said = []
    spk = Speaker(None, None, echo=said.append)
    local = LocalLLM()
    a = Assistant(local, spk, MetricsSink(None, False), "test", memory=memory)
    client = FakeClient()
    a.cloud = ClaudeBackend(client=client)
    a.audit = AuditLog(db)
    yield a, client, local, said, AuditLog(db)
    spk.close()


def ask(a, text, decision=None, session=False):
    decision = decision or Router().decide(f"ask Claude {text}", "r")
    return a.respond_cloud(decision.text, TurnMetrics("text"), REZA, decision, session=session)


def test_cloud_request_contains_nothing_personal(setup):
    a, client, local, said, audit = setup
    reply = ask(a, "what's the tallest mountain on Mars")
    assert reply == "Olympus Mons, about 22 kilometres high." and local.calls == 0
    body = json.dumps(client.calls[0])
    assert "Reza" not in body and "Sara" not in body and "lectures" not in body
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert client.calls[0]["output_config"] == {"effort": "low"}
    assert client.calls[0]["fallbacks"] == "default"


def test_include_sends_only_what_was_allowed(setup):
    a, client, *_ = setup
    d = Router().decide("Ask Claude to plan my week, and include what you know about my schedule", "r")
    a.respond_cloud(d.text, TurnMetrics("text"), REZA, d, session=False)
    body = json.dumps(client.calls[0])
    assert "lectures Monday" in body and "Reza" in body
    assert "Sara" not in body                                     # unrelated memory stays home


def test_every_cloud_request_is_audited_exactly_as_sent(setup):
    a, client, _, _, audit = setup
    ask(a, "what's the tallest mountain on Mars")
    [(ts, kind, d)] = audit.recent(kind="cloud_request")
    assert d["body"]["messages"] == client.calls[0]["messages"]
    assert d["usage"]["input_tokens"] == 420 and d["usage"]["cost_usd"] > 0
    assert d["included_personal"] is False
    n, cost = audit.month_cost()
    assert n == 1 and cost == pytest.approx(420 * 4 / 1e6 + 60 * 20 / 1e6)


def test_api_failure_is_spoken_and_falls_back_to_local(setup):
    a, client, local, said, audit = setup
    import anthropic, httpx2 as httpx

    client.error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    reply = ask(a, "what's entropy")
    assert reply == "Local answer." and local.calls == 1
    assert any("can't reach Claude" in s and "answer this one myself" in s for s in said)
    assert audit.recent(kind="cloud_request")[0][2]["error"]


def test_missing_key_is_a_spoken_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("brain.claude.api_key", lambda: None)
    with pytest.raises(LLMError) as exc:
        list(ClaudeBackend().stream_chat([{"role": "user", "content": "hi"}]))
    assert "API key" in exc.value.spoken


def test_refusal_is_said_plainly(setup):
    a, client, *_ = setup
    client.parts, client.stop = (), "refusal"
    assert ask(a, "something declined").strip() == "I'm not able to help with that one."


def test_claude_mode_keeps_its_own_history_and_one_offs_dont(setup):
    a, client, *_ = setup
    r = Router()
    r.decide("switch to Claude mode", "r")
    for q in ("first question", "second question"):
        d = r.decide(q, "r")
        a.respond_cloud(d.text, TurnMetrics("text"), REZA, d, session=True)
    assert [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"] == \
           ["first question", "second question"]
    ask(a, "a one-off")
    assert [m["content"] for m in client.calls[2]["messages"] if m["role"] == "user"] == ["a one-off"]
    a.end_cloud_session(REZA)
    d = r.decide("third question", "r")
    a.respond_cloud(d.text, TurnMetrics("text"), REZA, d, session=True)
    assert [m["content"] for m in client.calls[3]["messages"] if m["role"] == "user"] == ["third question"]


def test_no_audit_log_means_no_cloud(setup):
    a, client, local, said, _ = setup
    a.audit = None
    assert ask(a, "anything") == "Local answer." and client.calls == []


@pytest.mark.parametrize("heard", ["Cloud Mode on.", "Switch to cloud mode", "clod mode on", "Claude mode, please"])
def test_claude_mode_on_as_whisper_hears_it(heard):
    r = Router()
    assert r.decide(heard, "r").handled and r.in_claude_mode("r")


@pytest.mark.parametrize("heard", ["Cloud Mode off!", "Clod mode off.", "back to local", "turn off cloud mode"])
def test_claude_mode_off_as_whisper_hears_it(heard):
    # Live log: stuck in Claude mode because "Cloud Mode off" wasn't recognised.
    r = Router()
    r.decide("switch to Claude mode", "r")
    d = r.decide(heard, "r")
    assert d.handled and not r.in_claude_mode("r") and d.route == LOCAL


def test_ask_cloud_is_ask_claude_but_cloud_topics_are_not():
    assert Router().decide("Ask cloud what octopuses eat", "r").route == CLOUD
    assert Router().decide("Tell me about cloud computing", "r").route == LOCAL
