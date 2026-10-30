"""The turn pipeline and the Ollama client, with fake backends: no models, no audio."""

import json

import httpx
import pytest

from audio.tts import Speaker
from brain.llm import LLMError, OllamaBackend
from brain.profanity import ProfanityPolicy
from core.assistant import Assistant
from core.metrics import MetricsSink, TurnMetrics
from storage.encrypted_file import EncryptedLines

KEY = bytes(range(32))
from storage.users import User


class FakeLLM:
    name = "fake"

    def __init__(self, pieces=None, error=None):
        self.pieces = pieces or []
        self.error = error
        self.seen = None

    def stream_chat(self, messages):
        self.seen = messages
        yield from self.pieces
        if self.error:
            raise self.error

    def warm_up(self):
        pass


@pytest.fixture
def spoken():
    return []


@pytest.fixture
def make_assistant(tmp_path, spoken):
    speakers = []

    def _make(llm):
        speaker = Speaker(tts=None, player=None, echo=spoken.append)
        speakers.append(speaker)
        metrics = MetricsSink(EncryptedLines(tmp_path / "m.enc", KEY, "metrics"))
        # Language mode off here: these tests are about history, not few-shot examples.
        return Assistant(llm, speaker, metrics, "test", max_history_turns=2,
                         profanity=ProfanityPolicy("off"))

    yield _make
    for s in speakers:
        s.close()


def test_reply_is_spoken_sentence_by_sentence(make_assistant, spoken):
    a = make_assistant(FakeLLM(["Hello", " Reza.", " The capital", " is **Canberra**."]))
    reply = a.respond("capital of Australia?", TurnMetrics("text"))
    assert reply == "Hello Reza. The capital is **Canberra**."
    assert spoken == ["Hello Reza.", "The capital is Canberra."]
    assert a.last_metrics["ok"] is True
    assert a.last_metrics["llm_first_token_s"] is not None


def test_history_is_kept_and_bounded(make_assistant):
    llm = FakeLLM(["ok."])
    a = make_assistant(llm)
    for i in range(4):
        a.respond(f"q{i}", TurnMetrics("text"))
    roles = [m["role"] for m in llm.seen]
    assert roles[0] == "system"
    assert len(llm.seen) == 1 + 2 * 2 + 1          # system + 2 kept turns + new question
    assert llm.seen[-1]["content"] == "q3"


def test_llm_failure_is_spoken_not_raised(make_assistant, spoken):
    a = make_assistant(FakeLLM(error=LLMError("I can't reach Ollama. Is it running?")))
    assert a.respond("hi", TurnMetrics("text")) is None
    assert spoken == ["Sorry. I can't reach Ollama. Is it running?"]
    assert a.conversation_for(None).messages_with("x")[1:] == [{"role": "user", "content": "x"}]
    assert a.last_metrics["ok"] is False


def test_each_person_gets_their_own_history_and_name(make_assistant, spoken):
    llm = FakeLLM(["ok."])
    a = make_assistant(llm)
    sam, ali = User(id="s1", name="Sam"), User(id="a1", name="Ali")
    a.respond("sam's question", TurnMetrics("text"), sam)
    a.respond("ali's question", TurnMetrics("text"), ali)
    assert "Ali" in llm.seen[0]["content"] and "Sam" not in llm.seen[0]["content"]
    assert [m["content"] for m in llm.seen[1:]] == ["ali's question"]   # no leak of Sam's chat
    a.respond("again", TurnMetrics("text"), sam)
    assert [m["content"] for m in llm.seen[1:]] == ["sam's question", "ok.", "again"]


def test_unknown_speaker_gets_no_name_and_guest_history_is_dropped(make_assistant):
    llm = FakeLLM(["ok."])
    a = make_assistant(llm)
    a.respond("hi", TurnMetrics("text"), None)
    assert "don't know who is speaking" in llm.seen[0]["content"]
    a.forget_guest()
    a.respond("hi again", TurnMetrics("text"), None)
    assert len(llm.seen) == 2


def test_metrics_are_encrypted_and_never_contain_conversation_text(make_assistant, tmp_path):
    a = make_assistant(FakeLLM(["My secret reply."]))
    a.respond("my private question", TurnMetrics("text"))
    raw = (tmp_path / "m.enc").read_bytes()
    assert b"mode" not in raw and b"text" not in raw                # encrypted on disk
    rows = EncryptedLines(tmp_path / "m.enc", KEY, "metrics").read()
    assert rows[0]["mode"] == "text"
    assert "secret" not in json.dumps(rows) and "private" not in json.dumps(rows)


def _ollama(handler):
    return OllamaBackend("http://127.0.0.1:11434", "llama3.2:3b",
                         client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_streams_content():
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True and body["model"] == "llama3.2:3b"
        assert "think" not in body
        lines = [{"message": {"content": "Hi"}}, {"message": {"content": " Reza."}}, {"done": True}]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    assert "".join(_ollama(handler).stream_chat([])) == "Hi Reza."


def test_ollama_missing_model_is_explained():
    with pytest.raises(LLMError) as exc:
        list(_ollama(lambda r: httpx.Response(404, text="model not found")).stream_chat([]))
    assert "isn't installed" in exc.value.spoken


def test_ollama_down_is_explained():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError) as exc:
        list(_ollama(handler).stream_chat([]))
    assert "can't reach Ollama" in exc.value.spoken


def test_ollama_refuses_remote_host():
    with pytest.raises(Exception):
        OllamaBackend("http://10.0.0.2:11434", "m")


def test_style_examples_are_marked_as_not_real():
    from brain.llm import Conversation
    c = Conversation("SYS")
    c.examples = [{"role": "user", "content": "holy shit I got the internship"},
                  {"role": "assistant", "content": "Well done!"}]
    msgs = c.messages_with("what internship?")
    note = msgs[3]
    assert note["role"] == "system" and "made-up examples" in note["content"]
    assert msgs[-1] == {"role": "user", "content": "what internship?"}
