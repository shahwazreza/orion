"""Requests for data Orion doesn't have get an honest fixed answer, not a guess."""

import pytest

from brain.capabilities import unsupported


@pytest.mark.parametrize("text", [
    "Tell me about the weather today.", "What's the weather like?", "Will it rain tomorrow?",
    "Do I need an umbrella?", "Read my emails", "Do I have any new messages?",
    "What's on my calendar tomorrow?", "Am I free at three?", "What's in the news?",
    "Give me today's headlines", "What's Apple's stock price?", "Set a timer for ten minutes",
    "Wake me up at seven", "Remind me to call mum tonight", "Who won last night's game?",
])
def test_live_or_personal_data_is_refused_honestly(text):
    reply = unsupported(text)
    assert reply and reply.startswith("I can't") and "won't guess" in reply


@pytest.mark.parametrize("text", [
    "How does weather form?", "How do I write a professional email?", "I have good news!",
    "Remind me what photosynthesis is", "Tell me about the Gregorian calendar",
    "What's the temperature of the sun?", "Explain stock markets simply", "What's a good name for a cat?",
    "Tell me about the history of Rome",
])
def test_ordinary_questions_still_go_to_the_model(text):
    assert unsupported(text) is None


def test_assistant_never_asks_the_model_about_weather():
    from audio.tts import Speaker
    from core.assistant import Assistant
    from core.metrics import MetricsSink, TurnMetrics

    class LLM:
        model = "fake"
        called = False

        def stream_chat(self, messages):
            LLM.called = True
            yield "A high of twenty-five degrees with scattered showers."

    said = []
    spk = Speaker(None, None, echo=said.append)
    a = Assistant(LLM(), spk, MetricsSink(None, False), "test")
    reply = a.respond("Tell me about the weather today.", TurnMetrics("voice"))
    spk.close()
    assert not LLM.called
    assert said == [reply] and "can't check the weather" in reply
