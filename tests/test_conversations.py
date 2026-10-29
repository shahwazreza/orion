"""Conversation memory: saving, restoring, recall by day and meaning, deleting, privacy."""

import secrets
from datetime import date, datetime, timedelta
from unittest import mock

import pytest

from core.conversations import ConversationCommands, ConversationLog, parse_when, spoken_day
from core.dialog import Heard
from core.memory import MemoryCommands, MemoryStore
from storage.db import Database, derive_db_key
from storage.users import User
from tests.test_memory import Dialog, FakeEmbedder

REZA, MALI = User(id="r", name="Reza", role="owner"), User(id="m", name="Malihah")
TODAY = date(2026, 10, 29)


def at(d: date, hour=12):
    return datetime(d.year, d.month, d.day, hour).timestamp()


@pytest.fixture
def convo():
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    summaries = []

    def summarize(name, msgs):
        summaries.append(len(msgs))
        return f"You talked about {msgs[0].text.lower().rstrip('?')}."

    log = ConversationLog(db, FakeEmbedder(), summarize=summarize, background=False)
    return log, ConversationCommands(log, verified=lambda d, u: True), summaries, db


def add(log, user, q, r, when):
    with mock.patch("core.conversations.now", return_value=when):
        log.add_exchange(user, q, r)


def test_exchanges_are_saved_per_person_and_strangers_never(convo):
    log, _, _, db = convo
    log.add_exchange(REZA, "How far is the moon?", "About 384,000 km.")
    log.add_exchange(None, "secret question", "secret answer")
    assert [m.role for m in log.recent(REZA)] == ["user", "assistant"]
    assert log.recent(MALI) == []
    assert db.query("SELECT count(*) FROM messages")[0][0] == 2


def test_recent_restores_the_last_turns_of_the_past_day(convo):
    log, _, _, _ = convo
    old = datetime.now().timestamp() - 3 * 86400
    add(log, REZA, "an old question", "old answer", old)
    for i in range(12):
        log.add_exchange(REZA, f"question {i}", f"answer {i}")
    msgs = log.recent(REZA, turns=8)
    assert len(msgs) == 16 and msgs[0].text == "question 4" and msgs[-1].text == "answer 11"


def test_what_did_we_talk_about_yesterday_uses_the_daily_summary(convo):
    log, cmds, summaries, _ = convo
    add(log, REZA, "Tell me about black holes?", "They're regions of spacetime...", at(TODAY - timedelta(days=1)))
    d = Dialog()
    cmds.handle(Heard("What did we talk about yesterday?"), d, REZA, today=TODAY)
    assert d.said == ["You talked about tell me about black holes."]
    assert log.summary(REZA, (TODAY - timedelta(days=1)).isoformat())   # stored for next time
    d = Dialog()
    cmds.handle(Heard("what did we talk about on Monday"), d, REZA, today=TODAY)
    assert d.said[0].startswith("We didn't talk on Monday")


def test_recall_by_meaning(convo):
    log, cmds, _, _ = convo
    add(log, REZA, "My exam is on Friday and I'm nervous about calculus", "You'll do fine.",
        at(TODAY - timedelta(days=2)))
    d = Dialog()
    cmds.handle(Heard("What did I say about my calculus exam?"), d, REZA, today=TODAY)
    assert "you said: My exam is on Friday" in d.said[0] and "And I said: You'll do fine" in d.said[0]
    d = Dialog()
    cmds.handle(Heard("Did I ever mention Jupiter?"), d, REZA, today=TODAY)
    assert d.said[0].startswith("I don't remember us talking about Jupiter")


def test_memory_question_falls_back_to_conversations(convo):
    log, cmds, _, db = convo
    add(log, REZA, "My exam is on Friday and I'm nervous about calculus", "You'll do fine.",
        at(TODAY - timedelta(days=2)))
    mem = MemoryCommands(MemoryStore(db, FakeEmbedder()), verified=lambda d, u: True,
                         recall_conversation=cmds.recall_topic)
    d = Dialog()
    mem.handle(Heard("What did I tell you about my calculus exam?"), d, REZA)
    assert "you said: My exam is on Friday" in d.said[0]


def test_dont_save_this_conversation_drops_it_and_stops_saving(convo):
    log, cmds, _, _ = convo
    log.add_exchange(REZA, "kept from before", "ok")
    log.start_conversation(REZA)
    log.add_exchange(REZA, "private thing", "ok")
    cmds.handle(Heard("Don't save this conversation"), Dialog(), REZA)
    log.add_exchange(REZA, "another private thing", "ok")
    assert [m.text for m in log.recent(REZA)] == ["kept from before", "ok"]
    log.start_conversation(REZA)                             # next conversation saves again
    log.add_exchange(REZA, "new topic", "ok")
    assert log.recent(REZA)[-2].text == "new topic"


def test_forget_today_and_delete_all_need_confirmation(convo):
    log, cmds, _, _ = convo
    add(log, REZA, "yesterday's thing", "ok", at(TODAY - timedelta(days=1)))
    add(log, REZA, "today's thing", "ok", at(TODAY))
    cmds.handle(Heard("Forget today's conversation"), Dialog(["no"]), REZA, today=TODAY)
    assert len(log.on_day(REZA, TODAY.isoformat())) == 2
    cmds.handle(Heard("Forget today's conversation"), Dialog(["yes"]), REZA, today=TODAY)
    assert log.on_day(REZA, TODAY.isoformat()) == []
    assert len(log.on_day(REZA, (TODAY - timedelta(days=1)).isoformat())) == 2
    cmds.handle(Heard("Delete my conversation history"), Dialog(["yes"]), REZA, today=TODAY)
    assert log.on_day(REZA, (TODAY - timedelta(days=1)).isoformat()) == []


def test_deletes_need_a_verified_speaker(convo):
    log, _, _, _ = convo
    log.add_exchange(REZA, "q", "a")
    blocked = ConversationCommands(log, verified=lambda d, u: False)
    blocked.handle(Heard("delete my conversation history"), Dialog(["yes"]), REZA)
    assert len(log.recent(REZA)) == 2


def test_past_day_summaries_are_written_once(convo):
    log, _, summaries, _ = convo
    for i in (1, 2):
        add(log, REZA, f"day {i} question", "a", at(date.today() - timedelta(days=i)))
    log.add_exchange(REZA, "today's question", "a")
    log.summarize_past_days([REZA])
    log.summarize_past_days([REZA])
    assert len(summaries) == 2                               # today isn't summarised yet


def test_retention(convo):
    log, _, _, _ = convo
    add(log, REZA, "ancient", "a", datetime.now().timestamp() - 400 * 86400)
    log.add_exchange(REZA, "recent", "a")
    assert log.apply_retention(None) == 0
    assert log.apply_retention(365) == 2
    assert [m.text for m in log.recent(REZA)] == ["recent", "a"]


def test_context_for_local_model_skips_the_last_hour(convo):
    log, _, _, _ = convo
    add(log, REZA, "My exam is on Friday and I'm nervous about calculus", "You'll do fine.",
        at(TODAY - timedelta(days=2)))
    log.add_exchange(REZA, "calculus exam tips please", "Practise past papers.")
    ctx = log.context_for(REZA, "I'm nervous about my calculus exam")
    assert "nervous about calculus" in ctx and "Practise past papers" not in ctx


@pytest.mark.parametrize("text,days", [("", 1), ("today", 1), (" yesterday", 1), (" last week", 7),
                                       (" on tuesday", 1), (" about the moon", None)])
def test_parse_when(text, days):
    out = parse_when(text, TODAY)
    assert (out is None) if days is None else len(out) == days


def test_spoken_day():
    assert spoken_day(TODAY, TODAY) == "today"
    assert spoken_day(TODAY - timedelta(days=1), TODAY) == "yesterday"
    assert spoken_day(TODAY - timedelta(days=3), TODAY) == "on Monday"
    assert spoken_day(date(2026, 10, 1), TODAY) == "on Thursday the 1st of October"


def test_unindexed_messages_from_last_session_are_indexed_at_startup(convo):
    _, _, _, db = convo
    db.execute("INSERT INTO messages (user_id, ts, role, text) VALUES ('r', 1.0, 'user', 'my calculus exam')")
    db.execute("INSERT INTO messages (user_id, ts, role, text) VALUES ('r', 1.0, 'assistant', 'good luck')")
    log = ConversationLog(db, FakeEmbedder(), background=False)
    assert db.query("SELECT count(*) FROM messages WHERE role='user' AND embedding IS NULL")[0][0] == 0
    assert log.search(REZA, "calculus exam")


def test_close_finishes_pending_indexing():
    db = Database(None, derive_db_key(secrets.token_bytes(32)))
    log = ConversationLog(db, FakeEmbedder(), background=True)
    for i in range(20):
        log.add_exchange(REZA, f"question about topic {i}", "answer")
    log.close()
    assert db.query("SELECT count(*) FROM messages WHERE role='user' AND embedding IS NULL")[0][0] == 0
