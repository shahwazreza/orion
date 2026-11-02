"""Phase 4 alarms, timers, reminders: parsing, commands, ringing, catching up, the systemd timer."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from brain.capabilities import unsupported
from core.alarms import (AlarmCommands, Ringer, reminder_label, ring_reply, ring_tone, timer_label)
from core.dialog import Heard
from core.schedule import ALARM, GUEST, REMINDER, TIMER, Schedule, WakeupTimer, remove_hook
from core.when import (next_on_days, parse_clock, parse_duration, parse_repeat, resolve_clock, say_duration)
from storage.db import Database
from storage.users import User

TZ = ZoneInfo("America/Chicago")
REZA = User(id="r", name="Reza", role="owner")
SAM = User(id="s", name="Sam")


@pytest.fixture(autouse=True)
def local_tz(monkeypatch):
    """Every 'local' time in these tests is Chicago, whatever this computer's zone."""
    import core.alarms
    import core.schedule
    import core.when

    for mod in (core.when, core.alarms, core.schedule):
        monkeypatch.setattr(mod, "local_zone", lambda: TZ)


class Clock:
    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def ts(self) -> float:
        return self.t.timestamp()

    def advance(self, **kw):
        self.t += timedelta(**kw)


class Dialog:
    hears_voice = False

    def __init__(self, answers=()):
        self.said, self.asked, self.answers = [], [], list(answers)

    def say(self, text):
        self.said.append(text)

    def ask(self, prompt, timeout_s=None):
        self.asked.append(prompt)
        return Heard(self.answers.pop(0)) if self.answers else None


def make(now=datetime(2026, 11, 2, 22, 0, tzinfo=TZ)):
    clock = Clock(now)
    schedule = Schedule(Database(None, b"k" * 32), clock=clock.ts)
    return clock, schedule, AlarmCommands(schedule, now=clock)


def say(cmds, text, user=REZA, answers=()):
    d = Dialog(answers)
    assert cmds.handle(Heard(text), d, user) is True
    return " ".join(d.said), d


# ---- parsing -----------------------------------------------------------------------------
@pytest.mark.parametrize("text,seconds", [
    ("set a timer for 10 minutes", 600), ("10 minute timer", 600), ("an hour and a half", 5400),
    ("two and a half hours", 9000), ("1 hour 20 minutes", 4800), ("half an hour", 1800),
    ("timer for 90 seconds", 90), ("twenty five minutes", 1500), ("a quarter of an hour", 900),
    ("in 1.5 hours", 5400), ("a couple of minutes", 120),
])
def test_durations(text, seconds):
    assert parse_duration(text) == timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["set an alarm for 7", "what time is it", "my 3 kids"])
def test_no_duration(text):
    assert parse_duration(text) is None


def test_say_duration():
    assert say_duration(timedelta(minutes=90)) == "1 hour and 30 minutes"
    assert say_duration(timedelta(seconds=45)) == "45 seconds"


NIGHT = datetime(2026, 11, 2, 22, 0, tzinfo=TZ)        # Monday 10 PM
AFTERNOON = datetime(2026, 11, 2, 14, 0, tzinfo=TZ)    # Monday 2 PM


@pytest.mark.parametrize("text,now,expected", [
    ("set an alarm for 7", NIGHT, datetime(2026, 11, 3, 7, 0)),         # next 7 that comes round
    ("set an alarm for 7", AFTERNOON, datetime(2026, 11, 2, 19, 0)),
    ("alarm for 7 pm", NIGHT, datetime(2026, 11, 3, 19, 0)),
    ("alarm for 6:30 am", AFTERNOON, datetime(2026, 11, 3, 6, 30)),
    ("alarm at 17:45", AFTERNOON, datetime(2026, 11, 2, 17, 45)),
    ("alarm for 7.15", NIGHT, datetime(2026, 11, 3, 7, 15)),
    ("alarm at noon", NIGHT, datetime(2026, 11, 3, 12, 0)),
    ("alarm for seven thirty", NIGHT, datetime(2026, 11, 3, 7, 30)),
    ("alarm for 8 in the evening", AFTERNOON, datetime(2026, 11, 2, 20, 0)),
])
def test_clock_times(text, now, expected):
    ct = parse_clock(text)
    assert resolve_clock(ct, now) == expected.replace(tzinfo=TZ)


def test_wake_me_means_morning():
    ct = parse_clock("wake me up at 7")
    assert resolve_clock(ct, AFTERNOON, morning=True) == datetime(2026, 11, 3, 7, 0, tzinfo=TZ)


def test_named_day_bare_hours():
    ct = parse_clock("remind me tomorrow at 5")
    assert resolve_clock(ct, NIGHT, day=NIGHT.date() + timedelta(days=1)).hour == 17
    ct = parse_clock("tomorrow at 9")
    assert resolve_clock(ct, NIGHT, day=NIGHT.date() + timedelta(days=1)).hour == 9


@pytest.mark.parametrize("text", ["remind me in 2 hours", "set a timer for 10 minutes", "I have 3 kids"])
def test_not_clock_times(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize("text,days", [
    ("every weekday at 7", [0, 1, 2, 3, 4]), ("every day at 8", list(range(7))), ("daily at 6", list(range(7))),
    ("on weekends at 9", [5, 6]), ("every Monday and Wednesday at 6", [0, 2]), ("on Mondays at 9", [0]),
    ("on Monday at 9", None), ("at 7 tomorrow", None),
])
def test_repeats(text, days):
    assert parse_repeat(text) == days


def test_repeats_keep_wall_time_across_dst():
    # Clocks go back on 1 November 2026 in Chicago: 7 AM stays 7 AM.
    before = datetime(2026, 10, 31, 8, 0, tzinfo=TZ)
    nxt = next_on_days(7, 0, list(range(7)), before)
    assert (nxt.day, nxt.hour) == (1, 7)
    assert nxt.utcoffset() != before.utcoffset()


@pytest.mark.parametrize("text,label", [
    ("remind me to call my mom at 5", "call my mom"),
    ("remind me at 5 to call my mom", "call my mom"),
    ("remind me in 20 minutes to check the oven", "check the oven"),
    ("remind me tomorrow at 9 to submit the form", "submit the form"),
    ("remind me tomorrow morning to take out the trash", "take out the trash"),
    ("remind me about the dentist on Friday at 3 pm", "the dentist"),
    ("remind me at 5", ""),
])
def test_reminder_labels(text, label):
    assert reminder_label(text) == label


@pytest.mark.parametrize("text,label", [
    ("set a pasta timer for 8 minutes", "pasta"), ("set a timer for 10 minutes", ""), ("10 minute timer", ""),
    ("egg timer for 6 minutes", "egg"), ("start a 5 minute timer", ""),
])
def test_timer_labels(text, label):
    assert timer_label(text) == label


@pytest.mark.parametrize("text,reply", [
    ("stop", ("stop", None)), ("okay I'm up", ("stop", None)), ("snooze", ("snooze", None)),
    ("snooze for 5 minutes", ("snooze", 5.0)), ("five more minutes", ("snooze", 5.0)),
    ("mmm what", None), ("", None),
])
def test_ring_replies(text, reply):
    assert ring_reply(text) == reply


# ---- commands -----------------------------------------------------------------------------
def test_set_alarm_reads_back_the_day():
    clock, schedule, cmds = make()
    reply, _ = say(cmds, "set an alarm for 7")
    assert reply == "Alarm set for 7 AM tomorrow. That's 9 hours from now."
    [item] = schedule.for_user("r")
    assert item.kind == ALARM and item.due_at(TZ) == datetime(2026, 11, 3, 7, 0, tzinfo=TZ)


@pytest.mark.parametrize("text,expected", [
    ("set an alarm for 7 tomorrow", "Alarm set for 7 AM tomorrow."),
    ("alarm for 3 on Friday", "Alarm set for 3 PM on Friday."),
    ("alarm for 7 pm on Friday", "Alarm set for 7 PM on Friday."),
])
def test_alarm_on_another_day(text, expected):
    clock, schedule, cmds = make()
    assert say(cmds, text)[0].startswith(expected)


def test_repeating_alarm():
    clock, schedule, cmds = make()
    reply, _ = say(cmds, "set an alarm for 6:30 every weekday")
    assert reply.startswith("Alarm set for 6:30 AM on weekdays. The first one is 6:30 AM tomorrow")
    [item] = schedule.all()
    assert (item.hour, item.minute, item.days) == (6, 30, [0, 1, 2, 3, 4])


def test_alarm_asks_for_a_time():
    clock, schedule, cmds = make()
    reply, d = say(cmds, "set an alarm", answers=["6 am"])
    assert d.asked == ["For what time?"]
    assert "6 AM tomorrow" in reply


def test_duplicate_alarm_is_refused():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7")
    reply, _ = say(cmds, "wake me up at 7 am")
    assert reply == "You already have an alarm for 7 AM tomorrow."
    assert len(schedule.all()) == 1


def test_timers_and_labels():
    clock, schedule, cmds = make()
    assert say(cmds, "set a timer for 10 minutes")[0] == "Timer set for 10 minutes."
    assert say(cmds, "pasta timer for 8 minutes")[0] == "A pasta timer, starting now."
    clock.advance(minutes=3)
    reply, _ = say(cmds, "how long is left on the timers")
    assert "a pasta timer with 5 minutes left" in reply and "a 10 minute timer with 7 minutes left" in reply


def test_timer_too_long():
    clock, schedule, cmds = make()
    assert say(cmds, "set a timer for 30 hours")[0].startswith("Timers go up to 24 hours")
    assert schedule.all() == []


def test_reminders():
    clock, schedule, cmds = make()
    reply, _ = say(cmds, "remind me in 20 minutes to check the oven")
    assert reply == "Okay, I'll remind you to check the oven at 10:20 PM today."
    reply, d = say(cmds, "remind me to call my mom", answers=["tomorrow at 5"])
    assert d.asked == ["When should I remind you?"]
    assert reply == "Okay, I'll remind you to call your mom at 5 PM tomorrow."
    reply, d = say(cmds, "remind me at 9 tomorrow", answers=["submit the form"])
    assert d.asked == ["What should I remind you about?"]
    assert [i.label for i in schedule.for_user("r", (REMINDER,))] == ["check the oven", "submit the form",
                                                                      "call my mom"]      # soonest first


def test_guests_get_timers_only():
    clock, schedule, cmds = make()
    assert say(cmds, "set an alarm for 7", user=None)[0].startswith("I only set alarms and reminders")
    assert say(cmds, "remind me to call mom at 5", user=None)[0].startswith("I only set alarms and reminders")
    assert say(cmds, "set a timer for 5 minutes", user=None)[0] == "Timer set for 5 minutes."
    [t] = schedule.all()
    assert t.user_id == GUEST
    # A guest's timer is everyone's.
    assert "timer" in say(cmds, "what timers are running", user=SAM)[0]


def test_people_only_see_and_cancel_their_own():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7", user=REZA)
    say(cmds, "remind me at 9 to call the bank", user=REZA)
    assert say(cmds, "what alarms do I have", user=SAM)[0] == "You don't have any alarms set."
    assert say(cmds, "what are my reminders", user=SAM)[0] == "You don't have any reminders set."
    assert say(cmds, "cancel my alarm", user=SAM)[0] == "You don't have any alarms set."
    assert len(schedule.all()) == 2


def test_listing():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7")
    say(cmds, "alarm for 8 on weekends")
    assert say(cmds, "what alarms do I have")[0] == "2 alarms: 7 AM tomorrow and 8 AM on weekends."
    assert say(cmds, "when is my next alarm")[0].startswith("2 alarms")
    say(cmds, "remind me tomorrow at 9 to submit the form")
    assert say(cmds, "what are my reminders")[0] == "A reminder at 9 AM tomorrow: submit the form."


def test_cancel_by_time_and_by_words():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7")
    say(cmds, "set an alarm for 8")
    say(cmds, "remind me at 9 to call my mom")
    say(cmds, "remind me at 10 to pay rent")
    assert say(cmds, "cancel my 8 o'clock alarm")[0] == "Cancelled the 8 AM alarm tomorrow."
    assert say(cmds, "delete the reminder about mom")[0] == "Cancelled the reminder to call your mom."
    assert [i.label for i in schedule.for_user("r", (REMINDER,))] == ["pay rent"]


def test_cancel_asks_which():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7")
    say(cmds, "set an alarm for 8")
    reply, d = say(cmds, "cancel my alarm", answers=["the 7 AM one"])
    assert d.asked == ["Which one? You have the 7 AM alarm tomorrow and the 8 AM alarm tomorrow."]
    assert reply == "Cancelled the 7 AM alarm tomorrow."
    reply, d = say(cmds, "set an alarm for 9")[0], None
    reply, d = say(cmds, "cancel the alarm", answers=["hmm"])
    assert reply == "Okay, I've left them all."
    assert len(schedule.all()) == 2


def test_cancel_all():
    clock, schedule, cmds = make()
    for t in ("set an alarm for 7", "set an alarm for 8", "set a timer for 5 minutes"):
        say(cmds, t)
    assert say(cmds, "cancel all my alarms")[0] == "Cancelled 2 alarms."
    assert [i.kind for i in schedule.all()] == [TIMER]


@pytest.mark.parametrize("text", [
    "whenever I say I'm leaving, remind me to take my keys", "what's on my calendar tomorrow",
    "what time is it", "set my city to Tampa", "what's the weather", "remember that my alarm code is 1234",
    "tell me a joke about timers",
])
def test_not_alarm_commands(text):
    clock, schedule, cmds = make()
    assert cmds.handle(Heard(text), Dialog(), REZA) is False


def test_the_guard_answers_what_alarms_cant_do():
    assert "can't set alarms" in unsupported("set a reminder for my phone")
    assert unsupported("set a reminder for my phone", alarms=True).startswith("I can't do that one")


# ---- going off -----------------------------------------------------------------------------
class RingIO:
    """A scripted room: each hear() pops a reply; each ring takes ~5 s of the fake clock."""

    def __init__(self, clock, replies=()):
        self.clock, self.replies = clock, list(replies)
        self.said, self.rings, self.chimes = [], [], 0

    def say(self, text):
        self.said.append(text)

    def ring(self, volume):
        self.rings.append(volume)
        self.clock.advance(seconds=2.4)

    def chime(self):
        self.chimes += 1

    def hear(self, timeout_s):
        self.clock.advance(seconds=timeout_s if not self.replies else 1)
        return self.replies.pop(0) if self.replies else None


def ringer_for(schedule, clock, **kw):
    people = {"r": REZA, "s": SAM}
    return Ringer(schedule, people.get, clock=clock.ts, **kw)


def test_has_due_is_cheap_and_exact():
    clock, schedule, cmds = make()
    say(cmds, "set a timer for 10 minutes")
    assert not schedule.has_due()
    clock.advance(minutes=10)
    assert schedule.has_due()


def test_alarm_rings_until_stopped():
    clock, schedule, cmds = make(datetime(2026, 11, 2, 14, 0, tzinfo=TZ))
    say(cmds, "set an alarm for 3 pm")
    clock.advance(hours=1)
    io = RingIO(clock, [None, "mmm", "okay stop"])
    ringer_for(schedule, clock).go_off(io)
    assert io.said[0] == "It's 3 PM, Reza."
    assert len(io.rings) == 3 and io.rings == sorted(io.rings)       # gets louder
    assert io.said[-1] == "Okay."
    assert schedule.all() == []


def test_morning_alarm_starts_the_briefing():
    clock, schedule, cmds = make()
    say(cmds, "wake me up at 7")
    clock.t = datetime(2026, 11, 3, 7, 0, tzinfo=TZ)
    briefed = []
    ringer = ringer_for(schedule, clock)
    ringer.after_morning_alarm = lambda io, user: briefed.append(user.name)
    ringer.go_off(RingIO(clock, ["I'm up"]))
    assert briefed == ["Reza"]


def test_snooze():
    clock, schedule, cmds = make()
    say(cmds, "wake me up at 7")
    clock.t = datetime(2026, 11, 3, 7, 0, tzinfo=TZ)
    io = RingIO(clock, ["snooze"])
    ringer_for(schedule, clock, snooze_minutes=9).go_off(io)
    [item] = schedule.all()
    assert item.snoozed and item.due_at(TZ).strftime("%H:%M") == "07:09"
    assert io.said[-1] == "Snoozing until 7:09 AM."
    io = RingIO(clock, ["snooze for 5 minutes"])
    clock.t = item.due_at(TZ)
    ringer_for(schedule, clock).go_off(io)
    assert schedule.all()[0].due_at(TZ).strftime("%H:%M") == "07:14"


def test_gives_up_after_ring_minutes():
    clock, schedule, cmds = make()
    say(cmds, "set a timer for 1 minute")
    clock.advance(minutes=1)
    io = RingIO(clock)
    ringer_for(schedule, clock, ring_minutes=1).go_off(io)
    assert io.said == ["Your 1 minute timer is done."]
    assert 8 <= len(io.rings) <= 13
    assert schedule.all() == []


def test_repeating_alarm_moves_to_the_next_day():
    clock, schedule, cmds = make()
    say(cmds, "alarm for 7 every weekday")
    clock.t = datetime(2026, 11, 6, 7, 0, tzinfo=TZ)        # a Friday
    ringer_for(schedule, clock).go_off(RingIO(clock, ["stop"]))
    [item] = schedule.all()
    assert item.due_at(TZ) == datetime(2026, 11, 9, 7, 0, tzinfo=TZ)   # Monday


def test_alarm_and_timer_ring_together():
    clock, schedule, cmds = make(datetime(2026, 11, 2, 14, 50, tzinfo=TZ))
    say(cmds, "set an alarm for 3 pm")
    say(cmds, "pasta timer for 10 minutes")
    clock.advance(minutes=10)
    io = RingIO(clock, ["stop"])
    ringer_for(schedule, clock).go_off(io)
    assert io.said[0] == "It's 3 PM, Reza. Your pasta timer is done."
    assert schedule.all() == []


def test_reminder_is_said_with_a_chime_and_can_be_snoozed():
    clock, schedule, cmds = make()
    say(cmds, "remind me in 5 minutes to call my mom")
    clock.advance(minutes=5)
    io = RingIO(clock, ["remind me again in 15 minutes"])
    ringer_for(schedule, clock).go_off(io)
    assert io.chimes == 1 and io.rings == []
    assert io.said[0] == "Reza, a reminder: call your mom."
    assert io.said[1] == "Okay, I'll remind you again at 10:20 PM."
    clock.advance(minutes=15)
    io = RingIO(clock, [None])
    ringer_for(schedule, clock).go_off(io)
    assert schedule.all() == []


def test_catch_up_after_being_off():
    clock, schedule, cmds = make()
    say(cmds, "wake me up at 7")
    say(cmds, "alarm for 8 every day")
    say(cmds, "remind me tomorrow at 7:30 am to call my mom")
    say(cmds, "set a timer for 10 hours")
    clock.t = datetime(2026, 11, 3, 8, 2, tzinfo=TZ)         # Orion starts at 8:02
    ringer = ringer_for(schedule, clock, ring_minutes=5)
    lines = ringer.catch_up()
    assert "You missed an alarm at 7 AM, Reza." in lines
    assert "While I was off, Reza, you had a reminder at 7:30 AM: call your mom." in lines
    assert not any("8 AM" in line for line in lines)        # recent: still rings
    kinds = sorted(i.kind for i in schedule.all())
    assert kinds == [ALARM, TIMER]
    io = RingIO(clock, ["stop"])
    ringer.go_off(io)                                        # the 8 AM alarm, two minutes late
    [alarm] = [i for i in schedule.all() if i.kind == ALARM]
    assert alarm.due_at(TZ) == datetime(2026, 11, 4, 8, 0, tzinfo=TZ)


def test_ring_tone():
    tone = ring_tone(24000, 0.5)
    assert tone.dtype == np.float32 and 2.0 < len(tone) / 24000 < 3.0
    assert np.abs(tone).max() <= 0.5 + 1e-6


# ---- the voice loop rings and stops ------------------------------------------------------------
def test_voice_loop_rings_and_hears_stop():
    from tests.test_voice_loop import STT, FakeIdentity, Mic, ScriptedEndpointer, Speaker
    from core.voice_loop import VoiceLoop

    class NoWake:
        last_score = 0.0

        def detect(self, frame):
            return False

        def reset(self):
            pass

    clock, schedule, cmds = make(datetime(2026, 11, 2, 14, 0, tzinfo=TZ))
    say(cmds, "set a timer for 1 minute")
    clock.advance(minutes=1)
    speaker = Speaker()
    ep = ScriptedEndpointer(["silence", "speech"])
    loop = VoiceLoop(Mic(n=5), NoWake(), ep, STT(["okay, stop"]), None, speaker, None, FakeIdentity(), None,
                     show=lambda s: None)
    loop.ringer = ringer_for(schedule, clock)
    with pytest.raises(KeyboardInterrupt):
        loop.run("hey orion")
    assert speaker.said == ["Orion online.", "Your 1 minute timer is done.", "Okay, timer off."]
    assert ep.timeouts == [3.0, 3.0]            # two listening gaps between rings, no wake word
    assert ep.end_frames == 10                  # the short end-of-speech pause was put back
    assert schedule.all() == []


# ---- storage and the systemd timer -----------------------------------------------------------
class FakeSystemctl:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[2:])

        class R:
            returncode, stderr = 0, ""
        return R()


def test_wakeup_timer_holds_only_a_time(tmp_path):
    clock, schedule, cmds = make()
    run = FakeSystemctl()
    timer = WakeupTimer(tmp_path / "proj", early_s=90, directory=tmp_path / "units", python="/venv/python", run=run)
    schedule.on_change.append(timer.sync)
    say(cmds, "remind me at 7 am to call Dr Patel about my test results")
    text = timer.timer_path.read_text()
    assert "OnCalendar=2026-11-03 06:58:30" in text
    for private in ("Patel", "results", "Reza", "reminder"):
        assert private not in text and private not in timer.service_path.read_text()
    service = timer.service_path.read_text()
    assert "main.py alarm wake" in service and "StandardOutput=null" in service
    assert ["daemon-reload"] in run.calls and ["restart", "orion-alarm.timer"] in run.calls
    n = len(run.calls)
    say(cmds, "set an alarm for 9 am")                        # later: the timer doesn't change
    assert len(run.calls) == n
    for item in schedule.all():
        schedule.delete(item.id)
    assert not timer.timer_path.exists() and not timer.service_path.exists()
    assert ["disable", "--now", "orion-alarm.timer"] in run.calls


def test_wipe_hook_removes_the_units(tmp_path):
    run = FakeSystemctl()
    WakeupTimer(tmp_path, directory=tmp_path / "u", run=run).sync(
        type("I", (), {"due": datetime(2026, 11, 3, 7, tzinfo=TZ).timestamp()})())
    assert (tmp_path / "u" / "orion-alarm.timer").exists()
    remove_hook(tmp_path / "u", run=run)()
    assert list((tmp_path / "u").iterdir()) == []


def test_systemd_missing_is_not_fatal(tmp_path):
    def boom(cmd, **kw):
        raise FileNotFoundError("systemctl")
    WakeupTimer(tmp_path, directory=tmp_path / "u", run=boom).sync(
        type("I", (), {"due": datetime(2026, 11, 3, 7, tzinfo=TZ).timestamp()})())


def test_deleting_a_person_deletes_their_items_and_settings():
    clock, schedule, cmds = make()
    say(cmds, "set an alarm for 7", user=REZA)
    say(cmds, "set an alarm for 8", user=SAM)
    schedule.set_setting("r", "weather_place", "{}")
    assert schedule.delete_user("r") == 1
    assert [i.user_id for i in schedule.all()] == ["s"]
    assert schedule.setting("r", "weather_place") is None


def test_schedule_is_in_the_encrypted_database(tmp_path):
    from storage.db import open_database

    db = open_database(tmp_path, b"x" * 32)
    Schedule(db).add("r", REMINDER, datetime(2026, 11, 3, 7, tzinfo=TZ), "pick up the prescription")
    db.close()
    assert b"prescription" not in (tmp_path / "orion.db").read_bytes()


def test_main_wires_phase4(monkeypatch, tmp_path):
    import main
    from core.assistant import Assistant
    from core.config import load_config

    cfg = load_config(None, profile="laptop")
    synced = []

    class FakeTimer:
        def sync(self, item):
            synced.append(item)

    monkeypatch.setattr(main, "make_wakeup_timer", lambda cfg: FakeTimer())
    assistant = Assistant(llm=None, speaker=None, metrics=None, profile="laptop")
    store = type("Store", (), {"get": lambda self, uid: REZA if uid == "r" else None,
                               "owner": lambda self: REZA})()
    knowledge = main.Knowledge(db=Database(None, b"k" * 32))
    main.make_phase4(cfg, assistant, store, knowledge, audit=None)
    assert assistant.alarms and assistant.weather and assistant.briefing and assistant.ringer
    assert assistant.ringer.after_morning_alarm is not None
    assert synced == [None]                                  # armed (nothing due) at start-up
    assistant.alarms.schedule.add("r", ALARM, datetime(2030, 1, 1, 7, tzinfo=TZ))
    assert synced[-1].kind == ALARM                          # re-armed on every change


def test_main_without_a_database_leaves_phase4_off():
    import main
    from core.assistant import Assistant
    from core.config import load_config

    assistant = Assistant(llm=None, speaker=None, metrics=None, profile="laptop")
    main.make_phase4(load_config(None, profile="laptop"), assistant, None, main.Knowledge(), audit=None)
    assert assistant.alarms is None and assistant.ringer is None
