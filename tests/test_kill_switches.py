"""Phase 2.5: the wipe passphrase, the nuclear wipe, shutdown, mute, and stop."""

import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.lifecycle import (PassphraseHeard, ProcessFile, Shutdown, WipeConfirmed, is_mute, is_shutdown,
                            is_unmute, is_wipe_confirmation, stop_running)
from core.voice_loop import VoiceLoop
from security.killswitch import KillSwitch, normalize, problems_with
from security.wipe import KNOWN_SECRETS, Wipe
from storage.encrypted_file import EncryptedJSONFile

from tests.test_voice_loop import Assistant, FakeIdentity, Mic, ScriptedEndpointer, STT, Wake
from tests.test_voice_loop import Speaker as BaseSpeaker

KEY = bytes(range(32))
PHRASE = "Purple otter juggles quietly"


@pytest.fixture
def ks(tmp_path):
    k = KillSwitch(EncryptedJSONFile(tmp_path / "killswitch.enc", KEY, "killswitch"))
    k.set(PHRASE)
    return k


# ---- the passphrase -------------------------------------------------------------------------
def test_spoken_variants_match(ks):
    for heard in ["purple otter juggles quietly", "Purple otter, juggles quietly!", "Orion, purple otter "
                  "juggles quietly.", "Hey Orion purple otter juggles quietly", "Purple otter juggles quietly, Orion."]:
        assert ks.matches(heard), heard


def test_near_misses_and_ordinary_speech_dont_match(ks):
    for heard in ["purple otter juggles", "purple otter juggles quietly please", "purple otters juggle quietly",
                  "what's the weather", "", "Orion"]:
        assert not ks.matches(heard), heard


def test_only_a_salted_argon2id_hash_is_stored_and_its_encrypted(tmp_path, ks):
    raw = (tmp_path / "killswitch.enc").read_bytes()
    assert b"purple" not in raw.lower() and b"argon2" not in raw     # encrypted file
    state = EncryptedJSONFile(tmp_path / "killswitch.enc", KEY, "killswitch").load()
    assert state["hash"].startswith("$argon2id$") and "purple" not in state["hash"]
    other = KillSwitch(None)
    other.set(PHRASE)
    assert other._state.hash != state["hash"]                        # salted


def test_it_survives_a_restart(tmp_path, ks):
    again = KillSwitch(EncryptedJSONFile(tmp_path / "killswitch.enc", KEY, "killswitch"))
    assert again.is_set and again.matches(PHRASE)


@pytest.mark.parametrize("bad", ["two words", "a b c d", "purple 4 otters quietly", "confirm wipe now please",
                                 "orion purple otter juggles"])
def test_weak_or_confusing_passphrases_are_refused(bad):
    assert problems_with(bad)
    with pytest.raises(ValueError):
        KillSwitch(None).set(bad)


def test_problems_never_repeat_the_passphrase():
    assert not any("otter" in p for p in problems_with("otter 4"))


def test_normalize():
    assert normalize("Don't  PANIC, it's fine!") == "dont panic its fine"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_typed_attempts_lock_out_and_lockouts_double(tmp_path):
    clock = Clock()
    k = KillSwitch(EncryptedJSONFile(tmp_path / "k.enc", KEY, "killswitch"), clock=clock)
    k.set(PHRASE)
    for _ in range(3):
        assert not k.verify_typed("wrong guess here")
    assert k.locked_for() == 300
    assert not k.verify_typed(PHRASE)                  # even the right one, while locked
    clock.t += 301
    for _ in range(3):
        k.verify_typed("wrong again here")
    assert k.locked_for() == 600                        # doubled
    # The lockout is remembered across restarts (it's in the encrypted file).
    assert KillSwitch(EncryptedJSONFile(tmp_path / "k.enc", KEY, "killswitch"), clock=clock).locked_for() == 600
    clock.t += 601
    assert k.verify_typed(PHRASE) and k.locked_for() == 0


def test_spoken_checks_are_not_counted(ks):
    for _ in range(10):
        ks.matches("just chatting about the weather")
    assert ks.locked_for() == 0 and ks._state.fails == 0


# ---- phrases ----------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["Orion, shut down.", "shut down", "Shutdown now.", "Turn yourself off.",
                                  "hey orion power off", "Shut yourself down, please."])
def test_shutdown_phrases(text):
    assert is_shutdown(text)


@pytest.mark.parametrize("text", ["Shut down the computer.", "How do I shut down my laptop?", "turn off the lights",
                                  "don't shut down", "go offline"])
def test_not_shutdown(text):
    assert not is_shutdown(text)


def test_mute_and_unmute_phrases():
    assert is_mute("Orion, stop listening.") and is_mute("stop listening for a while")
    assert not is_mute("why do you stop listening when I pause?")
    assert is_unmute("Start listening.") and is_unmute("you can listen again")
    assert not is_unmute("start listening to music")


@pytest.mark.parametrize("text,want", [("Confirm wipe.", True), ("Yes, confirm the wipe.", True),
                                       ("confirmed wiped", True), ("yes", False), ("confirm", False),
                                       ("don't confirm wipe", False), ("wipe", False)])
def test_wipe_confirmation(text, want):
    assert is_wipe_confirmation(text) is want


# ---- the voice loop ---------------------------------------------------------------------------
class Speaker(BaseSpeaker):
    def interrupt(self): pass
    def resume(self): pass


def run_loop(texts, ks, script=None):
    shown = []
    speaker, assistant = Speaker(), Assistant()
    ep = ScriptedEndpointer(script or ["speech"] * len(texts))
    loop = VoiceLoop(Mic(n=50), Wake(), ep, STT(texts), assistant, speaker, None, FakeIdentity(), None,
                     follow_up_s=6.0, show=shown.append)
    loop.kill_switch = ks
    return loop, speaker, assistant, shown


def test_spoken_passphrase_then_confirm_raises_wipe_and_is_never_shown(ks, caplog):
    caplog.set_level(logging.DEBUG)
    loop, speaker, assistant, shown = run_loop(["Orion, purple otter juggles quietly.", "Confirm wipe."], ks)
    with pytest.raises(WipeConfirmed):
        loop.run("hey orion")
    assert assistant.heard == []                         # never reached the model (or the conversation log)
    everything = " ".join(shown + speaker.said + [r.getMessage() for r in caplog.records]).lower()
    assert "otter" not in everything and "purple" not in everything
    assert "confirm wipe" in speaker.said[-2].lower() and speaker.said[-1] == "Wiping now. Goodbye."


def test_anything_but_the_confirmation_cancels(ks):
    loop, speaker, assistant, _ = run_loop(["purple otter juggles quietly", "yes", "what time is it"], ks)
    with pytest.raises(KeyboardInterrupt):               # the fake mic runs out: no wipe happened
        loop.run("hey orion")
    assert "Cancelled." in speaker.said
    assert assistant.heard == []                         # the conversation ended; nothing was answered


def test_shutdown_needs_no_confirmation(ks, monkeypatch):
    monkeypatch.setattr("core.voice_loop.exit_soon", lambda *a, **k: None)   # no watchdog in the test run
    loop, speaker, assistant, _ = run_loop(["Orion, shut down."], ks)
    with pytest.raises(Shutdown):
        loop.run("hey orion")
    assert assistant.heard == []


def test_mute_ignores_everything_until_start_listening(ks):
    texts = ["stop listening", "what's the weather", "Orion, start listening.", "what's the weather"]
    loop, speaker, assistant, shown = run_loop(texts, ks)
    loop.wake = type("EveryFrame", (), {"last_score": 1.0, "detect": lambda s, f: True, "reset": lambda s: None})()
    with pytest.raises(KeyboardInterrupt):
        loop.run("hey orion")
    assert assistant.heard == ["what's the weather"]     # only the one after unmuting
    assert "stopped listening" in speaker.said[1] and "listening again" in speaker.said[2]
    assert sum(s == "You: what's the weather" for s in shown) == 1   # the muted one wasn't even shown


def test_muted_orion_still_obeys_the_kill_switches(ks):
    loop, speaker, assistant, _ = run_loop(["purple otter juggles quietly", "confirm wipe"], ks)
    loop.muted = True
    with pytest.raises(WipeConfirmed):
        loop.run("hey orion")


# ---- the wipe --------------------------------------------------------------------------------
class FakeKeyring:
    def __init__(self, names, events):
        self.store = {("orion", n): "secret" for n in names}
        self.events = events

    def delete_password(self, service, name):
        self.events.append(f"keyring:{name}")
        if (service, name) not in self.store:
            raise KeyError(name)
        del self.store[(service, name)]


def sandbox(tmp_path):
    root = tmp_path / "orion"
    (root / "data" / "backups").mkdir(parents=True)
    (root / "data" / "orion.db").write_bytes(b"x" * 5000)
    (root / "data" / "backups" / "orion-2026-10-29.db").write_bytes(b"y" * 100)
    (root / "data" / "users.enc").write_bytes(b"z")
    (root / "logs").mkdir()
    (root / "logs" / "metrics.enc").write_bytes(b"m")
    (root / "models").mkdir()
    (root / "models" / "whisper.bin").write_bytes(b"w")
    (root / "main.py").write_text("code")
    projects = tmp_path / "orion-projects"
    projects.mkdir()
    (projects / "app.py").write_text("mine")
    return root, projects


def test_wipe_deletes_key_first_then_data_and_secrets_and_keeps_your_work(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    root, projects = sandbox(tmp_path)
    events = []
    kr = FakeKeyring(KNOWN_SECRETS, events)
    w = Wipe(root, root / "data", root / "logs", keyring_module=kr, searxng=False)
    import security.wipe as wipe_mod

    real_shred = wipe_mod.shred

    def tracking_shred(p):
        events.append(f"file:{p.name}")
        real_shred(p)

    monkeypatch.setattr(wipe_mod, "shred", tracking_shred)
    w.run()
    assert events[0] == "keyring:data-key-v1"            # crypto-shred before anything else
    assert any(e.startswith("file:") for e in events)
    assert kr.store == {}                                 # every Orion secret gone
    assert not (root / "data").exists() and not (root / "logs").exists()
    assert (root / "main.py").exists() and (root / "models" / "whisper.bin").exists()
    assert (projects / "app.py").read_text() == "mine"
    assert caplog.records == []                           # the wipe itself is never logged


def test_wipe_refuses_folders_outside_orion_or_holding_your_projects(tmp_path):
    root, projects = sandbox(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / "data" / "orion-projects").mkdir()            # a project folder inside data: refuse it all
    Wipe(root, outside, root / "data", keyring_module=FakeKeyring([], []), searxng=False).run()
    assert (outside / "keep.txt").exists() and (root / "data" / "orion.db").exists()


def test_wipe_removes_searxng_container_and_config(tmp_path, monkeypatch):
    root, _ = sandbox(tmp_path)
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/podman")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    (tmp_path / "cfg" / "orion" / "searxng").mkdir(parents=True)
    Wipe(root, root / "data", root / "logs", keyring_module=FakeKeyring([], []),
         run=lambda cmd, **kw: calls.append(cmd)).run()
    assert ["podman", "rm", "-f", "orion-searxng"] in calls
    assert any(c[:3] == ["podman", "unshare", "rm"] for c in calls)


def test_every_keyring_entry_in_the_code_is_wiped():
    """Later phases must list their secrets in security.wipe.KNOWN_SECRETS."""
    root = Path(__file__).resolve().parents[1]
    names = set()
    for f in [*root.glob("*.py"), *root.glob("[a-z]*/*.py")]:
        if f.parts[-2] in ("tests", "venv"):
            continue
        names |= set(re.findall(r"set_password\(\s*(?:SERVICE|\"orion\")\s*,\s*\"([^\"]+)\"", f.read_text()))
        names |= {m for m in re.findall(r"\"([a-z-]+-(?:api-key|token))\"", f.read_text())}
    assert names, "scan found nothing: the pattern is out of date"
    assert names <= set(KNOWN_SECRETS), names - set(KNOWN_SECRETS)


# ---- python main.py stop -------------------------------------------------------------------------
def test_stop_reaches_the_running_orion_through_the_process_file(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "main.py"])
    try:
        pf = ProcessFile(tmp_path)
        (tmp_path / "orion.pid").write_text(str(proc.pid))
        assert pf.running_pid() == proc.pid
        assert stop_running(pf) is True
        assert proc.poll() is not None
    finally:
        proc.kill()


def test_stale_or_foreign_process_files_are_ignored(tmp_path):
    pf = ProcessFile(tmp_path)
    assert pf.running_pid() is None
    (tmp_path / "orion.pid").write_text("999999")
    assert pf.running_pid() is None and stop_running(pf) is None
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])   # not Orion
    try:
        (tmp_path / "orion.pid").write_text(str(other.pid))
        assert pf.running_pid() is None
    finally:
        other.kill()


def test_passphrase_heard_carries_nothing():
    assert PassphraseHeard().args == ()
