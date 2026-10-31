import textwrap

import pytest

from core.config import ConfigError, deep_merge, load_config
from core.hardware import Probe

LAPTOP = Probe(machine="x86_64", device_tree_model="")
JETSON = Probe(machine="aarch64", device_tree_model="NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super")
ORIN_NX = Probe(machine="aarch64", device_tree_model="NVIDIA Jetson Orin NX Engineering Reference Developer Kit")


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        active_profile: auto
        default_profile: laptop
        common:
          paths: {models: models, logs: /var/tmp/orion-logs}
          llm: {model: base, host: http://127.0.0.1:11434, num_ctx: 2048}
          stt: {model: small.en, device: auto}
        profiles:
          orin_nx:
            detect: {device_tree_model_contains: "Orin NX"}
            llm: {model: big}
          jetson:
            detect: {device_tree_model_contains: "Jetson"}
            llm: {model: medium}
            stt: {device: cuda}
          laptop:
            llm: {model: small}
    """))
    return p


def test_auto_detects_laptop_by_default(cfg_file, monkeypatch):
    monkeypatch.delenv("ORION_PROFILE", raising=False)
    cfg = load_config(cfg_file, probe=LAPTOP)
    assert cfg.profile == "laptop"
    assert cfg.get("llm.model") == "small"
    assert cfg.get("llm.num_ctx") == 2048          # inherited from common


def test_auto_detects_jetson_and_merges(cfg_file, monkeypatch):
    monkeypatch.delenv("ORION_PROFILE", raising=False)
    cfg = load_config(cfg_file, probe=JETSON)
    assert cfg.profile == "jetson"
    assert cfg.get("stt.device") == "cuda"
    assert cfg.get("stt.model") == "small.en"      # untouched keys survive the merge
    assert "detect" not in cfg.data


def test_more_specific_profile_listed_first_wins(cfg_file, monkeypatch):
    monkeypatch.delenv("ORION_PROFILE", raising=False)
    assert load_config(cfg_file, probe=ORIN_NX).profile == "orin_nx"


def test_explicit_and_env_override(cfg_file, monkeypatch):
    monkeypatch.setenv("ORION_PROFILE", "jetson")
    assert load_config(cfg_file, probe=LAPTOP).profile == "jetson"
    assert load_config(cfg_file, profile="laptop", probe=JETSON).profile == "laptop"


def test_unknown_profile_rejected(cfg_file):
    with pytest.raises(ConfigError):
        load_config(cfg_file, profile="toaster", probe=LAPTOP)


def test_overrides_and_paths(cfg_file, monkeypatch):
    monkeypatch.delenv("ORION_PROFILE", raising=False)
    cfg = load_config(cfg_file, probe=LAPTOP, overrides={"llm": {"model": "x"}})
    assert cfg.get("llm.model") == "x"
    assert cfg.path("paths.models") == cfg_file.parent / "models"
    assert str(cfg.path("paths.logs")) == "/var/tmp/orion-logs"


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_real_config_loads_for_every_profile():
    for name in ("laptop", "jetson"):
        cfg = load_config(profile=name, probe=LAPTOP)
        assert cfg.get("llm.host").startswith("http://127.0.0.1")
        assert cfg.get("wakeword.models")


def test_local_settings_file_wins_over_profiles_but_not_explicit_overrides(cfg_file, monkeypatch):
    monkeypatch.delenv("ORION_PROFILE", raising=False)
    (cfg_file.parent / "config.local.yaml").write_text(
        "outlook: {client_id: abc-123, tenant: consumers}\nllm: {model: local-choice}\n")
    cfg = load_config(cfg_file, probe=LAPTOP)
    assert cfg.get("outlook.client_id") == "abc-123" and cfg.get("outlook.tenant") == "consumers"
    assert cfg.get("llm.model") == "local-choice" and cfg.get("llm.num_ctx") == 2048
    assert load_config(cfg_file, probe=LAPTOP, overrides={"llm": {"model": "cli"}}).get("llm.model") == "cli"


def test_local_settings_file_must_be_a_mapping(cfg_file):
    (cfg_file.parent / "config.local.yaml").write_text("- just a list\n")
    with pytest.raises(ConfigError):
        load_config(cfg_file, probe=LAPTOP)


def test_local_settings_file_is_git_ignored():
    from pathlib import Path

    ignored = (Path(__file__).resolve().parent.parent / ".gitignore").read_text().splitlines()
    assert "config.local.yaml" in ignored
