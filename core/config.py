"""Configuration loading with hardware profiles.

Resolution order: ``common`` <- ``profiles.<name>`` <- demo <- ``config.local.yaml``
<- explicit overrides.

``config.local.yaml`` sits next to ``config.yaml``, is git-ignored, and holds
settings for this machine only (e.g. ``outlook.client_id``), in the same shape as
``common``.
The result is a plain nested dict wrapped in :class:`Config` for attribute-free,
dotted-key access (``cfg.get("llm.model")``).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from core import hardware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"


class ConfigError(ValueError):
    pass


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged recursively onto ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _profile_matches(rule: Mapping[str, Any], probe: hardware.Probe) -> bool:
    """A profile matches if every condition in its ``detect`` rule holds."""
    checks: dict[str, Callable[[Any], bool]] = {
        "device_tree_model_contains": lambda v: v.lower() in probe.device_tree_model.lower(),
        "machine": lambda v: probe.machine == v,
        "file_exists": lambda v: Path(v).exists(),
    }
    for key, expected in rule.items():
        if key not in checks:
            raise ConfigError(f"Unknown detect rule '{key}'. Known: {sorted(checks)}")
        if not checks[key](expected):
            return False
    return True


def select_profile(raw: Mapping[str, Any], requested: str | None, probe: hardware.Probe) -> str:
    profiles = raw.get("profiles") or {}
    requested = requested or os.environ.get("ORION_PROFILE") or raw.get("active_profile", "auto")
    if requested != "auto":
        if requested not in profiles:
            raise ConfigError(f"Profile '{requested}' not found. Available: {sorted(profiles)}")
        return requested
    for name, body in profiles.items():
        rule = (body or {}).get("detect")
        if rule and _profile_matches(rule, probe):
            return name
    default = raw.get("default_profile")
    if default not in profiles:
        raise ConfigError(f"default_profile '{default}' is not a defined profile")
    return default


@dataclass
class Config:
    data: dict[str, Any]
    profile: str
    root: Path = field(default=PROJECT_ROOT)
    demo: bool = False

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data.get(name) or {})

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        value = self.get(dotted)
        if value is None:
            raise ConfigError(f"Missing path setting '{dotted}'")
        p = Path(os.path.expanduser(str(value)))
        return p if p.is_absolute() else (self.root / p)


def load_config(
    path: str | os.PathLike[str] | None = None,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    probe: hardware.Probe | None = None,
    demo: bool = False,
) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    probe = probe or hardware.probe()
    name = select_profile(raw, profile, probe)
    body = dict(raw["profiles"][name] or {})
    body.pop("detect", None)

    merged = deep_merge(raw.get("common") or {}, body)
    if demo:
        merged = deep_merge(merged, raw.get("demo") or {})
    local_path = cfg_path.resolve().parent / LOCAL_CONFIG_NAME
    if local_path.exists():
        with open(local_path, encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        if not isinstance(local, Mapping):
            raise ConfigError(f"{LOCAL_CONFIG_NAME} must hold settings like config.yaml's 'common' section")
        merged = deep_merge(merged, local)
    if overrides:
        merged = deep_merge(merged, overrides)
    return Config(data=merged, profile=name, root=cfg_path.resolve().parent, demo=demo)


def save_local_settings(updates: Mapping[str, Any], root: Path = PROJECT_ROOT) -> Path:
    """Merge ``updates`` into ``config.local.yaml`` (git-ignored) and write it back, private to you.

    Other settings already in the file are kept; YAML comments in it are not.
    """
    path = root / LOCAL_CONFIG_NAME
    current: dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"{LOCAL_CONFIG_NAME} must hold settings like config.yaml's 'common' section")
        current = dict(loaded)
    merged = deep_merge(current, updates)
    text = ("# This computer's own settings, applied on top of config.yaml. Git-ignored: never committed.\n"
            + yaml.safe_dump(merged, sort_keys=False, default_flow_style=False))
    tmp = path.with_suffix(".yaml.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path
