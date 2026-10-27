"""Hardware detection: what machine is this, and which accelerators work."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    machine: str              # e.g. "x86_64", "aarch64"
    device_tree_model: str    # e.g. "NVIDIA Jetson Orin Nano Engineering Reference..." or ""


def probe() -> Probe:
    model = ""
    dt = Path("/proc/device-tree/model")
    try:
        model = dt.read_bytes().rstrip(b"\x00").decode(errors="replace")
    except OSError:
        pass
    return Probe(machine=platform.machine(), device_tree_model=model)


def ctranslate2_cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def onnxruntime_cuda_available() -> bool:
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def resolve_stt_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn ``auto`` settings into a concrete (device, compute_type) pair.

    An explicit ``cuda`` request falls back to CPU if CUDA isn't usable, so the
    same config works on a machine without a GPU build.
    """
    want_cuda = device in ("auto", "cuda")
    use_cuda = want_cuda and ctranslate2_cuda_available()
    resolved_device = "cuda" if use_cuda else "cpu"
    if compute_type == "auto" or (not use_cuda and compute_type.startswith("float16")):
        compute_type = "float16" if use_cuda else "int8"
    return resolved_device, compute_type


def resolve_onnx_cuda(setting: bool | str) -> bool:
    if setting == "auto":
        return onnxruntime_cuda_available()
    return bool(setting) and onnxruntime_cuda_available()
