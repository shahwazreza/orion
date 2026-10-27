#!/usr/bin/env python3
"""Download the wake word, VAD, speaker recognition, and voice (Kokoro, Piper) models into ./models.

Every file is checked against a pinned SHA-256, so a tampered or changed
upstream file is rejected. Whisper models are fetched by faster-whisper on
first use into models/whisper/.

    python scripts/download_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
SPEAKER = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
PIPER_GB = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB"
PIPER = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium"

# (url, destination relative to models/, sha256)
# Hashes recorded on first download (2026-10-26). To update a model, change the
# URL, set its hash to "", run the script, review, then pin the printed hash.
FILES: list[tuple[str, str, str]] = [
    (f"{OWW}/melspectrogram.onnx", "openwakeword/melspectrogram.onnx",
     "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"),
    (f"{OWW}/embedding_model.onnx", "openwakeword/embedding_model.onnx",
     "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"),
    (f"{OWW}/silero_vad.onnx", "openwakeword/silero_vad.onnx",
     "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28"),
    (f"{OWW}/hey_jarvis_v0.1.onnx", "openwakeword/hey_jarvis_v0.1.onnx",
     "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"),
    (f"{SPEAKER}/nemo_en_titanet_small.onnx", "speaker/nemo_en_titanet_small.onnx",
     "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"),
    (f"{KOKORO}/kokoro-v1.0.onnx", "kokoro/kokoro-v1.0.onnx",
     "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"),
    (f"{KOKORO}/voices-v1.0.bin", "kokoro/voices-v1.0.bin",
     "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    (f"{PIPER}/en_GB-alan-medium.onnx", "piper/en_GB-alan-medium.onnx",
     "0a309668932205e762801f1efc2736cd4b0120329622adf62be09e56339d3330"),
    (f"{PIPER}/en_GB-alan-medium.onnx.json", "piper/en_GB-alan-medium.onnx.json",
     "c0f0d124e5895c00e7c03b35dcc8287f319a6998a365b182deb5c8e752ee8c1e"),
    (f"{PIPER_GB}/northern_english_male/medium/en_GB-northern_english_male-medium.onnx", "piper/en_GB-northern_english_male-medium.onnx",
     "57a219ae8e638873db7d18893304be5069c42868f392bb95c3ff17f0690d0689"),
    (f"{PIPER_GB}/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json", "piper/en_GB-northern_english_male-medium.onnx.json",
     "69557ed3d974463453e9b0c09dd99a7ed0e52b8b87b64b357dbeeb2540a97d47"),
    (f"{PIPER_GB}/semaine/medium/en_GB-semaine-medium.onnx", "piper/en_GB-semaine-medium.onnx",
     "d6dab6f3b92db43ea3f78c7f20dc8eadb47a1f15d8a1c9d451cf3ccd201a2f66"),
    (f"{PIPER_GB}/semaine/medium/en_GB-semaine-medium.onnx.json", "piper/en_GB-semaine-medium.onnx.json",
     "6425dcb878684043b77d772b173ae006d86a583b110303edda48b8438ecee5ee"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def main() -> int:
    failed = False
    for url, rel, expected in FILES:
        dest = MODELS / rel
        if not dest.exists():
            print(f"downloading {rel}")
            try:
                fetch(url, dest)
            except OSError as exc:
                print(f"  FAILED: {exc}")
                failed = True
                continue
        actual = sha256(dest)
        if expected and actual != expected:
            print(f"  CHECKSUM MISMATCH for {rel}; deleting it.\n  expected {expected}\n  got      {actual}")
            dest.unlink()
            failed = True
        elif not expected:
            print(f"  {rel}: unpinned, sha256={actual}")
        else:
            print(f"ok {rel}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
