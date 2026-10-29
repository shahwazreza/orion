#!/usr/bin/env python3
"""Build notebooks/train_hey_orion.ipynb from ONE set of cells (edit this, not the notebook).

    python scripts/wakeword/build_notebook.py                     # rewrite the notebook
    python scripts/wakeword/build_notebook.py --smoke smoke.py    # also a tiny local test run

The smoke script runs the same cells with tiny settings (120 clips, 300 steps,
validation features in place of the 16 GB set) and plain Python instead of
notebook "!" lines. It needs the training dependencies installed (see the
notebook's install cell) and runs third-party training code, so run it in a
throwaway environment.

Compatibility fixes, each found running the pipeline on current Colab:
  - scipy<1.15: `acoustics` needs scipy.special.sph_harm (removed in 1.15)
  - TTS checkpoint: PyTorch 2.6+ refuses pickles by default; SHA-256 is verified
    first, then it's loaded with weights_only=False
  - TFLite export removed (needs tensorflow-cpu 2.8.1, no longer installable)
  - classic ONNX exporter (dynamo=False) + onnxscript: PyTorch's new default
    exporter crashed the export after a full training run
  - openWakeWord added to sys.path: Colab can't import a `pip install -e`
    package until the runtime restarts
  - torchaudio 2.9+ removed `info` and moved `load`: a soundfile-based shim
  - step 7 checks all four feature files: openWakeWord skips the whole step if
    any exists, so an interrupted run silently left the rest missing
  - training targets 0.5 false activations/hour with a 64-unit network: the
    first full run (0.2, 32 units) reached only 49% recall
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "train_hey_orion.ipynb")
_ap.add_argument("--smoke", type=Path, help="also write a tiny local smoke-test script here")
_args = _ap.parse_args()
OUT_NOTEBOOK = _args.out
OUT_SMOKE = _args.smoke

INTRO = """# Train Orion's "hey orion" wake word

This notebook trains a custom openWakeWord model for **"hey orion"** on a free Colab GPU,
using openWakeWord's synthetic-data pipeline. It's Phase 6 of
[Orion](https://github.com/shahwazreza/orion): see `docs/wake-word.md` there for what to do with the result.

**What happens**

1. A multi-speaker TTS model (Piper, LibriTTS voices) says "hey orion" tens of thousands of times,
   with varied speakers, speeds and intonation. It also says similar-sounding phrases to learn
   what *not* to react to.
2. Each clip is mixed with real room echoes (MIT impulse responses) and background audio (AudioSet).
3. A small network is trained against ~2,000 hours of pre-computed general audio (ACAV100M
   features), aiming for at most 0.5 false activations per hour (tighten it later with Orion's
   per-model threshold).
4. The result is exported as `hey_orion.onnx` (about 200 KB) for you to download.

**Privacy:** everything here is synthetic or public. Nothing from your computer, your voice, or
Orion's data is uploaded.

**Time and space:** on a free T4 GPU, about 1.5 to 3 hours with the default settings, and about
40 GB of disk (mostly the 16 GB negative-audio features). Run the cells in order:
*Runtime > Run all* works. If Colab disconnects, re-run from the top: finished steps are skipped.

**Before you start:** *Runtime > Change runtime type > T4 GPU*."""

GPU = """!nvidia-smi -L || echo "No GPU found: Runtime > Change runtime type > T4 GPU (CPU works but takes many hours)\""""

SETTINGS_MD = "## 1. Settings\nThe defaults are a good balance. Lower `N_SAMPLES` / `STEPS` only for a quick trial run."

SETTINGS = """TARGET_PHRASES = ["hey orion"]
MODEL_NAME = "hey_orion"

N_SAMPLES = 50_000        # synthetic "hey orion" clips for training (20k minimum, more is better)
N_SAMPLES_VAL = 5_000     # held out for validation
STEPS = 50_000            # training steps
# 0.2 pushed recall down to 49% in a first run (the trainer kept boosting negatives).
# 0.5 trades a few more false alarms for recall; tune further with Orion's per-model threshold.
TARGET_FP_PER_HOUR = 0.5  # aim for at most this many false activations per hour
LAYER_SIZE = 64           # 32 was too small for "hey orion" in a first run

# Phrases that sound alike but must NOT wake Orion. Add names of people you live with.
NEGATIVE_PHRASES = ["hey ryan", "hey brian", "o'brien", "hey arya", "hey lauren", "hey jarvis",
                    "hey siri", "orange", "oreo", "the onion", "all right then", "hey marian"]

N_BACKGROUND_CLIPS = 3_000  # AudioSet clips mixed in as background noise
N_RIRS = None               # room impulse responses to use (None = all ~270)
FULL_NEGATIVES = True       # download the 16 GB general-audio features (needed for a good model)"""

INSTALL_MD = "## 2. Install\nopenWakeWord is pinned to v0.6.0, the version Orion runs, so the model is compatible."

INSTALL = """!apt-get -qq install -y espeak-ng > /dev/null
![ -d openWakeWord ] || (git clone -q https://github.com/dscripka/openWakeWord && cd openWakeWord && git checkout -q v0.6.0)
![ -d piper-sample-generator ] || (git clone -q https://github.com/dscripka/piper-sample-generator && cd piper-sample-generator && git checkout -q f1988a4d54ed)
![ -f piper-sample-generator/models/en-us-libritts-high.pt ] || wget -q -O piper-sample-generator/models/en-us-libritts-high.pt https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt
!pip -q install -e ./openWakeWord --no-deps
# scipy<1.15: the `acoustics` package used for augmentation needs scipy.special.sph_harm (removed in 1.15).
!pip -q install onnxscript mutagen torchinfo "torchmetrics<1" "speechbrain<1" "audiomentations<1" "torch-audiomentations<1" acoustics pronouncing "datasets<3" deep-phonemizer==0.0.19 espeak-phonemizer webrtcvad pyyaml onnx onnxruntime soundfile "scipy<1.15\""""

DATA_MD = "## 3. Download room echoes, background audio and negative features"

DATA = """import io, os
# Colab can't see a package installed with `pip install -e` until the runtime restarts,
# so make the cloned openWakeWord importable directly.
import sys
if os.path.abspath("openWakeWord") not in sys.path:
    sys.path.insert(0, os.path.abspath("openWakeWord"))
import numpy as np
import scipy.io.wavfile
import soundfile as sf
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download

def save_wav(path, audio, sr=16000):
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    scipy.io.wavfile.write(path, sr, (audio * 32767).astype(np.int16))

# Room impulse responses (MIT, 16 kHz)
os.makedirs("mit_rirs", exist_ok=True)
if len(os.listdir("mit_rirs")) < 10:
    from datasets import load_dataset
    rirs = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for i, row in enumerate(rirs):
        if N_RIRS is not None and i >= N_RIRS:
            break
        save_wav(f"mit_rirs/rir_{i:04d}.wav", row["audio"]["array"], row["audio"]["sampling_rate"])
print(len(os.listdir("mit_rirs")), "room impulse responses")

# Background audio: AudioSet clips (FLAC inside parquet), resampled to 16 kHz
os.makedirs("background_clips", exist_ok=True)
if len(os.listdir("background_clips")) < min(N_BACKGROUND_CLIPS, 100):
    import pyarrow.parquet as pq
    saved, part = 0, 0
    while saved < N_BACKGROUND_CLIPS:
        path = hf_hub_download("agkphysics/AudioSet", f"data/bal_train/{part:02d}.parquet",
                               repo_type="dataset", local_dir="hf")
        table = pq.read_table(path, columns=["audio"])
        for row in table.column("audio").to_pylist():
            if saved >= N_BACKGROUND_CLIPS:
                break
            try:
                audio, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=True)
            except Exception:
                continue
            audio = audio.mean(axis=1)
            if sr != 16000:
                g = np.gcd(16000, sr)
                audio = resample_poly(audio, 16000 // g, sr // g)
            if len(audio) >= 16000:
                save_wav(f"background_clips/bg_{saved:05d}.wav", audio)
                saved += 1
        os.remove(path)          # free disk; each parquet holds ~500 clips
        part += 1
print(len(os.listdir("background_clips")), "background clips")

# Pre-computed openWakeWord features: general audio (negatives) and false-positive validation
for name in (["openwakeword_features_ACAV100M_2000_hrs_16bit.npy"] if FULL_NEGATIVES else []) + ["validation_set_features.npy"]:
    if not os.path.exists(name):
        hf_hub_download("davidscripka/openwakeword_features", name, repo_type="dataset", local_dir=".")
NEGATIVE_FEATURES = ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy" if FULL_NEGATIVES
                     else "validation_set_features.npy")
print("negative features:", NEGATIVE_FEATURES)

# openWakeWord's own feature models (melspectrogram + embedding)
import openwakeword.utils
openwakeword.utils.download_models(model_names=["embedding_model", "melspectrogram"])"""

CONFIG_MD = "## 4. Training config"

CONFIG = """import yaml
config = {
    "model_name": MODEL_NAME,
    "target_phrase": TARGET_PHRASES,
    "custom_negative_phrases": NEGATIVE_PHRASES,
    "n_samples": N_SAMPLES,
    "n_samples_val": N_SAMPLES_VAL,
    "tts_batch_size": 50,
    "augmentation_batch_size": 16,
    "piper_sample_generator_path": "./piper-sample-generator",
    "output_dir": "./output",
    "rir_paths": ["./mit_rirs"],
    "background_paths": ["./background_clips"],
    "background_paths_duplication_rate": [1],
    "false_positive_validation_data_path": "./validation_set_features.npy",
    "augmentation_rounds": 1,
    "feature_data_files": {"ACAV100M_sample": "./" + NEGATIVE_FEATURES},
    "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
    "model_type": "dnn",
    "layer_size": LAYER_SIZE,
    "steps": STEPS,
    "max_negative_weight": 1500,
    "target_false_positives_per_hour": TARGET_FP_PER_HOUR,
}
with open("hey_orion.yml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
print(open("hey_orion.yml").read())"""

PATCH_MD = ("## 5. Two compatibility fixes\n"
            "- **TTS checkpoint:** PyTorch 2.6+ refuses to load pickled checkpoints by default. The "
            "file's SHA-256 is checked against the official rhasspy release first, and only then "
            "loaded with `weights_only=False`.\n"
            "- **TensorFlow Lite:** Orion uses the ONNX model. The TFLite step needs "
            "`tensorflow-cpu==2.8.1`, which no longer installs on Colab, so it's removed.")

PATCH = """import hashlib
TTS = "piper-sample-generator/models/en-us-libritts-high.pt"
TTS_SHA256 = "5258f21adf77c29248d9a89d51ca9492f6e75a498140584dcbed0edcab81960a"
h = hashlib.sha256(open(TTS, "rb").read()).hexdigest()
assert h == TTS_SHA256, f"TTS checkpoint hash mismatch ({h}); delete it and re-run the install cell"
g = "piper-sample-generator/generate_samples.py"
src = open(g).read()
if "weights_only=False" not in src:
    open(g, "w").write(src.replace("torch.load(model_path)", "torch.load(model_path, weights_only=False)"))
print("TTS checkpoint verified")

p = "openWakeWord/openwakeword/train.py"
src = open(p).read()
if "ONNX only for Orion" not in src:          # safe to re-run
    start = src.index("        convert_onnx_to_tflite(")
    end = src.index(")\\n", start) + 2
    src = src[:start] + "        pass  # ONNX only for Orion (TFLite conversion removed)\\n" + src[end:]
    open(p, "w").write(src)
print("TFLite conversion disabled")

# Newer PyTorch defaults to a new ONNX exporter; openWakeWord was written for the classic one.
src = open(p).read()
if "dynamo=False" not in src:
    src = src.replace('model_name + ".onnx"), opset_version=13)',
                      'model_name + ".onnx"), opset_version=13, dynamo=False)')
    open(p, "w").write(src)
print("Classic ONNX exporter:", "dynamo=False" in src)"""

TORCHAUDIO_MD = ("## 5b. torchaudio compatibility\n"
                 "torchaudio 2.9+ (on Colab now) removed `torchaudio.info` and moved `torchaudio.load` "
                 "to a separate package, but openWakeWord v0.6.0 and torch-audiomentations still call "
                 "both. This installs a small shim, loaded by every Python process (including the "
                 "training steps), that reads audio with soundfile instead. It does nothing on older torchaudio.")
TORCHAUDIO = """import os, site
SHIM = \'\'\'
try:
    import torchaudio
    if not hasattr(torchaudio, "info"):        # torchaudio 2.9+ only
        import soundfile as sf
        import torch

        class _Info:
            def __init__(self, path):
                i = sf.info(str(path))
                self.num_frames, self.sample_rate, self.num_channels = i.frames, i.samplerate, i.channels

        def _info(path, *args, **kwargs):
            return _Info(path)

        def _load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, *args, **kwargs):
            frames = num_frames if num_frames and num_frames > 0 else -1
            data, sr = sf.read(str(path), start=frame_offset, frames=frames, dtype="float32", always_2d=True)
            t = torch.from_numpy(data.T.copy())
            return (t if channels_first else t.T), sr

        torchaudio.info, torchaudio.load = _info, _load
except Exception:
    pass                                         # never stop Python starting
\'\'\'
sp = site.getsitepackages()[0]
open(os.path.join(sp, "torchaudio_compat.py"), "w").write(SHIM)
open(os.path.join(sp, "torchaudio_compat.pth"), "w").write("import torchaudio_compat\\n")
import subprocess
print(subprocess.run(["python", "-c", "import torchaudio, glob; f = sorted(glob.glob('mit_rirs/*.wav'))[0]; "
                      "w, sr = torchaudio.load(f); print('torchaudio', torchaudio.__version__, 'shim OK:', tuple(w.shape), sr)"],
                     capture_output=True, text=True).stdout.strip())"""

GEN_MD = "## 6. Generate synthetic clips\nThe longest step: tens of thousands of TTS clips."
GEN = "!python openWakeWord/openwakeword/train.py --training_config hey_orion.yml --generate_clips"
AUG_MD = ("## 7. Add room echo and background noise, compute features\n"
          "openWakeWord skips this step if *any* feature file exists, so an interrupted run "
          "would leave the rest missing. This cell checks all four and redoes the step if needed.")
AUG = """import os, subprocess
feat = f"output/{MODEL_NAME}"
for d in ("positive_train", "positive_test", "negative_train", "negative_test"):
    n = len(os.listdir(os.path.join(feat, d))) if os.path.isdir(os.path.join(feat, d)) else 0
    print(f"{d}: {n} clips")
    assert n > 0, f"No {d} clips: re-run step 6 (generate) first"
needed = [f"{feat}/{k}_features_{s}.npy" for k in ("positive", "negative") for s in ("train", "test")]
missing = [f for f in needed if not os.path.exists(f)]
if missing:
    cmd = ["python", "openWakeWord/openwakeword/train.py", "--training_config", "hey_orion.yml", "--augment_clips"]
    if len(missing) < len(needed):
        print("Incomplete earlier run, redoing:", ", ".join(os.path.basename(f) for f in missing))
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True)
missing = [f for f in needed if not os.path.exists(f)]
assert not missing, f"Still missing {missing}: Colab may have run out of memory (see the output above)"
print("All four feature files ready")"""
TRAIN_MD = "## 8. Train"
TRAIN = "!python openWakeWord/openwakeword/train.py --training_config hey_orion.yml --train_model"

CHECK_MD = "## 9. Check the model and download it"
CHECK = """import glob, os, numpy as np, scipy.io.wavfile
# Colab can't see a package installed with `pip install -e` until the runtime restarts,
# so make the cloned openWakeWord importable directly.
import sys
if os.path.abspath("openWakeWord") not in sys.path:
    sys.path.insert(0, os.path.abspath("openWakeWord"))
from openwakeword.model import Model

onnx_path = f"output/{MODEL_NAME}.onnx"
m = Model(wakeword_models=[onnx_path], inference_framework="onnx")

def max_score(path):
    m.reset()
    sr, x = scipy.io.wavfile.read(path)
    x = np.concatenate([np.zeros(16000, np.int16), x.astype(np.int16), np.zeros(8000, np.int16)])
    x = x[: len(x) // 1280 * 1280]
    return max(max(m.predict(f).values()) for f in x.reshape(-1, 1280))

pos = sorted(glob.glob(f"output/{MODEL_NAME}/positive_test/*.wav"))[:200]
neg = sorted(glob.glob(f"output/{MODEL_NAME}/negative_test/*.wav"))[:200]
p = np.array([max_score(f) for f in pos]); n = np.array([max_score(f) for f in neg])
for t in (0.3, 0.5, 0.7):
    print(f"threshold {t}: detects {np.mean(p >= t):.0%} of held-out 'hey orion' clips, "
          f"fires on {np.mean(n >= t):.0%} of similar-sounding phrases")
print(onnx_path, os.path.getsize(onnx_path) // 1024, "KB")
try:
    from google.colab import files
    files.download(onnx_path)
except ImportError:
    pass"""

NEXT = """## 10. Use it in Orion

1. Copy the downloaded `hey_orion.onnx` to `~/orion/models/openwakeword/hey_orion.onnx`.
2. Test it with your own voice before switching: `python scripts/wakeword_test.py`.
3. In `config.yaml`: `models: [models/openwakeword/hey_orion.onnx, hey_jarvis]` to use both while you
   get used to it, then just the new one. Full details: `docs/wake-word.md`."""

CELLS = [("md", INTRO), ("code", GPU), ("md", SETTINGS_MD), ("code", SETTINGS), ("md", INSTALL_MD),
         ("code", INSTALL), ("md", DATA_MD), ("code", DATA), ("md", CONFIG_MD), ("code", CONFIG),
         ("md", PATCH_MD), ("code", PATCH), ("md", TORCHAUDIO_MD), ("code", TORCHAUDIO), ("md", GEN_MD), ("code", GEN), ("md", AUG_MD), ("code", AUG),
         ("md", TRAIN_MD), ("code", TRAIN), ("md", CHECK_MD), ("code", CHECK), ("md", NEXT)]


def nb_cell(kind, src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


nb = {"cells": [nb_cell(k, s) for k, s in CELLS],
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT_NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")

# Smoke test: same cells, tiny settings, "!cmd" lines run through the shell. Skips GPU + install.
SMOKE_SETTINGS = SETTINGS + """
N_SAMPLES, N_SAMPLES_VAL, STEPS = 120, 40, 300
N_BACKGROUND_CLIPS, N_RIRS, FULL_NEGATIVES = 60, 20, False"""


def py(src):
    out = []
    for line in src.split("\n"):
        m = re.match(r"^(\s*)!(.*)$", line)
        out.append(f"{m.group(1)}__import__('subprocess').run({m.group(2)!r}, shell=True, check=True)"
                   if m else line)
    return "\n".join(out)


smoke = "\n\n".join(["import os, sys", SMOKE_SETTINGS, py(DATA), py(CONFIG), py(PATCH),
                     py(GEN).replace("'python ", f"'{sys.executable} "),
                     py(AUG).replace("'python ", f"'{sys.executable} "),
                     py(TRAIN).replace("'python ", f"'{sys.executable} "), py(CHECK)])
if OUT_SMOKE:
    OUT_SMOKE.write_text(smoke + "\n")
print("wrote", OUT_NOTEBOOK, *(["and", OUT_SMOKE] if OUT_SMOKE else []))
