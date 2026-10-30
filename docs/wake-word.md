# Custom wake word: "hey orion" (Phase 6)

openWakeWord has no pretrained "hey orion", so Orion started with its built-in
"hey jarvis". This guide trains "hey orion" on a free Google Colab GPU and swaps it
in. Nothing personal is involved: the training data is synthetic speech and public
audio.

## 1. Train it on Colab

1. Open [`notebooks/train_hey_orion.ipynb`](../notebooks/train_hey_orion.ipynb) in
   Google Colab: on GitHub, open the file and use **Open in Colab**, or upload it at
   [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU.**
3. Optional: in the *Settings* cell, add the names of people you live with to
   `NEGATIVE_PHRASES` (e.g. `"hey ryan"`), so their names don't wake Orion.
4. **Runtime → Run all.** It takes about 1.5 to 3 hours with the defaults and about
   40 GB of Colab disk. If Colab disconnects, run all again: finished steps are skipped.
5. The last cell prints how the model does on held-out clips and downloads
   `hey_orion.onnx` (about 200 KB).

What the notebook does:

| Step | What |
|---|---|
| Generate | Piper TTS (LibriTTS, hundreds of voices) says "hey orion" 50,000 times with varied speed and intonation, plus similar-sounding phrases ("hey ryan", "o'brien") to learn what *not* to react to |
| Augment | Mixes in real room echoes (MIT impulse responses) and background audio (AudioSet) |
| Train | A small network (64 units) against ~2,000 hours of general audio (ACAV100M features), targeting at most 0.5 false activations per hour |
| Export | ONNX only, with PyTorch's classic exporter. The TensorFlow Lite step is removed because its pinned TensorFlow no longer installs on Colab |

Versions are pinned: openWakeWord **v0.6.0** (what Orion runs), piper-sample-generator
commit `f1988a4`, and the TTS checkpoint is verified by SHA-256 before it's loaded.

**Why the training targets:** the first full run used openWakeWord's defaults (0.2 false
activations per hour, 32 units) and reached only **49% recall** with 0.97 false activations
per hour on the validation set. Aiming for 0.2 makes the trainer keep boosting the weight on
negatives at the cost of recall. The notebook now aims for 0.5 with a 64-unit network;
false alarms are then tuned with Orion's per-model threshold (`wakeword.thresholds`).

### Compatibility fixes built into the notebook

Each was found running the pipeline on current Colab (Python 3.13, recent PyTorch):

| Problem | Fix |
|---|---|
| `acoustics` uses `scipy.special.sph_harm`, removed in SciPy 1.15 | `scipy<1.15` |
| PyTorch 2.6+ refuses to load the pickled TTS checkpoint | SHA-256 verified, then `weights_only=False` |
| TFLite export needs `tensorflow-cpu==2.8.1` (no longer installable) | ONNX only |
| PyTorch's new default ONNX exporter crashed the export after training | `dynamo=False` (classic exporter) + `onnxscript` |
| Colab can't import a `pip install -e` package until the runtime restarts | Cloned openWakeWord added to `sys.path` |
| torchaudio 2.9+ removed `torchaudio.info` and moved `torchaudio.load` | A soundfile-based shim, loaded in every process |
| Step 7 skips itself if any feature file exists, so a crash left the rest missing | Checks all four files and redoes the step if needed |

### Changing the notebook

The notebook is generated: edit `scripts/wakeword/build_notebook.py`, then run
`python scripts/wakeword/build_notebook.py`. A test fails if the notebook and the
generator disagree. `--smoke smoke.py` also writes a tiny local test run of the same
cells (it runs third-party training code, so use a throwaway environment).

## 2. Test it with your voice

Copy the file to `~/orion/models/openwakeword/hey_orion.onnx`, stop Orion, then:

```bash
python scripts/wakeword_test.py
```

It compares every model at once (your configured ones plus `hey_orion.onnx`):

1. **Detection:** you say the phrase 10 times when prompted, with different
   distances and tones. A good model fires on at least 9 of 10.
2. **False alarms:** 5 minutes of normal talk, music or a video without saying it.
   Aim for 0 or 1 (each per-5-minute alarm is 12 per hour).

Options: `--say 15`, `--listen 10`, `--threshold 0.6`, `--models a.onnx b.onnx`.
Nothing is recorded.

## 3. Switch Orion over

In `config.yaml`:

```yaml
  wakeword:
    models: [models/openwakeword/hey_orion.onnx, hey_jarvis]   # both work while you get used to it
    phrase: "hey orion"                                         # what Orion tells people to say
    thresholds: {hey_orion: 0.5}                                # tune from the test above
```

Then, once you're happy, drop `hey_jarvis` from the list. Orion strips the wake
phrase from what you say before it reaches the language model, so he won't
correct you about his name either way.

**Tuning from the test:** if `hey_orion` misses you, lower its threshold (e.g.
`0.4`). If it fires on its own, raise it (`0.6`) or retrain with more
`NEGATIVE_PHRASES`, a higher `N_SAMPLES`, or a lower `TARGET_FP_PER_HOUR`.

## Limitations

- The training voices are LibriTTS (mostly American English). It usually works
  well for other accents, but test with `wakeword_test.py` before relying on it.
- The trained model isn't in the repo or `download_models.py` (it's yours to
  train), and `.gitignore` excludes `models/`.
- openWakeWord's pretrained "hey jarvis" is licensed for non-commercial use only
  (CC BY-NC-SA 4.0). A model you train yourself avoids that restriction.
