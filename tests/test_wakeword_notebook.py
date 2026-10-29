"""The Colab notebook is generated from scripts/wakeword/build_notebook.py and must match it."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_hey_orion.ipynb"


def test_notebook_is_up_to_date_with_its_generator(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(ROOT / "scripts/wakeword/build_notebook.py"), "--out", str(out)],
                   check=True, capture_output=True)
    assert json.loads(out.read_text()) == json.loads(NOTEBOOK.read_text()), \
        "Notebook edited by hand or not rebuilt: run python scripts/wakeword/build_notebook.py"


def test_code_cells_are_valid_python():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            ast.parse("\n".join("pass" if line.lstrip().startswith("!") else line for line in src.split("\n")))


def test_every_compatibility_fix_is_present():
    src = "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"])
    for needle in ("checkout -q v0.6.0", '"scipy<1.15', "onnxscript", "weights_only=False",
                   "5258f21adf77c29248d9a89d51ca9492f6e75a498140584dcbed0edcab81960a", "ONNX only for Orion",
                   "dynamo=False", 'sys.path.insert(0, os.path.abspath("openWakeWord"))', "torchaudio_compat",
                   "All four feature files ready", "TARGET_FP_PER_HOUR = 0.5", "LAYER_SIZE = 64"):
        assert needle in src, needle
