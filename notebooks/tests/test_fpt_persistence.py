import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).parents[1] / "cwt_hi_cnn_prepare_kaggle.ipynb"


def load_hi_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "def ewma_causal" in "".join(cell["source"])
    )
    namespace = {
        "np": np,
        "EWMA_ALPHA": 0.08,
        "HEALTHY_FILES": 200,
        "HI_SIGMA_SPAN": 6.0,
        "FPT_CONSECUTIVE": 5,
        "EPS": 1e-8,
    }
    exec(source, namespace)
    return namespace


def test_generated_notebook_contains_valid_persistent_fpt_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for source in code_cells:
        ast.parse(source)

    joined = "\n".join(code_cells)
    assert "FPT_CONSECUTIVE = 5" in joined
    assert "def find_persistent_fpt" in joined
    assert "'fpt_found': bool(fpt_found)" in joined


def test_four_consecutive_exceedances_do_not_create_fpt():
    find_fpt = load_hi_namespace()["find_persistent_fpt"]
    values = np.array([0.0, 0.0, 1.1, 1.2, 1.3, 1.4, 0.0])
    assert find_fpt(values, threshold=1.0, start_index=2, consecutive=5) is None


def test_five_consecutive_exceedances_return_first_index():
    find_fpt = load_hi_namespace()["find_persistent_fpt"]
    values = np.array([0.0, 0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.0])
    assert find_fpt(values, threshold=1.0, start_index=2, consecutive=5) == 2


def test_short_alarm_before_persistent_run_is_ignored():
    find_fpt = load_hi_namespace()["find_persistent_fpt"]
    values = np.array([0.0, 1.1, 1.2, 0.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert find_fpt(values, threshold=1.0, start_index=1, consecutive=5) == 4
