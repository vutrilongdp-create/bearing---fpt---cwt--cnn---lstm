import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "cwt_hi_cnn_prepare_kaggle_vhi2.ipynb"


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
        "FPT_CONFIRM_WINDOW": 30,
        "FPT_CONFIRM_REQUIRED": 20,
        "EPS": 1e-8,
    }
    exec(source, namespace)
    return namespace


def test_five_consecutive_without_twenty_of_thirty_is_rejected():
    find_fpt = load_hi_namespace()["find_confirmed_fpt"]
    values = np.zeros(40, dtype=np.float32)
    values[3:8] = 2.0
    assert find_fpt(values, 1.0, 0, 5, 30, 20) is None


def test_twenty_of_thirty_without_first_five_consecutive_is_rejected():
    find_fpt = load_hi_namespace()["find_confirmed_fpt"]
    values = np.full(40, 2.0, dtype=np.float32)
    values[2] = 0.0
    values[7] = 0.0
    assert find_fpt(values, 1.0, 0, 5, 30, 20) == 8


def test_confirmed_run_returns_first_candidate_index():
    find_fpt = load_hi_namespace()["find_confirmed_fpt"]
    values = np.zeros(45, dtype=np.float32)
    values[4:9] = 2.0
    values[10:25] = 2.0
    values[33] = 2.0
    assert find_fpt(values, 1.0, 0, 5, 30, 20) == 4


def test_vhi2_notebook_code_cells_parse_and_record_confirmation_config():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for source in code_cells:
        ast.parse(source)
    joined = "\n".join(code_cells)
    assert "FPT_CONFIRM_WINDOW = 30" in joined
    assert "FPT_CONFIRM_REQUIRED = 20" in joined
    assert "def find_confirmed_fpt" in joined
