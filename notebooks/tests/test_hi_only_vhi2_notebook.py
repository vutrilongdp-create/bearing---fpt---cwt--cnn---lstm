import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "hi_fpt_prepare_kaggle_vhi2.ipynb"


def test_hi_only_notebook_has_no_cwt_computation():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "import pywt" not in code
    assert "skimage" not in code
    assert "compute_cwt" not in code
    assert "fit_train_cwt_scaler" not in code


def test_hi_only_notebook_contains_vhi2_detector_and_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for source in code_cells:
        ast.parse(source)
    code = "\n".join(code_cells)
    assert "def find_confirmed_fpt" in code
    assert "FPT_CONFIRM_WINDOW = 30" in code
    assert "FPT_CONFIRM_REQUIRED = 20" in code
    assert "hi_fpt_summary.csv" in code
    assert "y_rms_ewma" in code
