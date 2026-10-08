import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01b_irrms_paper_audit_kaggle.ipynb"


def notebook_code() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    ]
    return "\n".join(cells)


def test_notebook_code_parses_and_has_empty_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    ast.parse(notebook_code())


def test_notebook_preserves_frozen_irrms_method():
    code = notebook_code()
    required = [
        "HEALTHY_FILES = 100",
        "IRRMS_WINDOW = 30",
        "FPT_CONSECUTIVE = 5",
        "RECOVERY_RUN = 30",
        "SEARCH_START = 100",
        "STD_DDOF = 0",
        'THRESHOLD_OPERATOR = ">"',
        "FROZEN_IRRMS_THRESHOLD = 1.2637933790683746",
        "def rms_two_channel",
        "def relative_rms",
        "def causal_irrms",
        "def evaluate_irrms_fpt",
        "baseline_irrms_quantiles",
    ]
    for token in required:
        assert token in code, f"Missing required token: {token}"


def test_notebook_checks_frozen_fpt_values():
    code = notebook_code()
    required = [
        '"bearing1_1": 1871',
        '"bearing1_2": 826',
        '"bearing2_1": 151',
        '"bearing2_2": 198',
        '"bearing3_1": 493',
        '"bearing3_2": 1597',
        "recomputed FPT",
        "q99_delta > 1e-7",
    ]
    for token in required:
        assert token in code, f"Missing frozen FPT check: {token}"


def test_notebook_is_paper_audit_only():
    code = notebook_code()
    required = [
        "paper_only_irrms_method_audit_and_figures",
        "irrms_paper_audit.zip",
        "irrms_paper_audit_manifest.json",
        "all_bearings_irrms_fpt_overview.png",
    ]
    for token in required:
        assert token in code, f"Missing output token: {token}"
    forbidden = [
        "Test_set",
        "Full_Test_Set",
        "pywt",
        "torch",
        "optimizer",
        "EarlyStopping",
        "StandardScaler",
        "MinMaxScaler",
        "fit_transform",
    ]
    for token in forbidden:
        assert token not in code, f"Forbidden token found: {token}"
