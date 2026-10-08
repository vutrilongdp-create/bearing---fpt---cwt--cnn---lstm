import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "02b_cwt_paper_figures_kaggle.ipynb"


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


def test_notebook_preserves_task2_cwt_contract():
    code = notebook_code()
    required = [
        'CWT_WAVELET = "morl"',
        "np.geomspace(1, 512, 128)",
        "IMAGE_SHAPE = (128, 128)",
        "LOG_POWER_EPS = 1e-3",
        "np.log2(np.abs(coef) ** 2 + LOG_POWER_EPS)",
        "preserve_range=True",
        "anti_aliasing=True",
        "purpose\": \"paper_only_cwt_figures_not_for_training_or_evaluation",
    ]
    for token in required:
        assert token in code, f"Missing required token: {token}"


def test_notebook_uses_publication_figure_style():
    code = notebook_code()
    required = [
        '"Times New Roman"',
        '"font.family": "serif"',
        'COLORBAR_LABEL = "CWT log-power"',
        'TITLE_FONTSIZE = 12',
        'CHANNEL_TITLE_FONTSIZE = 11',
        'fig.colorbar(plotted, ax=axes.ravel().tolist(), label=COLORBAR_LABEL',
        'fig.colorbar(plotted, ax=axes.ravel().tolist(), label=COLORBAR_LABEL, shrink=0.82)',
        'return "pre-FPT"',
        'return "post-FPT"',
    ]
    for token in required:
        assert token in code, f"Missing publication style token: {token}"


def test_notebook_does_not_touch_training_cache_or_official_test():
    code = notebook_code()
    forbidden = [
        "cwt_unscaled_cache",
        "_cwt_unscaled.npz",
        "StandardScaler",
        "MinMaxScaler",
        "fit_transform",
        "Test_set",
        "Full_Test_Set",
    ]
    for token in forbidden:
        assert token not in code, f"Forbidden token found: {token}"


def test_notebook_includes_expected_paper_examples():
    code = notebook_code()
    required = [
        "THREE_TIMEPOINT_BEARINGS = list(FROZEN_FPT)",
        'f"{bearing}_three_timepoint_cwt"',
        "bearing3_1_before_fpt",
        "bearing3_1_at_fpt",
        "bearing3_1_after_fpt",
        "bearing1_1_early_healthy",
        "bearing1_2_early_healthy",
        "bearing3_1_causal_sequence_at_fpt",
        "bearing1_2_causal_sequence_at_fpt",
    ]
    for token in required:
        assert token in code, f"Missing figure example: {token}"


def test_notebook_includes_three_timepoint_visualization_contract():
    code = notebook_code()
    required = [
        "def three_timepoint_indices",
        '("Start", 0)',
        '("Frozen FPT", int(fpt_index))',
        '("End", int(n_files - 1))',
        "def save_three_timepoint_panel",
        "Expected [3,2,128,128]",
        '"kind": "three_timepoint_panel"',
        '"timepoints": [',
        'f"{bearing}: CWT log-power | frozen FPT={fpt_index}"',
    ]
    for token in required:
        assert token in code, f"Missing three-timepoint token: {token}"


def test_after_fpt_examples_are_resolved_against_bearing_length():
    code = notebook_code()
    assert '"index": 520' not in code
    assert '"offset_from_fpt": 27' in code
    assert "def resolve_figure_index" in code
    assert "min(n_files - 1, fpt_index + offset)" in code
