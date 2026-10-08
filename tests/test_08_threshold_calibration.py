import ast
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "08_threshold_calibration.ipynb"


def notebook_code_cells() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def load_calibration_namespace():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code_cells = [
        "".join(cell["source"])
        for cell in cells
        if cell.get("cell_type") == "code"
    ]
    target_cells = [
        cell
        for cell in code_cells
        if "def detect_fpt_from_predictions" in cell
        or "def select_threshold_from_inner_val" in cell
    ]
    ns = {
        "np": np,
        "Path": Path,
        "PROBABILITY_THRESHOLD": 0.5,
        "FPT_CONSECUTIVE": 5,
        "FILE_INTERVAL_SECONDS": 10,
        "THRESHOLD_GRID": [round(float(x), 2) for x in np.arange(0.10, 0.951, 0.05)],
        "CALIBRATION_SELECTION_RULE": [
            "prefer_not_missed",
            "prefer_no_pre_fpt_alarm_run",
            "minimize_abs_fpt_error_files",
            "minimize_pre_fpt_positive_sample_count",
            "prefer_higher_threshold",
        ],
    }
    for cell in target_cells:
        exec(cell, ns)
    return ns


def test_threshold_grid_exact_values():
    ns = load_calibration_namespace()
    expected = [round(float(x), 2) for x in np.arange(0.10, 0.951, 0.05)]
    assert ns["THRESHOLD_GRID"] == expected
    assert expected[0] == 0.10
    assert expected[-1] == 0.95
    assert len(expected) == 18


def test_detect_fpt_returns_first_five_consecutive_index():
    ns = load_calibration_namespace()
    probs = np.array([0.1, 0.7, 0.8, 0.9, 0.6, 0.51, 0.2, 0.9])
    assert ns["detect_fpt_from_predictions"](probs, threshold=0.5, consecutive=5) == 1


def test_detect_fpt_returns_none_when_run_is_short():
    ns = load_calibration_namespace()
    probs = np.array([0.9, 0.9, 0.9, 0.9, 0.1])
    assert ns["detect_fpt_from_predictions"](probs, threshold=0.5, consecutive=5) is None


def make_records(file_indices, probs):
    return [
        {
            "file_index": int(idx),
            "target_index": int(idx),
            "y_true": float(idx >= 10),
            "y_prob": float(prob),
        }
        for idx, prob in zip(file_indices, probs)
    ]


def test_event_metrics_exact_detection():
    ns = load_calibration_namespace()
    records = make_records(
        range(15),
        [0.1] * 10 + [0.7, 0.8, 0.9, 0.9, 0.9],
    )
    result = ns["compute_event_metrics"](
        records, reference_fpt=10, threshold=0.5, consecutive=5
    )
    assert result["predicted_fpt"] == 10
    assert result["signed_delay_files"] == 0
    assert result["abs_fpt_error_files"] == 0
    assert result["pre_fpt_alarm_run"] is False
    assert result["missed_detection"] is False


def test_event_metrics_early_alarm_and_missed_detection():
    ns = load_calibration_namespace()
    early_records = make_records(
        range(15),
        [0.9, 0.9, 0.9, 0.9, 0.9] + [0.1] * 10,
    )
    early = ns["compute_event_metrics"](
        early_records, reference_fpt=10, threshold=0.5, consecutive=5
    )
    assert early["predicted_fpt"] == 0
    assert early["signed_delay_files"] == -10
    assert early["pre_fpt_alarm_run"] is True
    assert early["missed_detection"] is False

    missed_records = make_records(range(15), [0.1] * 15)
    missed = ns["compute_event_metrics"](
        missed_records, reference_fpt=10, threshold=0.5, consecutive=5
    )
    assert missed["predicted_fpt"] is None
    assert missed["missed_detection"] is True


def test_threshold_selection_prefers_no_miss_then_no_pre_alarm_then_error_then_higher_threshold():
    ns = load_calibration_namespace()

    # threshold 0.50: early alarm at 0; threshold 0.70: exact detection at 10.
    records = make_records(
        range(15),
        [0.6, 0.6, 0.6, 0.6, 0.6] + [0.1] * 5 + [0.8, 0.8, 0.8, 0.8, 0.8],
    )
    selected = ns["select_threshold_from_inner_val"](
        records, reference_fpt=10, threshold_grid=[0.50, 0.70, 0.90]
    )
    assert selected["selected_threshold"] == 0.70
    assert selected["inner_val_event_metrics"]["predicted_fpt"] == 10
    assert selected["inner_val_event_metrics"]["pre_fpt_alarm_run"] is False


def test_threshold_selection_uses_higher_threshold_as_final_tie_breaker():
    ns = load_calibration_namespace()
    records = make_records(
        range(15),
        [0.1] * 10 + [0.8, 0.8, 0.8, 0.8, 0.8],
    )
    selected = ns["select_threshold_from_inner_val"](
        records, reference_fpt=10, threshold_grid=[0.50, 0.70]
    )
    assert selected["selected_threshold"] == 0.70


def test_notebook_code_cells_parse_and_include_required_tokens():
    joined = notebook_code_cells()
    ast.parse(joined)
    required = [
        "THRESHOLD_GRID",
        "select_threshold_from_inner_val",
        "inner_val_records",
        "selected_threshold",
        "threshold_calibration_manifest.json",
        "summary_metrics_calibrated.json",
        "create_artifact_zip",
        "fpt_threshold_calibration_artifacts.zip",
        "all decisions frozen",
        "inner-validation threshold calibration",
    ]
    for token in required:
        assert token in joined, f"Missing required token: {token}"


def test_notebook_forbidden_tokens_absent():
    joined = notebook_code_cells()
    forbidden = [
        "Test_set",
        "Full_Test_Set",
        "BatchNorm",
        "StandardScaler",
        "MinMaxScaler",
        "fit_transform",
        "select_threshold(test_predictions)",
        "outer_test_metric[t]",
        "tune_grid_after_results",
    ]
    for token in forbidden:
        assert token not in joined, f"Forbidden token found: {token}"


def test_inner_val_threshold_selection_precedes_outer_test_event_evaluation():
    joined = notebook_code_cells()
    select_pos = joined.index("selected_threshold_info = select_threshold_from_inner_val")
    outer_eval_pos = joined.index("outer_event_metrics_calibrated = compute_event_metrics")
    assert select_pos < outer_eval_pos
