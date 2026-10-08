import ast
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "01_rrms_fpt_reference_kaggle.ipynb"


def load_reference_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    source = next(s for s in code_cells if "def rms_two_channel" in s)
    namespace = {
        "np": np,
        "HEALTHY_FILES": 100,
        "IRRMS_WINDOW": 30,
        "FPT_CONSECUTIVE": 5,
        "RECOVERY_RUN": 30,
        "SEARCH_START": 100,
        "STD_DDOF": 0,
        "THRESHOLD_OPERATOR": ">",
        "NEAR_END_MIN_DEGRADED_FILES": 30,
        "EPS": 1e-8,
    }
    exec(source, namespace)
    return namespace


def expected_irrms_value(window):
    window = np.asarray(window, dtype=np.float64)
    x = np.arange(len(window), dtype=np.float64)
    x_centered = x - x.mean()
    slope = np.dot(x_centered, window - window.mean()) / np.dot(
        x_centered, x_centered
    )
    intercept = window.mean() - slope * x.mean()
    trend = slope * x + intercept
    residual = window - trend
    lower = residual.mean() - 3.0 * residual.std(ddof=0)
    upper = residual.mean() + 3.0 * residual.std(ddof=0)
    current = residual[-1]
    if current <= lower:
        return float(window.mean()), "lower"
    if current < upper:
        return float(trend[-1]), "middle"
    return float(trend[-1] + upper), "upper"


def test_two_channel_rms():
    fn = load_reference_namespace()["rms_two_channel"]
    record = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    assert fn(record) == pytest.approx(5.0)


def test_rrms_uses_only_first_100_samples():
    fn = load_reference_namespace()["relative_rms"]
    rms = np.r_[np.ones(100) * 2.0, np.ones(20) * 20.0]
    rrms, norm = fn(rms, healthy_files=100)
    assert norm == pytest.approx(2.0)
    assert np.allclose(rrms[:100], 1.0)
    assert np.allclose(rrms[100:], 10.0)


def test_causal_irrms_has_nan_warmup_and_does_not_use_future_values():
    fn = load_reference_namespace()["causal_irrms"]
    base = np.linspace(0.9, 1.1, 60, dtype=np.float32)
    changed_future = base.copy()
    changed_future[30:] = 100.0
    irrms_base = fn(base, window=30)
    irrms_changed = fn(changed_future, window=30)
    assert np.isnan(irrms_base[:29]).all()
    assert irrms_changed[29] == pytest.approx(irrms_base[29], abs=1e-7)


def test_exact_linear_rrms_follows_current_trend_not_window_mean():
    fn = load_reference_namespace()["causal_irrms"]
    rrms = (1.0 + np.arange(100, dtype=np.float32) / 32.0).astype(np.float32)
    irrms = fn(rrms, window=30)
    assert irrms[-1] == pytest.approx(rrms[-1], abs=1e-7)
    assert irrms[-1] != pytest.approx(rrms[-30:].mean(), abs=1e-7)


@pytest.mark.parametrize(
    ("window", "expected_branch"),
    [
        (np.r_[np.linspace(1.0, 1.2, 29), 0.2], "lower"),
        (
            np.linspace(1.0, 1.2, 30)
            + 0.01 * np.sin(np.linspace(0.0, 2.0 * np.pi, 30)),
            "middle",
        ),
        (np.r_[np.linspace(1.0, 1.2, 29), 2.2], "upper"),
    ],
)
def test_causal_irrms_implements_all_three_residual_branches(
    window, expected_branch
):
    expected, actual_branch = expected_irrms_value(window)
    assert actual_branch == expected_branch
    actual = load_reference_namespace()["causal_irrms"](window, window=30)
    assert actual[-1] == pytest.approx(expected, abs=1e-6)


def test_four_consecutive_irrms_exceedances_are_rejected():
    fn = load_reference_namespace()["find_irrms_fpt"]
    irrms = np.r_[np.ones(40), np.ones(4) * 1.2]
    assert fn(irrms, threshold=1.1, start_index=0) is None


def test_five_consecutive_irrms_exceedances_return_first_index():
    fn = load_reference_namespace()["find_irrms_fpt"]
    irrms = np.r_[np.ones(40), np.ones(5) * 1.2]
    assert fn(irrms, threshold=1.1, start_index=0) == 40


def test_candidate_is_rejected_after_long_irrms_recovery():
    fn = load_reference_namespace()["evaluate_irrms_fpt"]
    irrms = np.r_[
        np.ones(100),
        np.ones(10) * 1.2,
        np.ones(30),
        np.ones(10) * 1.3,
    ]
    fpt, rejected = fn(irrms, threshold=1.1, start_index=100)
    assert fpt == 140
    assert rejected == 1


def test_first_run_start_returns_episode_start():
    fn = load_reference_namespace()["_first_run_start"]
    mask = np.array([False, True, True, True, True, True, False])
    assert fn(mask, 5) == 1
    assert fn(mask, 6) is None


def test_nan_warmup_never_counts_as_an_exceedance():
    fn = load_reference_namespace()["find_irrms_fpt"]
    irrms = np.r_[np.full(29, np.nan), np.ones(5) * 1.2]
    assert fn(irrms, threshold=1.1, start_index=0) == 29


def test_search_start_prevents_an_earlier_fpt():
    fn = load_reference_namespace()["find_irrms_fpt"]
    irrms = np.r_[np.ones(90), np.ones(20) * 1.2]
    assert fn(irrms, threshold=1.1, start_index=100) == 100


def test_baseline_quantiles_use_71_finite_points_per_bearing():
    fn = load_reference_namespace()["baseline_irrms_quantiles"]
    first = np.r_[np.full(29, np.nan), np.linspace(0.90, 1.05, 71), [9.0]]
    second = np.r_[np.full(29, np.nan), np.linspace(0.95, 1.10, 71), [9.0]]
    result = fn({"bearing_a": first, "bearing_b": second}, start=29, stop=100)
    pooled = np.r_[first[29:100], second[29:100]]
    assert result["q99"] == pytest.approx(np.quantile(pooled, 0.99))
    assert result["q99_5"] == pytest.approx(np.quantile(pooled, 0.995))


def test_baseline_quantiles_reject_missing_finite_values():
    fn = load_reference_namespace()["baseline_irrms_quantiles"]
    invalid = np.ones(120, dtype=np.float32)
    invalid[50] = np.nan
    with pytest.raises(ValueError, match="71 finite"):
        fn({"bearing_a": invalid}, start=29, stop=100)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                fpt_found=True,
                fpt_idx=120,
                baseline_alarm_runs=0,
                degraded_count=50,
            ),
            "ok",
        ),
        (
            dict(
                fpt_found=False,
                fpt_idx=None,
                baseline_alarm_runs=1,
                degraded_count=None,
            ),
            "baseline_crossing;not_found",
        ),
        (
            dict(
                fpt_found=True,
                fpt_idx=100,
                baseline_alarm_runs=0,
                degraded_count=20,
            ),
            "search_start_hit;near_end",
        ),
    ],
)
def test_boundary_flags_are_readable_and_keep_missing_counts_unknown(kwargs, expected):
    fn = load_reference_namespace()["build_boundary_flags"]
    assert fn(**kwargs) == expected


def test_notebook_cells_parse_and_record_sensitivity_contract():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for source in code_cells:
        ast.parse(source)
    joined = "\n".join(code_cells)
    assert "HEALTHY_FILES = 100" in joined
    assert "IRRMS_WINDOW = 30" in joined
    assert "SEARCH_START = 100" in joined
    assert "STD_DDOF = 0" in joined
    assert "THRESHOLD_OPERATOR = '>'" in joined
    assert "current_residual <= lower" in joined
    assert "NEAR_END_MIN_DEGRADED_FILES = 30" in joined
    assert "FROZEN_IRRMS_THRESHOLD = 1.2637933790683746" in joined
    assert "irrms_threshold_candidates.csv" in joined
    assert "irrms_fpt_sensitivity.csv" in joined
    assert "irrms_threshold_auto_screen.csv" in joined
    assert "irrms_fpt_sensitivity_config.json" in joined
    assert "fpt_reference_frozen_summary.csv" in joined
    assert "fpt_reference_frozen_config.json" in joined
    assert "_fpt_labels.npz" in joined
    assert "task1_fpt_visualizations" in joined
    assert "task1_fpt_overview.png" in joined
    assert "task1_fpt_visualization_summary.csv" in joined
    assert "task1_fpt_visualizations.zip" in joined
    assert "Times New Roman" in joined
    assert "FPT_LINE_COLOR = '#D62728'" in joined
    assert "FPT_LINESTYLE = '-.'" in joined
    assert "IRRMS_LINEWIDTH = 2.05" in joined
    assert "Task 1 FPT reference | Q99=" in joined
    assert "threshold_frozen = True" in joined
    assert "auto_screen_pass" in joined
    assert "manual_review_required" in joined
    assert "mechanically_eligible" not in joined
    assert "SLOPE_THRESHOLD" not in joined
    assert "rolling_slope_causal" not in joined
