import ast
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "03_prepare_fold_scalers_kaggle.ipynb"
BEARINGS = [
    "bearing1_1", "bearing1_2", "bearing2_1",
    "bearing2_2", "bearing3_1", "bearing3_2",
]


def notebook_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def load_pure_namespace():
    cells = notebook_code_cells()
    sources = [
        source
        for source in cells
        if "def build_fold_schedule" in source
        or "def empty_channel_moments" in source
    ]
    namespace = {"np": np}
    for source in sources:
        exec(source, namespace)
    return namespace


def test_fold_schedule_is_disjoint_and_complete():
    folds = load_pure_namespace()["build_fold_schedule"](BEARINGS)
    assert len(folds) == 6
    assert [fold["outer_test_bearing"] for fold in folds] == BEARINGS
    assert [fold["inner_val_bearing"] for fold in folds] == BEARINGS[1:] + BEARINGS[:1]
    for fold in folds:
        assert len(fold["fit_bearings"]) == 4
        assert set(fold["fit_bearings"]).isdisjoint(
            {fold["inner_val_bearing"], fold["outer_test_bearing"]}
        )


def test_parallel_welford_matches_numpy_and_zscore_is_float32():
    ns = load_pure_namespace()
    values = np.arange(7 * 2 * 3 * 5, dtype=np.float16).reshape(7, 2, 3, 5)
    state = ns["empty_channel_moments"](2)
    for chunk in (values[:2], values[2:6], values[6:]):
        state = ns["merge_channel_moments"](
            state, ns["batch_channel_moments"](chunk)
        )
    scaler = ns["finalize_channel_scaler"](state, epsilon=1e-8)
    expected_mean = values.astype(np.float64).mean(axis=(0, 2, 3))
    expected_std = values.astype(np.float64).std(axis=(0, 2, 3), ddof=0)
    assert np.allclose(scaler["mean"], expected_mean, rtol=0, atol=1e-12)
    assert np.allclose(scaler["std"], expected_std, rtol=0, atol=1e-12)
    transformed = ns["apply_channel_zscore"](
        values, scaler["mean"], scaler["std"]
    )
    assert transformed.dtype == np.float32
    assert transformed.shape == values.shape
    assert np.allclose(transformed.mean(axis=(0, 2, 3)), 0.0, atol=2e-6)
    assert np.allclose(transformed.std(axis=(0, 2, 3)), 1.0, atol=2e-6)


def test_scaler_rejects_zero_variance():
    ns = load_pure_namespace()
    state = ns["batch_channel_moments"](
        np.ones((2, 2, 3, 3), dtype=np.float16)
    )
    with pytest.raises(ValueError, match="std"):
        ns["finalize_channel_scaler"](state, epsilon=1e-8)


def test_notebook_records_leakage_safe_contract():
    cells = notebook_code_cells()
    for source in cells:
        ast.parse(source)
    joined = "\n".join(cells)
    required = [
        "CACHE_DIR = Path('/kaggle/input/datasets/longvu274/dataset-slug/')",
        "CACHE_VERSION = 'cwt-unscaled-log-power-v2'",
        "SCALER_KIND = 'per_channel_zscore'",
        "FROZEN_IRRMS_THRESHOLD = 1.2637933790683746",
        "def validate_task2_manifest(",
        "def fit_fold_scaler(",
        "def probe_nonfit_cache(",
        "atomic_write_json(scaler_path, scaler)",
        "fold_scaler_audit.csv",
        "fold_manifest.json",
        "fit_cache_sha256",
        "outer_not_in_fit",
        "inner_not_in_fit",
        "postfit_probe_finite",
    ]
    for token in required:
        assert token in joined
    assert joined.index("atomic_write_json(scaler_path, scaler)") < joined.index(
        "probe_results = probe_nonfit_cache("
    )
    forbidden = [
        "torch", "nn.Module", "optimizer", "EarlyStopping",
        "Test_set", "Full_Test_Set", "StandardScaler", "MinMaxScaler",
    ]
    for token in forbidden:
        assert token not in joined


def test_notebook_uses_only_canonical_learning_bearings():
    joined = "\n".join(notebook_code_cells())
    for bearing in BEARINGS:
        assert f"'{bearing}'" in joined
    assert "bearing1_3" not in joined
