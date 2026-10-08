import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "03_cwt_image_cache.ipynb"


def notebook_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def load_validation_namespace():
    source = next(
        source for source in notebook_code_cells() if "def validate_label_alignment" in source
    )
    namespace = {
        "np": np,
        "FROZEN_IRRMS_THRESHOLD": 1.2637933790683746,
        "LABEL_VERSION": "causal-irrms-fpt-reference-v3-q99",
    }
    exec(source, namespace)
    return namespace


def valid_label_arguments():
    y_state = np.r_[np.zeros(4, dtype=np.uint8), np.ones(3, dtype=np.uint8)]
    return dict(
        y_state=y_state,
        file_index=np.arange(len(y_state), dtype=np.int32),
        bearing_id=np.full(len(y_state), "bearing1_1"),
        fpt_index=4,
        frozen_threshold=1.2637933790683746,
        label_version="causal-irrms-fpt-reference-v3-q99",
        expected_bearing="bearing1_1",
    )


def test_label_alignment_accepts_frozen_task1_contract():
    fn = load_validation_namespace()["validate_label_alignment"]
    result = fn(**valid_label_arguments())
    assert result == {"n_files": 7, "fpt_index": 4}


def test_label_alignment_rejects_non_contiguous_file_index():
    fn = load_validation_namespace()["validate_label_alignment"]
    arguments = valid_label_arguments()
    arguments["file_index"] = np.array([0, 1, 2, 3, 5, 6, 7])
    with pytest.raises(ValueError, match="file_index"):
        fn(**arguments)


def test_label_alignment_rejects_threshold_drift():
    fn = load_validation_namespace()["validate_label_alignment"]
    arguments = valid_label_arguments()
    arguments["frozen_threshold"] = 1.1
    with pytest.raises(ValueError, match="threshold"):
        fn(**arguments)


def test_label_alignment_rejects_label_transition_mismatch():
    fn = load_validation_namespace()["validate_label_alignment"]
    arguments = valid_label_arguments()
    arguments["y_state"] = np.r_[np.zeros(5, dtype=np.uint8), np.ones(2, dtype=np.uint8)]
    with pytest.raises(ValueError, match="FPT"):
        fn(**arguments)


def test_notebook_records_unscaled_cwt_cache_contract():
    code_cells = notebook_code_cells()
    for source in code_cells:
        ast.parse(source)
    joined = "\n".join(code_cells)

    required = [
        "CACHE_KIND = 'unscaled_log_power'",
        "FROZEN_IRRMS_THRESHOLD = 1.2637933790683746",
        "CWT_SCALES = np.geomspace(1, 512, 128)",
        "CACHE_DTYPE = np.float16",
        "CWT_FREQUENCIES_HZ = pywt.scale2frequency(",
        "NYQUIST_FREQUENCY_HZ = SAMPLING_FREQ / 2.0",
        "pywt.cwt(",
        "CWT_WAVELET,",
        "np.log2(np.abs(coef) ** 2 + LOG_POWER_EPS)",
        "resize(",
        "preserve_range=True",
        "anti_aliasing=True",
        "dtype=CACHE_DTYPE",
        "cwt_log_power=cwt_cache",
        "y_state=labels['y_state']",
        "file_index=labels['file_index']",
        "bearing_id=labels['bearing_id']",
        "cwt_wavelet=np.array(CWT_WAVELET)",
        "cwt_frequencies_hz=CWT_FREQUENCIES_HZ.astype(np.float32)",
        "nyquist_frequency_hz=np.float64(NYQUIST_FREQUENCY_HZ)",
        "image_shape=np.asarray(IMAGE_SHAPE, dtype=np.int32)",
        "storage_dtype=np.array(str(np.dtype(CACHE_DTYPE)))",
        "temp_path.replace(cache_path)",
        "cwt_cache_manifest.json",
        "sha256_file",
    ]
    for token in required:
        assert token in joined

    forbidden = ["StandardScaler", "MinMaxScaler", "fit_transform", "Test_set", "Full_Test_Set"]
    for token in forbidden:
        assert token not in joined

    assert "CWT_SCALES,\n        'morl'" not in joined


def test_notebook_records_visualization_and_cache_hardening_contract():
    joined = "\n".join(notebook_code_cells())
    required = [
        "def validate_existing_cache(",
        "validate_label_alignment(",
        "raw_sha256",
        "label_sha256",
        "expected_shape = (n_files, 2, *IMAGE_SHAPE)",
        "def finalize_streaming_stats(",
        "VISUALIZATION_DIR = OUT_DIR / 'visualizations'",
        "selected_indices = [0, fpt_index, n_files - 1]",
        "fig, axes = plt.subplots(2, 3",
        "np.percentile(examples, [1.0, 99.0])",
        "visualization_sha256",
        "display(Image(filename=str(figure_path)))",
        "'scaling_policy':",
        "'official_test_policy':",
    ]
    for token in required:
        assert token in joined


def test_streaming_statistics_match_direct_float64_calculation():
    source = next(
        source for source in notebook_code_cells() if "def finalize_streaming_stats" in source
    )
    namespace = {
        "np": np,
        "Path": Path,
        "IMAGE_SHAPE": (128, 128),
        "CACHE_DTYPE": np.float16,
        "CACHE_KIND": "unscaled_log_power",
        "CACHE_VERSION": "cwt-unscaled-log-power-v2",
        "CWT_WAVELET": "morl",
        "CWT_SCALES": np.geomspace(1, 512, 128),
        "CWT_FREQUENCIES_HZ": np.geomspace(20800, 40.625, 128),
        "NYQUIST_FREQUENCY_HZ": 12800.0,
        "sha256_file": lambda path: "unused",
        "raw_path": lambda bearing: Path("unused-raw"),
        "label_path": lambda bearing: Path("unused-label"),
        "validate_label_alignment": lambda **kwargs: None,
    }
    exec(source, namespace)
    values = np.array([-2.0, -0.5, 1.0, 4.0], dtype=np.float16).astype(np.float64)
    result = namespace["finalize_streaming_stats"](
        values.size,
        float(values.sum()),
        float(np.square(values).sum()),
        float(values.min()),
        float(values.max()),
    )
    assert result["cwt_min"] == float(values.min())
    assert result["cwt_max"] == float(values.max())
    assert result["cwt_mean"] == pytest.approx(float(values.mean()), abs=1e-12)
    assert result["cwt_std"] == pytest.approx(float(values.std()), abs=1e-12)


def test_existing_cache_validation_compares_current_frozen_labels():
    joined = "\n".join(notebook_code_cells())
    required = [
        "current_labels = load_frozen_labels(bearing)",
        "np.array_equal(y_state, current_labels['y_state'])",
        "np.array_equal(file_index, current_labels['file_index'])",
        "fpt_index != int(current_labels['fpt_index'])",
    ]
    for token in required:
        assert token in joined


def cache_validation_fixture(tmp_path):
    source = next(
        source for source in notebook_code_cells() if "def validate_existing_cache" in source
    )
    raw_file = tmp_path / "bearing1_1.pkz"
    label_file = tmp_path / "bearing1_1_fpt_labels.npz"
    raw_file.write_bytes(b"raw-source")
    label_file.write_bytes(b"frozen-label-source")

    y_state = np.array([0, 0, 1, 1], dtype=np.uint8)
    file_index = np.arange(4, dtype=np.int32)
    bearing_id = np.full(4, "bearing1_1")
    scales = np.geomspace(1, 512, 128).astype(np.float32)
    frequencies = np.geomspace(20800, 40.625, 128).astype(np.float32)

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    current_labels = {
        "y_state": y_state,
        "file_index": file_index,
        "bearing_id": bearing_id,
        "fpt_index": np.int32(2),
    }
    namespace = {
        "np": np,
        "Path": Path,
        "IMAGE_SHAPE": (128, 128),
        "CACHE_DTYPE": np.float16,
        "CACHE_KIND": "unscaled_log_power",
        "CACHE_VERSION": "cwt-unscaled-log-power-v2",
        "CWT_WAVELET": "morl",
        "CWT_SCALES": scales.astype(np.float64),
        "CWT_FREQUENCIES_HZ": frequencies.astype(np.float64),
        "NYQUIST_FREQUENCY_HZ": 12800.0,
        "sha256_file": sha256_file,
        "raw_path": lambda bearing: raw_file,
        "label_path": lambda bearing: label_file,
        "load_frozen_labels": lambda bearing: current_labels,
        "validate_label_alignment": lambda **kwargs: None,
    }
    exec(source, namespace)

    cache_path = tmp_path / "bearing1_1_cwt_unscaled.npz"
    payload = {
        "cwt_log_power": np.zeros((4, 2, 128, 128), dtype=np.float16),
        "y_state": y_state,
        "file_index": file_index,
        "bearing_id": bearing_id,
        "fpt_index": np.int32(2),
        "frozen_irrms_threshold": np.float64(1.2637933790683746),
        "label_version": np.array("causal-irrms-fpt-reference-v3-q99"),
        "cache_kind": np.array("unscaled_log_power"),
        "cache_version": np.array("cwt-unscaled-log-power-v2"),
        "cwt_wavelet": np.array("morl"),
        "cwt_scales": scales,
        "cwt_frequencies_hz": frequencies,
        "nyquist_frequency_hz": np.float64(12800.0),
        "image_shape": np.array([128, 128], dtype=np.int32),
        "storage_dtype": np.array("float16"),
        "raw_sha256": np.array(sha256_file(raw_file)),
        "label_sha256": np.array(sha256_file(label_file)),
        "cwt_min": np.float64(0.0),
        "cwt_max": np.float64(0.0),
        "cwt_mean": np.float64(0.0),
        "cwt_std": np.float64(0.0),
    }
    return namespace["validate_existing_cache"], cache_path, payload


def test_existing_cache_validator_accepts_complete_matching_cache(tmp_path):
    validate, cache_path, payload = cache_validation_fixture(tmp_path)
    np.savez_compressed(cache_path, **payload)
    result = validate(cache_path, "bearing1_1")
    assert result["status"] == "existing_validated"
    assert result["shape"] == [4, 2, 128, 128]


def test_existing_cache_validator_rejects_internal_label_drift(tmp_path):
    validate, cache_path, payload = cache_validation_fixture(tmp_path)
    payload["y_state"] = np.array([0, 1, 1, 1], dtype=np.uint8)
    np.savez_compressed(cache_path, **payload)
    with pytest.raises(ValueError, match="y_state differs"):
        validate(cache_path, "bearing1_1")


def test_existing_cache_validator_rejects_raw_hash_drift(tmp_path):
    validate, cache_path, payload = cache_validation_fixture(tmp_path)
    payload["raw_sha256"] = np.array("stale")
    np.savez_compressed(cache_path, **payload)
    with pytest.raises(ValueError, match="raw_sha256"):
        validate(cache_path, "bearing1_1")


def test_notebook_uses_only_six_learning_bearings():
    joined = "\n".join(notebook_code_cells())
    for bearing in (
        "bearing1_1",
        "bearing1_2",
        "bearing2_1",
        "bearing2_2",
        "bearing3_1",
        "bearing3_2",
    ):
        assert f"'{bearing}'" in joined
    assert "bearing1_3" not in joined
