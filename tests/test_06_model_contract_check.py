import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "06_model_contract_check.ipynb"


def notebook_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def load_pure_namespace():
    namespace = {"np": np}
    for source in notebook_code_cells():
        if "def build_target_index" in source or "def apply_channel_zscore" in source:
            exec(source, namespace)
    return namespace


def test_target_index_is_causal_chronological_and_aligned():
    build = load_pure_namespace()["build_target_index"]
    index = build(["bearing_a", "bearing_b"], {"bearing_a": 18, "bearing_b": 17}, 16)
    assert index == [
        ("bearing_a", 15), ("bearing_a", 16), ("bearing_a", 17),
        ("bearing_b", 15), ("bearing_b", 16),
    ]
    for bearing, target in index:
        indices = list(range(target - 15, target + 1))
        assert len(indices) == 16
        assert indices[-1] == target
        assert indices[0] >= 0


def test_channel_zscore_preserves_shapes_and_returns_float32():
    transform = load_pure_namespace()["apply_channel_zscore"]
    x = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    y = transform(x, mean=[1.0, 2.0], std=[2.0, 4.0])
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.isfinite(y).all()


def test_notebook_records_dataset_and_shared_model_contract():
    cells = notebook_code_cells()
    for source in cells:
        ast.parse(source)
    joined = "\n".join(cells)
    required = [
        "DATA_DIR = Path('/kaggle/input/dataset-slug/')",
        "SEQUENCE_LENGTH = 16",
        "class CnnSampleDataset(Dataset):",
        "class CnnLstmSequenceDataset(Dataset):",
        "class CwtEncoder(nn.Module):",
        "nn.GroupNorm(8, out_channels)",
        "nn.AdaptiveAvgPool2d((1, 1))",
        "class CwtCnnClassifier(nn.Module):",
        "class CwtCnnLstmClassifier(nn.Module):",
        "nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)",
        "def build_paired_models(",
        "state_dict_sha256(cnn.encoder.state_dict())",
        "state_dict_sha256(cnn_lstm.encoder.state_dict())",
        "assert cnn_logits.shape == (BATCH_SIZE,)",
        "assert lstm_logits.shape == (BATCH_SIZE,)",
        "model_contract.json",
        "atomic_write_json",
    ]
    for token in required:
        assert token in joined
    forbidden = [
        "BatchNorm", "Sigmoid", "BCELoss", "MSELoss", "optimizer",
        "EarlyStopping", "Test_set", "Full_Test_Set",
    ]
    for token in forbidden:
        assert token not in joined


def test_notebook_records_alignment_and_provenance_audits():
    joined = "\n".join(notebook_code_cells())
    required = [
        "cnn_targets == lstm_targets",
        "indices=list(range(target_index - SEQUENCE_LENGTH + 1, target_index + 1))",
        "encoder_hashes_equal",
        "encoder_parameters_are_distinct",
        "no_batchnorm",
        "target_counts_by_fold",
        "fold_manifest_sha256",
        "scaler_sha256",
        "cache_sha256",
        "official_data_policy",
    ]
    for token in required:
        assert token in joined
