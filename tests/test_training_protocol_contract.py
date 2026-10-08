"""Contract tests for Task 5 — training protocol notebook.

Tests follow the exec-from-notebook pattern established in Tasks 1–4.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / '05_train_cnn_vs_cnn_lstm_kaggle.ipynb'

# ── Frozen constants ─────────────────────────────────────────────────

FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
FROZEN_FPT = {
    'bearing1_1': 1871, 'bearing1_2': 826,
    'bearing2_1': 151, 'bearing2_2': 198,
    'bearing3_1': 493, 'bearing3_2': 1597,
}
CANONICAL_BEARINGS = [
    'bearing1_1', 'bearing1_2',
    'bearing2_1', 'bearing2_2',
    'bearing3_1', 'bearing3_2',
]
SEQUENCE_LENGTH = 16
FOLD_MANIFEST_SHA256 = 'a3c9697196619b4b820400f660aacaf3d050baa7193b2e0e17a415dc230e6c38'
MODEL_CONTRACT_SHA256 = 'ffda0a69b25b1d99c9e5b911be2ca86cff489e53a215baa4343530c70f6b51bd'


# ── Helpers ──────────────────────────────────────────────────────────

def notebook_code_cells() -> str:
    """Return joined code from all code cells."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    cells = [
        ''.join(c['source']) for c in nb['cells']
        if c['cell_type'] == 'code'
    ]
    return '\n'.join(cells)


def load_pure_namespace():
    """Load pure functions from notebook cells 2+3 (helpers)."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code_cells = [
        ''.join(c['source']) for c in nb['cells']
        if c['cell_type'] == 'code'
    ]
    # Find the cell with build_target_index
    target_cell = next(c for c in code_cells if 'def build_target_index' in c)
    ns = {
        'SEQUENCE_LENGTH': 16,
        'BEARINGS': CANONICAL_BEARINGS,
        'EXPECTED_N_FILES': {
            'bearing1_1': 2803, 'bearing1_2': 871,
            'bearing2_1': 911, 'bearing2_2': 797,
            'bearing3_1': 515, 'bearing3_2': 1637,
        },
    }
    # Provide numpy for the cell
    ns['np'] = np
    ns['hashlib'] = __import__('hashlib')
    ns['json'] = json
    ns['Path'] = Path
    exec(target_cell, ns)
    return ns


def load_metrics_namespace():
    """Load metrics/FPT detection functions from notebook."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code_cells = [
        ''.join(c['source']) for c in nb['cells']
        if c['cell_type'] == 'code'
    ]
    target_cell = next(c for c in code_cells if 'def detect_fpt_from_predictions' in c)
    ns = {
        'PROBABILITY_THRESHOLD': 0.5,
        'FPT_CONSECUTIVE': 5,
        'FILE_INTERVAL_SECONDS': 10,
    }
    ns['np'] = np
    # Provide sklearn functions
    from sklearn.metrics import (
        precision_recall_fscore_support,
        average_precision_score,
        roc_auc_score,
        balanced_accuracy_score,
    )
    ns['precision_recall_fscore_support'] = precision_recall_fscore_support
    ns['average_precision_score'] = average_precision_score
    ns['roc_auc_score'] = roc_auc_score
    ns['balanced_accuracy_score'] = balanced_accuracy_score
    exec(target_cell, ns)
    return ns


def load_training_utils_namespace():
    """Load training utility functions (pos_weight, sampler, early stopping)."""
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    code_cells = [
        ''.join(c['source']) for c in nb['cells']
        if c['cell_type'] == 'code'
    ]
    target_cell = next(c for c in code_cells if 'def compute_pos_weight' in c)
    ns = {'np': np}
    # Provide torch stubs for WeightedRandomSampler
    try:
        from torch.utils.data import WeightedRandomSampler
        ns['WeightedRandomSampler'] = WeightedRandomSampler
    except ImportError:
        pass
    exec(target_cell, ns)
    return ns


# ── Pure function tests ──────────────────────────────────────────────

class TestBuildTargetIndex:
    def test_targets_start_at_sequence_minus_one(self):
        ns = load_pure_namespace()
        targets = ns['build_target_index'](
            ['bearing3_1'], ns['EXPECTED_N_FILES'], SEQUENCE_LENGTH,
        )
        first_target = targets[0][1]
        assert first_target == SEQUENCE_LENGTH - 1  # 15

    def test_targets_end_at_last_file(self):
        ns = load_pure_namespace()
        targets = ns['build_target_index'](
            ['bearing3_1'], ns['EXPECTED_N_FILES'], SEQUENCE_LENGTH,
        )
        last_target = targets[-1][1]
        assert last_target == ns['EXPECTED_N_FILES']['bearing3_1'] - 1  # 514

    def test_cnn_and_lstm_get_identical_targets(self):
        ns = load_pure_namespace()
        cnn_targets = ns['build_target_index'](
            CANONICAL_BEARINGS[:4], ns['EXPECTED_N_FILES'], SEQUENCE_LENGTH,
        )
        lstm_targets = ns['build_target_index'](
            CANONICAL_BEARINGS[:4], ns['EXPECTED_N_FILES'], SEQUENCE_LENGTH,
        )
        assert cnn_targets == lstm_targets

    def test_rejects_zero_sequence_length(self):
        ns = load_pure_namespace()
        with pytest.raises(ValueError, match='positive'):
            ns['build_target_index'](['bearing1_1'], ns['EXPECTED_N_FILES'], 0)


class TestApplyChannelZscore:
    def test_zscore_produces_float32(self):
        ns = load_pure_namespace()
        x = np.random.randn(2, 128, 128).astype(np.float16)
        result = ns['apply_channel_zscore'](x, [0.0, 0.0], [1.0, 1.0])
        assert result.dtype == np.float32

    def test_zscore_normalizes_correctly(self):
        ns = load_pure_namespace()
        x = np.ones((2, 4, 4), dtype=np.float32) * 10.0
        result = ns['apply_channel_zscore'](x, [10.0, 10.0], [2.0, 2.0])
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_zscore_rejects_zero_std(self):
        ns = load_pure_namespace()
        x = np.ones((2, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match='std'):
            ns['apply_channel_zscore'](x, [0.0, 0.0], [0.0, 1.0])


class TestDetectFpt:
    def test_five_consecutive_returns_first_index(self):
        ns = load_metrics_namespace()
        probs = np.array([0.1, 0.1, 0.6, 0.7, 0.8, 0.9, 0.95, 0.1])
        result = ns['detect_fpt_from_predictions'](probs, threshold=0.5, consecutive=5)
        assert result == 2

    def test_four_consecutive_returns_none(self):
        ns = load_metrics_namespace()
        probs = np.array([0.1, 0.6, 0.7, 0.8, 0.9, 0.1, 0.1])
        result = ns['detect_fpt_from_predictions'](probs, threshold=0.5, consecutive=5)
        assert result is None

    def test_gap_resets_counter(self):
        ns = load_metrics_namespace()
        probs = np.array([0.6, 0.7, 0.8, 0.1, 0.6, 0.7, 0.8, 0.9, 0.95])
        result = ns['detect_fpt_from_predictions'](probs, threshold=0.5, consecutive=5)
        assert result == 4

    def test_all_below_threshold_returns_none(self):
        ns = load_metrics_namespace()
        probs = np.zeros(100)
        result = ns['detect_fpt_from_predictions'](probs, threshold=0.5, consecutive=5)
        assert result is None


class TestComputeSampleMetrics:
    def test_perfect_classification(self):
        ns = load_metrics_namespace()
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        m = ns['compute_sample_metrics'](y_true, y_prob)
        assert m['precision'] == pytest.approx(1.0)
        assert m['recall'] == pytest.approx(1.0)
        assert m['f1'] == pytest.approx(1.0)
        assert m['auc_pr'] is not None
        assert m['auc_pr'] > 0.99

    def test_handles_single_class(self):
        ns = load_metrics_namespace()
        y_true = np.ones(10, dtype=float)
        y_prob = np.ones(10) * 0.9
        m = ns['compute_sample_metrics'](y_true, y_prob)
        assert m['auc_pr'] is None
        assert m['roc_auc'] is None


class TestPosWeight:
    def test_pos_weight_from_training_only(self):
        ns = load_training_utils_namespace()
        # Simulate: 3 bearings, each with some files
        cache = {
            'b1': {'y_state': np.array([0, 0, 0, 1, 1])},
            'b2': {'y_state': np.array([0, 0, 1, 1, 1])},
        }
        targets = [('b1', 0), ('b1', 1), ('b1', 2), ('b1', 3), ('b1', 4),
                   ('b2', 0), ('b2', 1), ('b2', 2), ('b2', 3), ('b2', 4)]
        pw = ns['compute_pos_weight'](targets, cache)
        # 5 neg, 5 pos → pw = 1.0
        assert pw == pytest.approx(1.0)

    def test_pos_weight_rejects_no_positives(self):
        ns = load_training_utils_namespace()
        cache = {'b1': {'y_state': np.array([0, 0, 0])}}
        targets = [('b1', 0), ('b1', 1), ('b1', 2)]
        with pytest.raises(ValueError, match='positive'):
            ns['compute_pos_weight'](targets, cache)


class TestEarlyStopping:
    def test_tracks_best_and_stops(self):
        ns = load_training_utils_namespace()
        es = ns['EarlyStoppingTracker'](patience=3, mode='max')
        assert es.step(0.5, 0) is True   # improved
        assert es.step(0.6, 1) is True   # improved
        assert es.step(0.55, 2) is False  # not improved
        assert es.should_stop() is False
        assert es.step(0.55, 3) is False
        assert es.should_stop() is False
        assert es.step(0.55, 4) is False
        assert es.should_stop() is True   # 3 non-improvements
        assert es.best_epoch == 1
        assert es.best_value == pytest.approx(0.6)


# ── Notebook contract tests ──────────────────────────────────────────

class TestNotebookTrainingContract:
    """Verify required and forbidden tokens in the generated notebook."""

    def test_all_cells_parse(self):
        joined = notebook_code_cells()
        ast.parse(joined)

    def test_required_training_tokens(self):
        joined = notebook_code_cells()
        required = [
            # Frozen constants
            'FROZEN_IRRMS_THRESHOLD',
            'FROZEN_FPT',
            'LABEL_VERSION',
            'CACHE_VERSION',
            'EXPECTED_FOLD_MANIFEST_SHA256',
            'EXPECTED_MODEL_CONTRACT_SHA256',
            # Model architecture
            'CwtEncoder',
            'CwtCnnClassifier',
            'CwtCnnLstmClassifier',
            'GroupNorm',
            'build_paired_models',
            'state_dict_sha256',
            # Dataset classes
            'CnnSampleDataset',
            'CnnLstmSequenceDataset',
            'SEQUENCE_LENGTH',
            # Training
            'BCEWithLogitsLoss',
            'AdamW',
            'EarlyStoppingTracker',
            'compute_pos_weight',
            'build_balanced_sampler',
            'WeightedRandomSampler',
            'discover_data_dir',
            'FPT_TASK5_DATA_DIR',
            # Evaluation
            'compute_sample_metrics',
            'detect_fpt_from_predictions',
            'compute_event_metrics',
            'evaluate_outer_test',
            'y_logit',
            # Provenance
            'sha256_file',
            'validate_inputs',
            'atomic_write_json',
            'checkpoint_provenance',
            'smoke_gate_report',
            'training_protocol_config',
            'training_run_manifest',
            # Anti-leakage markers
            'training labels ONLY',
            'INNER-VAL only',
            'all decisions frozen',
        ]
        for token in required:
            assert token in joined, f'Missing required token: {token}'

    def test_forbidden_tokens(self):
        joined = notebook_code_cells()
        forbidden = [
            'Test_set',
            'Full_Test_Set',
            'BatchNorm',
            'StandardScaler',
            'MinMaxScaler',
            'fit_transform',
            'dataset-slug',
            'bearing1_3',
            'bearing1_4',
            'bearing2_3',
            'bearing3_3',
        ]
        for token in forbidden:
            assert token not in joined, f'Forbidden token found: {token}'

    def test_no_batchnorm_in_model(self):
        joined = notebook_code_cells()
        assert 'nn.BatchNorm' not in joined
        assert 'BatchNorm1d' not in joined
        assert 'BatchNorm2d' not in joined

    def test_uses_groupnorm(self):
        joined = notebook_code_cells()
        assert 'GroupNorm(8' in joined

    def test_uses_only_canonical_bearings(self):
        joined = notebook_code_cells()
        for b in CANONICAL_BEARINGS:
            assert b in joined, f'Missing bearing: {b}'
        assert 'bearing1_3' not in joined

    def test_early_stopping_on_inner_val_only(self):
        joined = notebook_code_cells()
        assert 'val_auc_pr' in joined
        # The stopper receives val metric, never test metric
        assert 'stopper.step(val_auc_pr' in joined or 'stopper.step(val_auc_pr,' in joined

    def test_test_loader_created_after_training(self):
        """Outer-test loader must be created after training loop ends."""
        joined = notebook_code_cells()
        # test_loader creation must come after the training loop
        assert 'test_loader is NOT created here' in joined
        train_loop_pos = joined.index('stopper.should_stop()')
        test_loader_pos = joined.index('test_loader = DataLoader(test_ds')
        assert train_loop_pos < test_loader_pos, \
            'test_loader should be created after training loop'

    def test_pos_weight_from_training_only_marker(self):
        joined = notebook_code_cells()
        assert 'pos_weight from training labels ONLY' in joined

    def test_smoke_gate_before_training(self):
        joined = notebook_code_cells()
        smoke_pos = joined.index('run_smoke_gate')
        train_pos = joined.index('train_fold_model')
        assert smoke_pos < train_pos, 'Smoke gate must run before training'

    def test_frozen_fpt_values_match(self):
        joined = notebook_code_cells()
        for bearing, fpt in FROZEN_FPT.items():
            pattern = f"'{bearing}': {fpt}"
            assert pattern in joined, f'Missing frozen FPT: {bearing}={fpt}'

    def test_frozen_hash_values_present(self):
        joined = notebook_code_cells()
        assert FOLD_MANIFEST_SHA256 in joined
        assert MODEL_CONTRACT_SHA256 in joined

    def test_official_data_policy_closed(self):
        joined = notebook_code_cells()
        assert 'closed; learning bearings only' in joined

    def test_protocol_config_saved_before_training(self):
        joined = notebook_code_cells()
        config_pos = joined.index('training_protocol_config.json')
        train_pos = joined.index('result = train_fold_model(')
        assert config_pos < train_pos, \
            'Protocol config must be saved before training starts'

    def test_checkpoint_preserves_provenance(self):
        joined = notebook_code_cells()
        required = [
            "'provenance': checkpoint_provenance",
            "'fit_bearings': list(fold['fit_bearings'])",
            "'inner_val_bearing': fold['inner_val_bearing']",
            "'outer_test_bearing': outer_bearing",
            "'fold_manifest_sha256': input_hashes['fold_manifest_sha256']",
            "'model_contract_sha256': input_hashes['model_contract_sha256']",
            "'training_seed': train_seed",
        ]
        for token in required:
            assert token in joined, f'Missing checkpoint provenance token: {token}'

    def test_predictions_save_logits_and_probabilities(self):
        joined = notebook_code_cells()
        assert 'test_y_logit = np.array' in joined
        assert 'y_logit=test_y_logit' in joined
        assert 'y_prob=test_y_prob' in joined

    def test_cudnn_deterministic(self):
        joined = notebook_code_cells()
        assert 'cudnn.deterministic = True' in joined
        assert 'cudnn.benchmark = False' in joined
