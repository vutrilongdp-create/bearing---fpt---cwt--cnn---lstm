"""Generate 08_threshold_calibration.ipynb deterministically."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(True),
    }


cells = [
    # ── Cell 1: Header ───────────────────────────────────────────────
    md("""\
# Task 6 — Inner-Validation Threshold Calibration for FPT Detection

Secondary analysis for CNN vs CNN-LSTM FPT detection. Training matches Task 5,
but the event decision threshold is selected from the inner-validation bearing
only and then applied once to the outer-test bearing.

**Protocol**: smoke gate → 1-fold pilot → 6 folds × 2 models.
Keep `PILOT_ONLY = True` for the first Kaggle smoke run; switch to `False`
only after reviewing pilot threshold-calibration artifacts."""),

    # ── Cell 2: Imports + frozen configuration ────────────────────────
    code("""\
# 1) Imports and frozen training configuration
import copy
import csv
import hashlib
import json
import os
import random
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# ── Execution mode ──
PILOT_ONLY = True
PILOT_FOLD = 4

# ── Paths ──
KAGGLE_INPUT_ROOT = Path('/kaggle/input')
DATA_DIR_ENV_VAR = 'FPT_TASK6_DATA_DIR'
OUT_DIR = Path('/kaggle/working/fpt_threshold_calibration/')
ZIP_PATH = Path('/kaggle/working/fpt_threshold_calibration_artifacts.zip')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Frozen data constants ──
BEARINGS = [
    'bearing1_1', 'bearing1_2', 'bearing2_1',
    'bearing2_2', 'bearing3_1', 'bearing3_2',
]
EXPECTED_N_FILES = {
    'bearing1_1': 2803, 'bearing1_2': 871,
    'bearing2_1': 911, 'bearing2_2': 797,
    'bearing3_1': 515, 'bearing3_2': 1637,
}
FROZEN_FPT = {
    'bearing1_1': 1871, 'bearing1_2': 826,
    'bearing2_1': 151, 'bearing2_2': 198,
    'bearing3_1': 493, 'bearing3_2': 1597,
}
FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
LABEL_VERSION = 'causal-irrms-fpt-reference-v3-q99'
CACHE_VERSION = 'cwt-unscaled-log-power-v2'

# ── Frozen model constants ──
SEQUENCE_LENGTH = 16
ENCODER_DIM = 128
LSTM_HIDDEN_SIZE = 128
MODEL_SEED = 274

# ── Training hyperparameters (same for both models) ──
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
PROBABILITY_THRESHOLD = 0.5
FIXED_THRESHOLD_REFERENCE = 0.5
FPT_CONSECUTIVE = 5
FILE_INTERVAL_SECONDS = 10
THRESHOLD_GRID = [round(float(x), 2) for x in np.arange(0.10, 0.951, 0.05)]
CALIBRATION_SELECTION_RULE = [
    'prefer_not_missed',
    'prefer_no_pre_fpt_alarm_run',
    'minimize_abs_fpt_error_files',
    'minimize_pre_fpt_positive_sample_count',
    'prefer_higher_threshold',
]

# ── Frozen provenance hashes ──
EXPECTED_FOLD_MANIFEST_SHA256 = 'a3c9697196619b4b820400f660aacaf3d050baa7193b2e0e17a415dc230e6c38'
EXPECTED_MODEL_CONTRACT_SHA256 = 'ffda0a69b25b1d99c9e5b911be2ca86cff489e53a215baa4343530c70f6b51bd'

# ── Kaggle input discovery ──
def required_input_files() -> list[str]:
    return (
        ['fold_manifest.json', 'model_contract.json']
        + [f'fold_{idx}_scaler.json' for idx in range(6)]
        + [f'{bearing}_cwt_unscaled.npz' for bearing in BEARINGS]
    )


def discover_data_dir() -> Path:
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        candidate = Path(env_value)
        missing = [name for name in required_input_files() if not (candidate / name).exists()]
        if missing:
            raise FileNotFoundError(
                f'{DATA_DIR_ENV_VAR}={candidate} is missing required files: {missing[:5]}'
            )
        return candidate

    candidates = []
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(p for p in KAGGLE_INPUT_ROOT.iterdir() if p.is_dir()):
            if all((candidate / name).exists() for name in required_input_files()):
                candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            'Multiple Task6 input datasets found. Set '
            f'{DATA_DIR_ENV_VAR} to one of: {[str(p) for p in candidates]}'
        )
    raise FileNotFoundError(
        'Could not auto-discover Task6 input dataset under /kaggle/input. '
        f'Attach the dataset containing CWT cache + fold scalers + model_contract, '
        f'or set {DATA_DIR_ENV_VAR}.'
    )


DATA_DIR = discover_data_dir()

# ── Device + reproducibility ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Device:', DEVICE)
print('PILOT_ONLY:', PILOT_ONLY, '  PILOT_FOLD:', PILOT_FOLD)
print('DATA_DIR:', DATA_DIR)
print('OUT_DIR:', OUT_DIR)
print('ZIP_PATH:', ZIP_PATH)"""),

    # ── Cell 3: Pure data helpers ────────────────────────────────────
    code("""\
# 2) Pure data helpers

def build_target_index(
    bearings: list[str],
    n_files_by_bearing: dict[str, int],
    sequence_length: int = 16,
) -> list[tuple[str, int]]:
    if sequence_length <= 0:
        raise ValueError('sequence_length must be positive')
    result = []
    for bearing in bearings:
        n_files = int(n_files_by_bearing[bearing])
        if n_files < sequence_length:
            raise ValueError(f'{bearing}: fewer files than sequence length')
        result.extend(
            (bearing, target_index)
            for target_index in range(sequence_length - 1, n_files)
        )
    return result


def apply_channel_zscore(values: np.ndarray, mean, std) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[-3] != 2:
        raise ValueError(f'Expected [...,2,H,W], got {values.shape}')
    shape = [1] * values.ndim
    shape[-3] = 2
    mean = np.asarray(mean, dtype=np.float32).reshape(shape)
    std = np.asarray(std, dtype=np.float32).reshape(shape)
    if np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError('Invalid scaler std')
    result = (values - mean) / std
    if not np.isfinite(result).all():
        raise ValueError('Scaled values are non-finite')
    return result.astype(np.float32, copy=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    temp_path.replace(path)


def create_artifact_zip(source_dir: Path, zip_path: Path) -> dict:
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if zip_path.exists():
        zip_path.unlink()

    members = []
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob('*')):
            if not path.is_file():
                continue
            arcname = path.relative_to(source_dir.parent).as_posix()
            archive.write(path, arcname=arcname)
            members.append(arcname)

    return {
        'zip_path': str(zip_path),
        'source_dir': str(source_dir),
        'n_files': len(members),
        'members': members,
    }


def load_npz_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            'cwt_log_power': data['cwt_log_power'].copy(),
            'y_state': data['y_state'].copy(),
            'file_index': data['file_index'].copy(),
            'fpt_index': int(np.asarray(data['fpt_index']).item()),
        }


def validate_model_contract(data_dir: Path) -> dict:
    model_contract_path = data_dir / 'model_contract.json'
    if not model_contract_path.exists():
        raise FileNotFoundError(model_contract_path)
    actual_hash = sha256_file(model_contract_path)
    if EXPECTED_MODEL_CONTRACT_SHA256 is not None:
        if actual_hash != EXPECTED_MODEL_CONTRACT_SHA256:
            raise ValueError(
                'model_contract.json SHA-256 mismatch. '
                f'Expected {EXPECTED_MODEL_CONTRACT_SHA256}, got {actual_hash}'
            )
    return {
        'path': str(model_contract_path),
        'sha256': actual_hash,
    }


def validate_inputs(data_dir: Path):
    fold_manifest_path = data_dir / 'fold_manifest.json'
    fold_manifest_sha256 = sha256_file(fold_manifest_path)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding='utf-8'))
    if len(fold_manifest.get('folds', [])) != 6:
        raise ValueError('Expected six frozen folds')
    scalers = {}
    scaler_sha256 = {}
    expected_cache_hash = {}
    cache_sha256 = {}
    for fold in fold_manifest['folds']:
        index = int(fold['fold_index'])
        scaler_path = data_dir / f'fold_{index}_scaler.json'
        actual_scaler_hash = sha256_file(scaler_path)
        if actual_scaler_hash != fold['scaler_sha256']:
            raise ValueError(f'Fold {index}: scaler SHA-256 mismatch')
        scaler = json.loads(scaler_path.read_text(encoding='utf-8'))
        if set(scaler['fit_bearings']) & {
            scaler['inner_val_bearing'], scaler['outer_test_bearing']
        }:
            raise ValueError(f'Fold {index}: scaler leakage')
        scalers[index] = scaler
        scaler_sha256[index] = actual_scaler_hash
        expected_cache_hash.update(scaler['fit_cache_sha256'])
    for bearing in BEARINGS:
        cache_path = data_dir / f'{bearing}_cwt_unscaled.npz'
        actual = sha256_file(cache_path)
        if actual != expected_cache_hash[bearing]:
            raise ValueError(f'{bearing}: cache SHA-256 mismatch')
        cache_sha256[bearing] = actual
        
    model_contract_info = validate_model_contract(data_dir)
    
    return fold_manifest, scalers, {
        'fold_manifest_sha256': fold_manifest_sha256,
        'model_contract_sha256': model_contract_info['sha256'],
        'scaler_sha256': scaler_sha256,
        'cache_sha256': cache_sha256,
    }


print('Pure data helpers OK')"""),

    # ── Cell 4: Dataset classes ───────────────────────────────────────
    code("""\
# 3) Aligned causal datasets

class CnnSampleDataset(Dataset):
    def __init__(self, cache_by_bearing, target_index, mean, std):
        self.cache_by_bearing = cache_by_bearing
        self.target_index = list(target_index)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, item):
        bearing, target_index = self.target_index[item]
        payload = self.cache_by_bearing[bearing]
        x = apply_channel_zscore(
            payload['cwt_log_power'][target_index], self.mean, self.std
        )
        y = np.float32(payload['y_state'][target_index])
        meta = dict(
            bearing_id=bearing,
            target_index=int(target_index),
            file_index=int(payload['file_index'][target_index]),
            indices=[int(target_index)],
        )
        return torch.from_numpy(x), torch.tensor(y), meta


class CnnLstmSequenceDataset(Dataset):
    def __init__(
        self, cache_by_bearing, target_index, mean, std, sequence_length=16
    ):
        self.cache_by_bearing = cache_by_bearing
        self.target_index = list(target_index)
        self.mean = mean
        self.std = std
        self.sequence_length = int(sequence_length)

    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, item):
        bearing, target_index = self.target_index[item]
        start = target_index - self.sequence_length + 1
        if start < 0:
            raise IndexError('Sequence would use a negative/future-invalid index')
        payload = self.cache_by_bearing[bearing]
        x = apply_channel_zscore(
            payload['cwt_log_power'][start:target_index + 1], self.mean, self.std
        )
        y = np.float32(payload['y_state'][target_index])
        meta = dict(
            bearing_id=bearing,
            target_index=int(target_index),
            file_index=int(payload['file_index'][target_index]),
            indices=list(range(target_index - SEQUENCE_LENGTH + 1, target_index + 1)),
        )
        return torch.from_numpy(x), torch.tensor(y), meta


print('Aligned causal datasets OK')"""),

    # ── Cell 5: Model definitions ─────────────────────────────────────
    code("""\
# 4) Shared GroupNorm encoder and classifiers

class ConvGroupBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CwtEncoder(nn.Module):
    output_dim = 128

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvGroupBlock(2, 16),
            ConvGroupBlock(16, 32),
            ConvGroupBlock(32, 64),
            ConvGroupBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class CwtCnnClassifier(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else CwtEncoder()
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


class CwtCnnLstmClassifier(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else CwtEncoder()
        self.temporal = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch * steps, channels, height, width))
        sequence = features.reshape(batch, steps, ENCODER_DIM)
        temporal_output, _ = self.temporal(sequence)
        return self.head(temporal_output[:, -1]).squeeze(-1)


# Deterministic paired initialization and hashing

def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_paired_models(seed: int = MODEL_SEED):
    set_deterministic_seed(seed)
    reference_encoder = CwtEncoder()
    reference_state = copy.deepcopy(reference_encoder.state_dict())
    cnn = CwtCnnClassifier(CwtEncoder())
    cnn_lstm = CwtCnnLstmClassifier(CwtEncoder())
    cnn.encoder.load_state_dict(reference_state)
    cnn_lstm.encoder.load_state_dict(reference_state)
    return cnn, cnn_lstm


print('Shared model definitions OK')"""),

    # ── Cell 6: Training utilities ────────────────────────────────────
    code("""\
# 5) Training utilities

def compute_pos_weight(target_index, cache_by_bearing):
    '''Compute BCEWithLogitsLoss pos_weight from training labels ONLY.
    Note: We calculate it for audit purposes, but do NOT use it in BCE loss 
    when WeightedRandomSampler is active to prevent double-weighting.'''
    n_pos = 0
    n_total = 0
    for bearing, tidx in target_index:
        label = int(cache_by_bearing[bearing]['y_state'][tidx])
        n_pos += label
        n_total += 1
    n_neg = n_total - n_pos
    if n_pos == 0:
        raise ValueError('No positive samples in training set')
    if n_neg == 0:
        raise ValueError('No negative samples in training set')
    return float(n_neg) / float(n_pos)


def build_balanced_sampler(target_index, cache_by_bearing, generator=None):
    '''Build WeightedRandomSampler by (bearing, class) groups — training only.'''
    group_counts = {}
    for bearing, tidx in target_index:
        label = int(cache_by_bearing[bearing]['y_state'][tidx])
        group = (bearing, label)
        group_counts[group] = group_counts.get(group, 0) + 1
    weights = []
    for bearing, tidx in target_index:
        label = int(cache_by_bearing[bearing]['y_state'][tidx])
        group = (bearing, label)
        weights.append(1.0 / group_counts[group])
    
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor, 
        num_samples=len(weights_tensor), 
        replacement=True,
        generator=generator,
    )


class EarlyStoppingTracker:
    '''Track inner-validation metric for early stopping — never sees outer-test.'''
    def __init__(self, patience: int, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_epoch = -1

    def step(self, value, epoch):
        improved = (value > self.best_value) if self.mode == 'max' else (value < self.best_value)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device):
    '''Run one training epoch. Returns average loss.'''
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, _meta in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_epoch(model, loader, criterion, device):
    '''Evaluate model. Returns (avg_loss, logits_array, targets_array).'''
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y, _meta in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return (
        total_loss / max(n_batches, 1),
        np.concatenate(all_logits),
        np.concatenate(all_targets),
    )


print('Training utilities OK')"""),

    # ── Cell 7: Metrics and FPT detection ─────────────────────────────
    code("""\
# 6) Metrics and FPT detection

def compute_sample_metrics(y_true, y_prob, threshold=0.5):
    '''Compute sample-level classification metrics.'''
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0,
    )
    metrics = {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'n_samples': int(len(y_true)),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
    }
    if len(np.unique(y_true)) > 1:
        metrics['auc_pr'] = float(average_precision_score(y_true, y_prob))
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['auc_pr'] = None
        metrics['roc_auc'] = None
    return metrics


def detect_fpt_from_predictions(prob, threshold=0.5, consecutive=5):
    '''Detect FPT as first index of `consecutive` predictions >= threshold.
    Returns the index in y_prob array, or None if no detection.'''
    prob = np.asarray(prob, dtype=np.float32)
    mask = prob >= threshold
    run = 0
    for idx, value in enumerate(mask):
        if value:
            run += 1
            if run >= consecutive:
                return idx - consecutive + 1
        else:
            run = 0
    return None


def compute_pre_fpt_alarm_run(file_indices, y_prob, reference_fpt, threshold=0.5, consecutive=5):
    '''Check if there is an alarm run of `consecutive` positive predictions BEFORE the reference FPT.'''
    file_indices = np.asarray(file_indices)
    y_prob = np.asarray(y_prob)
    before_mask = file_indices < reference_fpt
    before_prob = y_prob[before_mask]
    if len(before_prob) == 0:
        return False
    return detect_fpt_from_predictions(
        before_prob, threshold=threshold, consecutive=consecutive
    ) is not None


def evaluate_records(model, loader, device):
    '''Evaluate model and return records sorted by file_index.'''
    model.eval()
    records = []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            logits = model(x)
            logits_np = logits.cpu().numpy()
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                bid = meta['bearing_id'][i]
                records.append({
                    'bearing_id': bid,
                    'target_index': int(meta['target_index'][i]),
                    'file_index': int(meta['file_index'][i]),
                    'y_true': float(y[i]),
                    'y_logit': float(logits_np[i]),
                    'y_prob': float(probs[i]),
                })
    records.sort(key=lambda r: r['file_index'])
    return records


def compute_event_metrics(records, reference_fpt,
                          threshold=PROBABILITY_THRESHOLD,
                          consecutive=FPT_CONSECUTIVE,
                          interval=FILE_INTERVAL_SECONDS):
    '''Compute event-level FPT detection metrics for one outer-test bearing.'''
    y_prob = np.array([r['y_prob'] for r in records])
    file_indices = np.array([r['file_index'] for r in records])

    pred_pos = detect_fpt_from_predictions(y_prob, threshold, consecutive)
    pred_fpt = int(file_indices[pred_pos]) if pred_pos is not None else None

    before_mask = file_indices < reference_fpt
    pre_fpt_positive_sample_count = int((y_prob[before_mask] >= threshold).sum())
    pre_fpt_alarm_run = compute_pre_fpt_alarm_run(file_indices, y_prob, reference_fpt, threshold, consecutive)

    if pred_fpt is None:
        return {
            'reference_fpt': int(reference_fpt),
            'predicted_fpt': None,
            'signed_delay_files': None,
            'signed_delay_seconds': None,
            'abs_fpt_error_files': None,
            'pre_fpt_positive_sample_count': pre_fpt_positive_sample_count,
            'pre_fpt_alarm_run': bool(pre_fpt_alarm_run),
            'missed_detection': True,
        }
    signed = pred_fpt - reference_fpt
    return {
        'reference_fpt': int(reference_fpt),
        'predicted_fpt': int(pred_fpt),
        'signed_delay_files': int(signed),
        'signed_delay_seconds': int(signed) * interval,
        'abs_fpt_error_files': abs(int(signed)),
        'pre_fpt_positive_sample_count': pre_fpt_positive_sample_count,
        'pre_fpt_alarm_run': bool(pre_fpt_alarm_run),
        'missed_detection': False,
    }


def threshold_sort_key(row):
    metrics = row['inner_val_event_metrics']
    missed = bool(metrics['missed_detection'])
    pre_alarm = bool(metrics['pre_fpt_alarm_run'])
    abs_error = metrics['abs_fpt_error_files']
    if abs_error is None:
        abs_error = 10**9
    return (
        int(missed),
        int(pre_alarm),
        int(abs_error),
        int(metrics['pre_fpt_positive_sample_count']),
        -float(row['threshold']),
    )


def select_threshold_from_inner_val(records, reference_fpt, threshold_grid=THRESHOLD_GRID):
    '''Select event threshold using inner-validation records only.'''
    scan_rows = []
    for threshold in threshold_grid:
        event_metrics = compute_event_metrics(
            records,
            reference_fpt=reference_fpt,
            threshold=float(threshold),
            consecutive=FPT_CONSECUTIVE,
            interval=FILE_INTERVAL_SECONDS,
        )
        row = {
            'threshold': float(threshold),
            'inner_val_event_metrics': event_metrics,
        }
        row['sort_key'] = threshold_sort_key(row)
        scan_rows.append(row)

    selected = min(scan_rows, key=threshold_sort_key)
    warnings = []
    if all(row['inner_val_event_metrics']['missed_detection'] for row in scan_rows):
        warnings.append('all_thresholds_missed_inner_val')
    if all(row['inner_val_event_metrics']['pre_fpt_alarm_run'] for row in scan_rows):
        warnings.append('all_thresholds_have_inner_val_pre_fpt_alarm')

    return {
        'selected_threshold': float(selected['threshold']),
        'inner_val_event_metrics': selected['inner_val_event_metrics'],
        'threshold_scan': scan_rows,
        'calibration_warning': warnings,
        'selection_rule': CALIBRATION_SELECTION_RULE,
    }


def flatten_event_metrics(prefix, metrics):
    return {
        f'{prefix}_reference_fpt': metrics['reference_fpt'],
        f'{prefix}_predicted_fpt': metrics['predicted_fpt'],
        f'{prefix}_signed_delay_files': metrics['signed_delay_files'],
        f'{prefix}_signed_delay_seconds': metrics['signed_delay_seconds'],
        f'{prefix}_abs_fpt_error_files': metrics['abs_fpt_error_files'],
        f'{prefix}_pre_fpt_positive_sample_count': metrics['pre_fpt_positive_sample_count'],
        f'{prefix}_pre_fpt_alarm_run': metrics['pre_fpt_alarm_run'],
        f'{prefix}_missed_detection': metrics['missed_detection'],
    }


def write_threshold_scan_csv(scan_rows, out_path: Path):
    fieldnames = [
        'threshold',
        'inner_val_reference_fpt',
        'inner_val_predicted_fpt',
        'inner_val_signed_delay_files',
        'inner_val_signed_delay_seconds',
        'inner_val_abs_fpt_error_files',
        'inner_val_pre_fpt_positive_sample_count',
        'inner_val_pre_fpt_alarm_run',
        'inner_val_missed_detection',
        'sort_key',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scan_rows:
            flat = {'threshold': row['threshold'], 'sort_key': repr(row['sort_key'])}
            flat.update(flatten_event_metrics('inner_val', row['inner_val_event_metrics']))
            writer.writerow(flat)


def save_confusion_matrix_artifacts(y_true, y_prob, threshold, out_prefix: Path):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.save(str(out_prefix) + '_confusion_matrix.npy', cm)
    cm_json = {
        'labels': ['healthy_0', 'degraded_1'],
        'matrix': cm.tolist(),
        'threshold': float(threshold),
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1]),
    }
    with open(str(out_prefix) + '_confusion_matrix.json', 'w', encoding='utf-8') as f:
        json.dump(cm_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    return cm_json


def save_fpt_probability_plot(file_indices, y_true, y_prob, reference_fpt,
                              predicted_fpt, threshold, out_path: Path, title: str):
    file_indices = np.asarray(file_indices)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    plt.figure(figsize=(14, 4))
    plt.plot(file_indices, y_prob, linewidth=1.5, label='P(degradation)')
    plt.plot(file_indices, y_true, linewidth=1.0, alpha=0.5, label='Reference label')
    plt.axhline(threshold, linestyle='--', linewidth=1.2, label=f'Probability threshold={threshold}')
    plt.axvline(reference_fpt, linestyle='--', linewidth=1.5, label=f'Reference FPT={reference_fpt}')
    if predicted_fpt is not None:
        plt.axvline(predicted_fpt, linestyle=':', linewidth=1.8, label=f'Predicted FPT={predicted_fpt}')
    plt.xlabel('File index')
    plt.ylabel('Probability / label')
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


print('Metrics and FPT detection OK')"""),

    # ── Cell 8: Smoke gate ────────────────────────────────────────────
    code("""\
# 7) Smoke gate — must pass before any training

def check_cnn_lstm_dataset_alignment(cnn_ds, lstm_ds, n_check: int = 8):
    if len(cnn_ds) != len(lstm_ds):
        return False, 'CNN and LSTM dataset length mismatch'
    indices = np.linspace(0, len(cnn_ds) - 1, min(n_check, len(cnn_ds))).astype(int)
    for i in indices:
        _, y_cnn, meta_cnn = cnn_ds[i]
        _, y_lstm, meta_lstm = lstm_ds[i]
        if meta_cnn['bearing_id'] != meta_lstm['bearing_id']: return False, 'bearing mismatch'
        if meta_cnn['target_index'] != meta_lstm['target_index']: return False, 'target_index mismatch'
        if meta_cnn['file_index'] != meta_lstm['file_index']: return False, 'file_index mismatch'
        if float(y_cnn) != float(y_lstm): return False, 'label mismatch'
        
        t = int(meta_lstm['target_index'])
        expected_indices = list(range(t - SEQUENCE_LENGTH + 1, t + 1))
        if meta_lstm['indices'] != expected_indices: return False, 'lstm sequence indices mismatch'
        if meta_cnn['indices'] != [t]: return False, 'cnn indices mismatch'
    return True, 'aligned'


def run_smoke_gate(fold_manifest, scalers, input_hashes, cache_by_bearing):
    '''Validate runtime invariants on fold 0 before full training.'''
    report = {'passed': True, 'checks': {}}

    def check(name, condition, detail=''):
        report['checks'][name] = {'passed': bool(condition), 'detail': detail}
        if not condition:
            report['passed'] = False
            print(f'  FAIL: {name} — {detail}')
        else:
            print(f'  OK:   {name}')

    print('=== SMOKE GATE ===')

    # 1) Provenance hashes
    check('fold_manifest_sha256',
          input_hashes['fold_manifest_sha256'] == EXPECTED_FOLD_MANIFEST_SHA256,
          f"got {input_hashes['fold_manifest_sha256'][:16]}...")
    check('model_contract_sha256',
          input_hashes['model_contract_sha256'] == EXPECTED_MODEL_CONTRACT_SHA256,
          f"got {input_hashes['model_contract_sha256'][:16]}...")

    # 2) Fold 0 setup
    fold = fold_manifest['folds'][0]
    scaler = scalers[0]
    train_targets = build_target_index(fold['fit_bearings'], EXPECTED_N_FILES, SEQUENCE_LENGTH)
    val_targets = build_target_index([fold['inner_val_bearing']], EXPECTED_N_FILES, SEQUENCE_LENGTH)
    test_targets = build_target_index([fold['outer_test_bearing']], EXPECTED_N_FILES, SEQUENCE_LENGTH)

    # 3) Target alignment CNN == LSTM via Dataset items
    cnn_ds = CnnSampleDataset(cache_by_bearing, train_targets[:BATCH_SIZE*2], scaler['mean'], scaler['std'])
    lstm_ds = CnnLstmSequenceDataset(cache_by_bearing, train_targets[:BATCH_SIZE*2], scaler['mean'], scaler['std'])
    alignment_ok, alignment_msg = check_cnn_lstm_dataset_alignment(cnn_ds, lstm_ds)
    check('cnn_lstm_dataset_alignment', alignment_ok, alignment_msg)

    # 4) Sets are disjoint
    train_set = set(train_targets)
    val_set = set(val_targets)
    test_set = set(test_targets)
    check('sets_disjoint',
          len(train_set & val_set) == 0 and len(train_set & test_set) == 0
          and len(val_set & test_set) == 0)

    # 5) pos_weight calculated from training only
    pw_audit = compute_pos_weight(train_targets, cache_by_bearing)
    check('pos_weight_finite', np.isfinite(pw_audit) and pw_audit > 0, f'pos_weight_audit={pw_audit:.4f}')

    # 6) Build models and mini-batch smoke
    cnn, cnn_lstm = build_paired_models(MODEL_SEED)
    enc_hash_cnn = state_dict_sha256(cnn.encoder.state_dict())
    enc_hash_lstm = state_dict_sha256(cnn_lstm.encoder.state_dict())
    check('encoder_hashes_equal', enc_hash_cnn == enc_hash_lstm)

    cnn, cnn_lstm = cnn.to(DEVICE), cnn_lstm.to(DEVICE)
    # Smoke gate uses plain BCE (without pos_weight) as intended in training
    criterion = nn.BCEWithLogitsLoss()

    cnn_x, cnn_y, _ = next(iter(DataLoader(cnn_ds, batch_size=BATCH_SIZE)))
    lstm_x, lstm_y, _ = next(iter(DataLoader(lstm_ds, batch_size=BATCH_SIZE)))
    cnn_x, cnn_y = cnn_x.to(DEVICE), cnn_y.to(DEVICE)
    lstm_x, lstm_y = lstm_x.to(DEVICE), lstm_y.to(DEVICE)

    # 7) CNN forward/backward
    cnn_logits = cnn(cnn_x)
    cnn_loss = criterion(cnn_logits, cnn_y)
    cnn_loss.backward()
    cnn_grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in cnn.parameters() if p.requires_grad
    )
    check('cnn_shape', cnn_logits.shape == cnn_y.shape,
          f'{list(cnn_logits.shape)} vs {list(cnn_y.shape)}')
    check('cnn_loss_finite', torch.isfinite(cnn_loss).item())
    check('cnn_grad_finite', cnn_grad_ok)

    # 8) LSTM forward/backward
    lstm_logits = cnn_lstm(lstm_x)
    lstm_loss = criterion(lstm_logits, lstm_y)
    lstm_loss.backward()
    lstm_grad_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in cnn_lstm.parameters() if p.requires_grad
    )
    check('lstm_shape', lstm_logits.shape == lstm_y.shape,
          f'{list(lstm_logits.shape)} vs {list(lstm_y.shape)}')
    check('lstm_loss_finite', torch.isfinite(lstm_loss).item())
    check('lstm_grad_finite', lstm_grad_ok)

    print(f'\\n=== SMOKE GATE {"PASSED" if report["passed"] else "FAILED"} ===')
    if not report['passed']:
        raise RuntimeError('Smoke gate FAILED — cannot proceed with training')

    # Save smoke report
    report_path = OUT_DIR / 'smoke_gate_report.json'
    atomic_write_json(report_path, report)
    print(f'Saved: {report_path}')
    return report


print('Smoke gate function OK')"""),

    # ── Cell 9: Per-fold training function ────────────────────────────
    code("""\
# 8) Per-fold, per-model training function

def train_fold_model(fold_idx, model_name, fold_manifest, scalers,
                     cache_by_bearing, input_hashes):
    '''Train one model on one fold. Returns result dict.'''
    t0 = time.time()
    fold = fold_manifest['folds'][fold_idx]
    scaler = scalers[fold_idx]
    outer_bearing = fold['outer_test_bearing']
    print(f'\\n{"="*60}')
    print(f'Fold {fold_idx} | {model_name.upper()} | outer_test={outer_bearing}')
    print(f'{"="*60}')

    # ── Build target indices ──
    train_targets = build_target_index(fold['fit_bearings'], EXPECTED_N_FILES, SEQUENCE_LENGTH)
    val_targets = build_target_index([fold['inner_val_bearing']], EXPECTED_N_FILES, SEQUENCE_LENGTH)
    test_targets = build_target_index([outer_bearing], EXPECTED_N_FILES, SEQUENCE_LENGTH)
    print(f'Targets: train={len(train_targets)}  val={len(val_targets)}  test={len(test_targets)}')

    # ── pos_weight from training labels ONLY (for audit, not applied to loss) ──
    pw_audit = compute_pos_weight(train_targets, cache_by_bearing)
    print(f'pos_weight_audit={pw_audit:.4f} (logged, but NOT used in loss because of Sampler)')

    # ── Build models (paired init, identical encoder) ──
    cnn, cnn_lstm = build_paired_models(MODEL_SEED)
    model = cnn if model_name == 'cnn' else cnn_lstm
    model = model.to(DEVICE)

    # ── Set training seed (IDENTICAL for both models in the same fold) ──
    train_seed = MODEL_SEED * 1000 + fold_idx
    set_deterministic_seed(train_seed)
    print(f'Training seed: {train_seed}')

    # ── Build datasets ──
    DatasetClass = CnnLstmSequenceDataset if model_name == 'cnn_lstm' else CnnSampleDataset
    train_ds = DatasetClass(cache_by_bearing, train_targets, scaler['mean'], scaler['std'])
    val_ds = DatasetClass(cache_by_bearing, val_targets, scaler['mean'], scaler['std'])
    test_ds = DatasetClass(cache_by_bearing, test_targets, scaler['mean'], scaler['std'])

    # ── DataLoaders ──
    sampler_generator = torch.Generator().manual_seed(train_seed)
    sampler = build_balanced_sampler(train_targets, cache_by_bearing, generator=sampler_generator)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    # test_loader is NOT created here — built only AFTER all decisions are frozen

    # ── Loss, optimizer ──
    # We omit pos_weight to prevent double-weighting alongside WeightedRandomSampler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop with early stopping on INNER-VAL only ──
    stopper = EarlyStoppingTracker(patience=PATIENCE, mode='max')
    history = []
    best_state = None

    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_logits, val_y = evaluate_epoch(model, val_loader, criterion, DEVICE)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid
        val_metrics = compute_sample_metrics(val_y, val_probs)
        val_auc_pr = val_metrics.get('auc_pr') or 0.0

        improved = stopper.step(val_auc_pr, epoch)
        if improved:
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_auc_pr': round(val_auc_pr, 6),
        })

        tag = ' *' if improved else ''
        print(f'  epoch {epoch:3d}  train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  val_aucpr={val_auc_pr:.4f}{tag}')

        if stopper.should_stop():
            print(f'  Early stopping at epoch {epoch} (patience={PATIENCE})')
            break

    best_epoch = stopper.best_epoch
    print(f'Best epoch: {best_epoch}  val_auc_pr: {stopper.best_value:.4f}')

    # ── Load best checkpoint ──
    model.load_state_dict(best_state)
    checkpoint_hash = state_dict_sha256(best_state)

    # ── Select threshold from INNER-VALIDATION only ──
    inner_val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)
    inner_val_records = evaluate_records(model, inner_val_loader, DEVICE)
    inner_val_reference_fpt = cache_by_bearing[fold['inner_val_bearing']]['fpt_index']
    selected_threshold_info = select_threshold_from_inner_val(
        inner_val_records,
        reference_fpt=inner_val_reference_fpt,
        threshold_grid=THRESHOLD_GRID,
    )
    selected_threshold = float(selected_threshold_info['selected_threshold'])
    print(f'Selected threshold from inner-val: {selected_threshold:.2f}')
    print(f'Calibration warnings: {selected_threshold_info["calibration_warning"]}')

    # ── Evaluate outer-test ONCE (all decisions frozen) ──
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_records = evaluate_records(model, test_loader, DEVICE)
    test_y_true = np.array([r['y_true'] for r in test_records])
    test_y_logit = np.array([r['y_logit'] for r in test_records])
    test_y_prob = np.array([r['y_prob'] for r in test_records])
    test_file_idx = np.array([r['file_index'] for r in test_records])
    test_target_idx = np.array([r['target_index'] for r in test_records])

    sample_metrics = compute_sample_metrics(test_y_true, test_y_prob)
    ref_fpt = cache_by_bearing[outer_bearing]['fpt_index']
    outer_event_metrics_fixed = compute_event_metrics(
        test_records,
        ref_fpt,
        threshold=FIXED_THRESHOLD_REFERENCE,
        consecutive=FPT_CONSECUTIVE,
    )
    outer_event_metrics_calibrated = compute_event_metrics(
        test_records,
        ref_fpt,
        threshold=selected_threshold,
        consecutive=FPT_CONSECUTIVE,
    )
    event_metrics = outer_event_metrics_calibrated

    training_time = time.time() - t0
    print(f'\\nOuter-test sample metrics: {json.dumps(sample_metrics, indent=2)}')
    print(f'Outer-test calibrated event metrics:  {json.dumps(event_metrics, indent=2)}')
    print(f'Training time: {training_time:.1f}s')

    # ── Save artifacts ──
    prefix = OUT_DIR / f'fold_{fold_idx}_{model_name}'

    # History CSV
    history_path = prefix.with_name(f'{prefix.name}_history.csv')
    with open(history_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_auc_pr'])
        writer.writeheader()
        writer.writerows(history)

    # Best checkpoint
    ckpt_path = prefix.with_name(f'{prefix.name}_best.pt')
    checkpoint_provenance = {
        'version': 'fpt-threshold-calibration-checkpoint-v1',
        'task': 'fpt_detection',
        'analysis_type': 'secondary_inner_val_threshold_calibration',
        'fold_index': fold_idx,
        'model_name': model_name,
        'fit_bearings': list(fold['fit_bearings']),
        'inner_val_bearing': fold['inner_val_bearing'],
        'outer_test_bearing': outer_bearing,
        'train_target_count': len(train_targets),
        'val_target_count': len(val_targets),
        'test_target_count': len(test_targets),
        'best_epoch': best_epoch,
        'best_val_auc_pr': float(stopper.best_value),
        'fold_manifest_sha256': input_hashes['fold_manifest_sha256'],
        'scaler_sha256': input_hashes['scaler_sha256'][fold_idx],
        'model_contract_sha256': input_hashes['model_contract_sha256'],
        'cache_sha256': input_hashes['cache_sha256'],
        'label_version': LABEL_VERSION,
        'cache_version': CACHE_VERSION,
        'model_seed': MODEL_SEED,
        'training_seed': train_seed,
        'sequence_length': SEQUENCE_LENGTH,
        'fixed_threshold_reference': FIXED_THRESHOLD_REFERENCE,
        'selected_threshold': selected_threshold,
        'threshold_selection_source': 'inner_validation_only',
        'calibration_warning': selected_threshold_info['calibration_warning'],
        'fpt_consecutive': FPT_CONSECUTIVE,
        'official_data_policy': 'closed; learning bearings only',
    }
    torch.save({
        'model_state_dict': best_state,
        'epoch': best_epoch,
        'val_auc_pr': stopper.best_value,
        'fold_index': fold_idx,
        'model_name': model_name,
        'provenance': checkpoint_provenance,
    }, ckpt_path)

    # Predictions NPZ
    pred_path = prefix.with_name(f'{prefix.name}_predictions.npz')
    np.savez(pred_path,
             y_true=test_y_true, y_logit=test_y_logit, y_prob=test_y_prob,
             file_index=test_file_idx, target_index=test_target_idx,
             bearing_id=np.array([outer_bearing]))

    # Inner-validation predictions used for threshold selection
    inner_val_pred_path = prefix.with_name(f'{prefix.name}_inner_val_predictions.npz')
    np.savez(
        inner_val_pred_path,
        y_true=np.array([r['y_true'] for r in inner_val_records]),
        y_logit=np.array([r['y_logit'] for r in inner_val_records]),
        y_prob=np.array([r['y_prob'] for r in inner_val_records]),
        file_index=np.array([r['file_index'] for r in inner_val_records]),
        target_index=np.array([r['target_index'] for r in inner_val_records]),
        bearing_id=np.array([fold['inner_val_bearing']]),
    )

    # Inner-validation threshold scan
    threshold_scan_path = prefix.with_name(f'{prefix.name}_inner_val_threshold_scan.csv')
    write_threshold_scan_csv(selected_threshold_info['threshold_scan'], threshold_scan_path)
             
    # Confusion Matrix
    save_confusion_matrix_artifacts(test_y_true, test_y_prob, selected_threshold, prefix)
    
    # Probability / FPT timeline plot
    save_fpt_probability_plot(
        file_indices=test_file_idx,
        y_true=test_y_true,
        y_prob=test_y_prob,
        reference_fpt=ref_fpt,
        predicted_fpt=outer_event_metrics_calibrated['predicted_fpt'],
        threshold=selected_threshold,
        out_path=prefix.with_name(f'{prefix.name}_probability_plot.png'),
        title=f'{model_name.upper()} | fold {fold_idx} | outer-test {outer_bearing} | calibrated threshold',
    )

    # Inner-validation threshold calibration plot
    save_fpt_probability_plot(
        file_indices=np.array([r['file_index'] for r in inner_val_records]),
        y_true=np.array([r['y_true'] for r in inner_val_records]),
        y_prob=np.array([r['y_prob'] for r in inner_val_records]),
        reference_fpt=inner_val_reference_fpt,
        predicted_fpt=selected_threshold_info['inner_val_event_metrics']['predicted_fpt'],
        threshold=selected_threshold,
        out_path=prefix.with_name(f'{prefix.name}_inner_val_calibration_plot.png'),
        title=f'{model_name.upper()} | fold {fold_idx} | inner-val {fold["inner_val_bearing"]} | inner-validation threshold calibration',
    )

    # Metrics JSON
    result = {
        'fold_index': fold_idx,
        'model': model_name,
        'outer_test_bearing': outer_bearing,
        'inner_val_bearing': fold['inner_val_bearing'],
        'best_epoch': best_epoch,
        'best_val_auc_pr': round(stopper.best_value, 6),
        'total_epochs': len(history),
        'selected_threshold': selected_threshold,
        'threshold_selection_source': 'inner_validation_only',
        'calibration_warning': selected_threshold_info['calibration_warning'],
        'inner_val_event_metrics_selected': selected_threshold_info['inner_val_event_metrics'],
        'outer_event_metrics_fixed_threshold_0_5': outer_event_metrics_fixed,
        'sample_metrics': sample_metrics,
        'event_metrics': event_metrics,
        'checkpoint_sha256': checkpoint_hash,
        'checkpoint_provenance': checkpoint_provenance,
        'scaler_sha256': input_hashes['scaler_sha256'][fold_idx],
        'training_seed': train_seed,
        'training_time_seconds': round(training_time, 1),
        'pos_weight_audit': round(pw_audit, 4),
    }
    metrics_path = prefix.with_name(f'{prefix.name}_metrics.json')
    atomic_write_json(metrics_path, result)
    print(f'Saved artifacts for Fold {fold_idx} {model_name.upper()}')
    return result


print('Per-fold training function OK')"""),

    # ── Cell 10: Load data + validate + smoke ─────────────────────────
    code("""\
# 9) Load all CWT caches and validate inputs

print('Loading CWT caches...')
cache_by_bearing = {}
for bearing in BEARINGS:
    cache_path = DATA_DIR / f'{bearing}_cwt_unscaled.npz'
    cache_by_bearing[bearing] = load_npz_payload(cache_path)
    n = len(cache_by_bearing[bearing]['y_state'])
    fpt = cache_by_bearing[bearing]['fpt_index']
    assert n == EXPECTED_N_FILES[bearing], f'{bearing}: expected {EXPECTED_N_FILES[bearing]}, got {n}'
    assert fpt == FROZEN_FPT[bearing], f'{bearing}: expected FPT {FROZEN_FPT[bearing]}, got {fpt}'
    print(f'  {bearing}: {n} files, FPT={fpt}')

print('\\nValidating input artifacts...')
fold_manifest, scalers, input_hashes = validate_inputs(DATA_DIR)
print(f'  fold_manifest SHA-256: {input_hashes["fold_manifest_sha256"][:16]}...')
print(f'  model_contract SHA-256: {input_hashes["model_contract_sha256"][:16]}...')
print('  All input hashes verified.')

# Save frozen training protocol config BEFORE any training
protocol_config = {
    'version': 'fpt-threshold-calibration-protocol-v1',
    'task': 'fpt_detection',
    'analysis_type': 'secondary_inner_val_threshold_calibration',
    'models': ['cnn', 'cnn_lstm'],
    'pilot_only': PILOT_ONLY,
    'pilot_fold': PILOT_FOLD,
    'sequence_length': SEQUENCE_LENGTH,
    'model_seed': MODEL_SEED,
    'encoder_dim': ENCODER_DIM,
    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'early_stopping_metric': 'val_auc_pr',
    'early_stopping_mode': 'max',
    'loss': 'BCEWithLogitsLoss',
    'loss_pos_weight': None,
    'optimizer': 'AdamW',
    'probability_threshold': PROBABILITY_THRESHOLD,
    'fixed_threshold_reference': FIXED_THRESHOLD_REFERENCE,
    'threshold_grid': THRESHOLD_GRID,
    'calibration_selection_rule': CALIBRATION_SELECTION_RULE,
    'threshold_selection_source': 'inner_validation_only',
    'outer_test_policy': 'evaluate_once_after_threshold_selection',
    'fpt_consecutive': FPT_CONSECUTIVE,
    'sampler': 'WeightedRandomSampler_bearing_class_groups',
    'pos_weight_source': 'training_labels_only_for_audit',
    'data_dir': str(DATA_DIR),
    'data_dir_env_var': DATA_DIR_ENV_VAR,
    'download_zip_path': str(ZIP_PATH),
    'required_input_files': required_input_files(),
    'fold_manifest_sha256': input_hashes['fold_manifest_sha256'],
    'model_contract_sha256': input_hashes['model_contract_sha256'],
    'label_version': LABEL_VERSION,
    'cache_version': CACHE_VERSION,
    'official_data_policy': 'closed; learning bearings only',
}
atomic_write_json(OUT_DIR / 'threshold_calibration_protocol_config.json', protocol_config)
print('Saved threshold_calibration_protocol_config.json')

# Run smoke gate
smoke_report = run_smoke_gate(fold_manifest, scalers, input_hashes, cache_by_bearing)"""),

    # ── Cell 11: Train (pilot or full) ────────────────────────────────
    code("""\
# 10) Training execution

fold_indices = [PILOT_FOLD] if PILOT_ONLY else list(range(6))
print(f'\\nTraining folds: {fold_indices}  (PILOT_ONLY={PILOT_ONLY})')

all_results = []
for fold_idx in fold_indices:
    for model_name in ['cnn', 'cnn_lstm']:
        result = train_fold_model(
            fold_idx, model_name, fold_manifest, scalers,
            cache_by_bearing, input_hashes,
        )
        all_results.append(result)

print(f'\\nCompleted {len(all_results)} training runs.')"""),

    # ── Cell 12: Aggregate metrics + manifest ─────────────────────────
    code("""\
# 11) Aggregate metrics and generate manifest

def aggregate_results(results, model_name):
    '''Macro-average metrics across folds for one model.'''
    model_results = [r for r in results if r['model'] == model_name]
    if not model_results:
        return None
    sample_keys = ['precision', 'recall', 'f1', 'auc_pr', 'roc_auc', 'balanced_accuracy']
    agg = {}
    for key in sample_keys:
        values = [r['sample_metrics'][key] for r in model_results
                  if r['sample_metrics'][key] is not None]
        if values:
            agg[f'{key}_mean'] = round(float(np.mean(values)), 4)
            agg[f'{key}_std'] = round(float(np.std(values)), 4)
        else:
            agg[f'{key}_mean'] = None
            agg[f'{key}_std'] = None

    abs_errors = [
        r['event_metrics']['abs_fpt_error_files']
        for r in model_results
        if r['event_metrics']['abs_fpt_error_files'] is not None
    ]
    signed_delays = [
        r['event_metrics']['signed_delay_files']
        for r in model_results
        if r['event_metrics']['signed_delay_files'] is not None
    ]
    selected_thresholds = [r['selected_threshold'] for r in model_results]

    agg['abs_fpt_error_mean'] = round(float(np.mean(abs_errors)), 1) if abs_errors else None
    agg['abs_fpt_error_std'] = round(float(np.std(abs_errors)), 1) if abs_errors else None
    agg['signed_delay_mean'] = round(float(np.mean(signed_delays)), 1) if signed_delays else None
    agg['signed_delay_std'] = round(float(np.std(signed_delays)), 1) if signed_delays else None
    agg['selected_threshold_mean'] = round(float(np.mean(selected_thresholds)), 3)
    agg['selected_threshold_std'] = round(float(np.std(selected_thresholds)), 3)
    agg['missed_detections'] = sum(1 for r in model_results if r['event_metrics']['missed_detection'])
    agg['pre_fpt_alarm_runs'] = sum(1 for r in model_results if r['event_metrics']['pre_fpt_alarm_run'])
    agg['early_detections'] = sum(
        1 for r in model_results
        if r['event_metrics']['signed_delay_files'] is not None
        and r['event_metrics']['signed_delay_files'] < 0
    )
    agg['late_detections'] = sum(
        1 for r in model_results
        if r['event_metrics']['signed_delay_files'] is not None
        and r['event_metrics']['signed_delay_files'] > 0
    )
    agg['exact_detections'] = sum(
        1 for r in model_results
        if r['event_metrics']['signed_delay_files'] == 0
    )
    agg['total_folds'] = len(model_results)
    return agg


# Per-model summary
summary = {}
for model_name in ['cnn', 'cnn_lstm']:
    agg = aggregate_results(all_results, model_name)
    if agg:
        summary[model_name] = agg
        print(f'\\n=== {model_name.upper()} macro summary ===')
        for k, v in agg.items():
            print(f'  {k}: {v}')

# Per-fold detail table
print('\\n=== Per-fold results ===')
print(f'{" Fold":>5} {"Model":>10} {"AUC-PR":>8} {"F1":>8} {"Ref FPT":>8} '
      f'{"Thr":>5} {"Pred FPT":>9} {"Delay":>7} {"Missed":>7}')
for r in sorted(all_results, key=lambda x: (x['fold_index'], x['model'])):
    sm = r['sample_metrics']
    em = r['event_metrics']
    aucpr = f'{sm["auc_pr"]:.4f}' if sm['auc_pr'] is not None else '  N/A'
    f1 = f'{sm["f1"]:.4f}'
    pred = str(em['predicted_fpt']) if em['predicted_fpt'] is not None else 'MISSED'
    delay = str(em['signed_delay_files']) if em['signed_delay_files'] is not None else 'N/A'
    missed = 'YES' if em['missed_detection'] else 'no'
    print(f'{r["fold_index"]:>5} {r["model"]:>10} {aucpr:>8} {f1:>8} '
          f'{em["reference_fpt"]:>8} {r["selected_threshold"]:>5.2f} {pred:>9} {delay:>7} {missed:>7}')

# Save manifest
manifest = {
    'version': 'fpt-threshold-calibration-manifest-v1',
    'analysis_type': 'secondary_inner_val_threshold_calibration',
    'pilot_only': PILOT_ONLY,
    'fold_indices': fold_indices,
    'n_runs': len(all_results),
    'per_fold_results': all_results,
    'summary': summary,
    'protocol_config_path': str(OUT_DIR / 'threshold_calibration_protocol_config.json'),
    'download_zip_path': str(ZIP_PATH),
    'official_data_policy': 'closed; learning bearings only',
    'threshold_selection_source': 'inner_validation_only',
}
manifest_path = OUT_DIR / 'threshold_calibration_manifest.json'
atomic_write_json(manifest_path, manifest)
print(f'\\nSaved: {manifest_path}')

# Save summary metrics
if summary:
    summary_path = OUT_DIR / 'summary_metrics_calibrated.json'
    atomic_write_json(summary_path, summary)
    print(f'Saved: {summary_path}')

zip_report = create_artifact_zip(OUT_DIR, ZIP_PATH)
print(f'Saved downloadable zip: {ZIP_PATH}')
print(f'Zip artifact count: {zip_report["n_files"]}')

print('\\n=== Task 6 threshold calibration complete ===')"""),
]

# ── Write notebook ────────────────────────────────────────────────────

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[2] / "notebooks" / "08_threshold_calibration.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {output.name}  ({len(cells)} cells)")
