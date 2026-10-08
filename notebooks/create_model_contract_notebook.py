"""Generate the Kaggle notebook that freezes causal datasets and model shapes."""

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
    md(
        """# Task 4 — causal datasets and shared CWT encoder contract

This notebook validates aligned CNN/CNN-LSTM target timestamps, causal sequences,
fold-specific scaling, shared encoder architecture, and paired initialization.
It performs forward smoke tests only; there is no training or model selection."""
    ),
    code(
        """# 1) Imports and frozen contract
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DATA_DIR = Path('/kaggle/input/dataset-slug/')
OUT_DIR = Path('/kaggle/working/fpt_model_contract/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [
    'bearing1_1', 'bearing1_2', 'bearing2_1',
    'bearing2_2', 'bearing3_1', 'bearing3_2',
]
EXPECTED_N_FILES = {
    'bearing1_1': 2803, 'bearing1_2': 871,
    'bearing2_1': 911, 'bearing2_2': 797,
    'bearing3_1': 515, 'bearing3_2': 1637,
}
SEQUENCE_LENGTH = 16
ENCODER_DIM = 128
LSTM_HIDDEN_SIZE = 128
MODEL_SEED = 274
BATCH_SIZE = 2

print('DATA_DIR:', DATA_DIR)
print('OUT_DIR :', OUT_DIR)"""
    ),
    code(
        """# 2) Pure target-index and scaling helpers
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


print('Pure target/scaling helpers OK')"""
    ),
    code(
        """# 3) Aligned causal datasets
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


print('Aligned causal datasets OK')"""
    ),
    code(
        """# 4) Shared GroupNorm encoder and classifiers
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


print('Shared model definitions OK')"""
    ),
    code(
        """# 5) Deterministic paired initialization and hashing
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


print('Paired initialization helpers OK')"""
    ),
    code(
        """# 6) Input artifact validation helpers
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
    temp_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temp_path.replace(path)


def load_npz_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            'cwt_log_power': data['cwt_log_power'].copy(),
            'y_state': data['y_state'].copy(),
            'file_index': data['file_index'].copy(),
            'fpt_index': int(np.asarray(data['fpt_index']).item()),
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
    return fold_manifest, scalers, {
        'fold_manifest_sha256': fold_manifest_sha256,
        'scaler_sha256': scaler_sha256,
        'cache_sha256': cache_sha256,
    }


print('Input validation helpers OK')"""
    ),
    code(
        """# 7) Real-data smoke tests and model contract artifact
fold_manifest, scalers, input_hashes = validate_inputs(DATA_DIR)
target_counts_by_fold = []
for fold in fold_manifest['folds']:
    split_counts = {}
    split_bearings = {
        'train': fold['fit_bearings'],
        'inner_val': [fold['inner_val_bearing']],
        'outer_test': [fold['outer_test_bearing']],
    }
    for split, bearings in split_bearings.items():
        split_counts[split] = len(
            build_target_index(bearings, EXPECTED_N_FILES, SEQUENCE_LENGTH)
        )
    target_counts_by_fold.append({'fold_index': fold['fold_index'], **split_counts})

fold0 = fold_manifest['folds'][0]
outer_bearing = fold0['outer_test_bearing']
cache_by_bearing = {
    outer_bearing: load_npz_payload(DATA_DIR / f'{outer_bearing}_cwt_unscaled.npz')
}
scaler0 = scalers[0]
cnn_targets = build_target_index([outer_bearing], EXPECTED_N_FILES, SEQUENCE_LENGTH)
lstm_targets = build_target_index([outer_bearing], EXPECTED_N_FILES, SEQUENCE_LENGTH)
assert cnn_targets == lstm_targets
cnn_dataset = CnnSampleDataset(
    cache_by_bearing, cnn_targets, scaler0['mean'], scaler0['std']
)
lstm_dataset = CnnLstmSequenceDataset(
    cache_by_bearing, lstm_targets, scaler0['mean'], scaler0['std'], SEQUENCE_LENGTH
)

sample_items = [cnn_dataset[0], cnn_dataset[-1]]
sequence_items = [lstm_dataset[0], lstm_dataset[-1]]
cnn_batch = torch.stack([item[0] for item in sample_items])
lstm_batch = torch.stack([item[0] for item in sequence_items])
assert cnn_batch.shape == (BATCH_SIZE, 2, 128, 128)
assert lstm_batch.shape == (BATCH_SIZE, SEQUENCE_LENGTH, 2, 128, 128)
assert [item[1].item() for item in sample_items] == [
    item[1].item() for item in sequence_items
]

cnn, cnn_lstm = build_paired_models(MODEL_SEED)
encoder_hash_cnn = state_dict_sha256(cnn.encoder.state_dict())
encoder_hash_lstm = state_dict_sha256(cnn_lstm.encoder.state_dict())
encoder_hashes_equal = encoder_hash_cnn == encoder_hash_lstm
encoder_parameters_are_distinct = all(
    left.data_ptr() != right.data_ptr()
    for left, right in zip(cnn.encoder.parameters(), cnn_lstm.encoder.parameters())
)
group_norm_count = sum(isinstance(module, nn.GroupNorm) for module in cnn.encoder.modules())
no_batchnorm = group_norm_count == 4

cnn.eval()
cnn_lstm.eval()
with torch.no_grad():
    encoder_output = cnn.encoder(cnn_batch)
    cnn_logits = cnn(cnn_batch)
    lstm_logits = cnn_lstm(lstm_batch)
assert encoder_output.shape == (BATCH_SIZE, ENCODER_DIM)
assert cnn_logits.shape == (BATCH_SIZE,)
assert lstm_logits.shape == (BATCH_SIZE,)

audit_flags = {
    'cnn_targets_equal_lstm_targets': cnn_targets == lstm_targets,
    'first_sequence_is_causal': sequence_items[0][2]['indices'] == list(range(16)),
    'last_sequence_ends_at_last_file': sequence_items[-1][2]['target_index'] == EXPECTED_N_FILES[outer_bearing] - 1,
    'scaled_inputs_finite': bool(torch.isfinite(cnn_batch).all() and torch.isfinite(lstm_batch).all()),
    'encoder_hashes_equal': encoder_hashes_equal,
    'encoder_parameters_are_distinct': encoder_parameters_are_distinct,
    'no_batchnorm': no_batchnorm,
    'forward_shapes_valid': True,
}
if not all(audit_flags.values()):
    raise RuntimeError(f'Model contract audit failed: {audit_flags}')

contract = {
    'version': 'fpt-model-contract-v1',
    'data_dir': str(DATA_DIR),
    'sequence_length': SEQUENCE_LENGTH,
    'encoder_channels': [2, 16, 32, 64, 128],
    'encoder_dim': ENCODER_DIM,
    'normalization': 'GroupNorm(8)',
    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
    'model_seed': MODEL_SEED,
    'cnn_parameter_count': sum(p.numel() for p in cnn.parameters()),
    'cnn_lstm_parameter_count': sum(p.numel() for p in cnn_lstm.parameters()),
    'encoder_parameter_count': sum(p.numel() for p in cnn.encoder.parameters()),
    'encoder_hash_cnn': encoder_hash_cnn,
    'encoder_hash_lstm': encoder_hash_lstm,
    'target_counts_by_fold': target_counts_by_fold,
    'smoke_shapes': {
        'cnn_input': list(cnn_batch.shape),
        'lstm_input': list(lstm_batch.shape),
        'encoder_output': list(encoder_output.shape),
        'cnn_logits': list(cnn_logits.shape),
        'lstm_logits': list(lstm_logits.shape),
    },
    'input_hashes': input_hashes,
    'audit_flags': audit_flags,
    'official_data_policy': 'closed; learning bearings only',
}
contract_path = OUT_DIR / 'model_contract.json'
atomic_write_json(contract_path, contract)
print(json.dumps(contract, indent=2))
print('Saved:', contract_path)"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("04_validate_fpt_model_contract_kaggle.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
