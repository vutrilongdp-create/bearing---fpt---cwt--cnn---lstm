"""Generate the Kaggle notebook for leakage-safe fold-specific CWT scalers."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(
        """# Task 3 — deterministic folds and train-only CWT scalers

This notebook validates the frozen Task 2 caches, defines six leave-one-bearing-out
folds, and fits one per-channel z-score scaler per fold. Scaler statistics use only
the four outer-training bearings. Inner-validation and outer-test CWT arrays remain
unopened until the scaler artifact has been written atomically.

No model, optimizer, early stopping, threshold selection, or official test data is
used in this notebook."""
    ),
    code(
        """# 1) Imports and frozen contracts
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path('/kaggle/input/datasets/longvu274/dataset-slug/')
OUT_DIR = Path('/kaggle/working/fpt_fold_scalers/')
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
EXPECTED_FPT = {
    'bearing1_1': 1871, 'bearing1_2': 826,
    'bearing2_1': 151, 'bearing2_2': 198,
    'bearing3_1': 493, 'bearing3_2': 1597,
}
CACHE_VERSION = 'cwt-unscaled-log-power-v2'
CACHE_KIND = 'unscaled_log_power'
LABEL_VERSION = 'causal-irrms-fpt-reference-v3-q99'
FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
SCALER_VERSION = 'per-channel-zscore-v1'
SCALER_KIND = 'per_channel_zscore'
SCALER_DDOF = 0
SCALER_EPSILON = 1e-8
CHANNEL_NAMES = ['horizontal', 'vertical']
CHUNK_FILES = 16

print('CACHE_DIR:', CACHE_DIR)
print('OUT_DIR  :', OUT_DIR)"""
    ),
    code(
        """# 2) Pure deterministic fold schedule
def build_fold_schedule(bearings: list[str]) -> list[dict]:
    if len(bearings) != 6 or len(set(bearings)) != 6:
        raise ValueError('Expected six unique learning bearings')
    folds = []
    for fold_index, outer_test in enumerate(bearings):
        inner_val = bearings[(fold_index + 1) % len(bearings)]
        fit_bearings = [
            bearing for bearing in bearings
            if bearing not in {outer_test, inner_val}
        ]
        folds.append({
            'fold_index': fold_index,
            'fit_bearings': fit_bearings,
            'inner_val_bearing': inner_val,
            'outer_test_bearing': outer_test,
        })
    return folds"""
    ),
    code(
        """# 3) Pure per-channel parallel Welford helpers
def empty_channel_moments(n_channels: int = 2) -> dict:
    return {
        'count': np.zeros(n_channels, dtype=np.int64),
        'mean': np.zeros(n_channels, dtype=np.float64),
        'm2': np.zeros(n_channels, dtype=np.float64),
    }


def batch_channel_moments(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(f'Expected [N,2,H,W], got {values.shape}')
    values64 = values.astype(np.float64)
    axes = (0, 2, 3)
    count = int(values64.shape[0] * values64.shape[2] * values64.shape[3])
    mean = values64.mean(axis=axes, dtype=np.float64)
    centered = values64 - mean.reshape(1, 2, 1, 1)
    m2 = np.square(centered).sum(axis=axes, dtype=np.float64)
    return {
        'count': np.full(2, count, dtype=np.int64),
        'mean': mean,
        'm2': m2,
    }


def merge_channel_moments(left: dict, right: dict) -> dict:
    left_count = np.asarray(left['count'], dtype=np.int64)
    right_count = np.asarray(right['count'], dtype=np.int64)
    total = left_count + right_count
    if np.any(total <= 0):
        raise ValueError('Cannot merge empty channel moments')
    delta = np.asarray(right['mean']) - np.asarray(left['mean'])
    safe_total = total.astype(np.float64)
    mean = np.asarray(left['mean']) + delta * right_count / safe_total
    cross = np.square(delta) * left_count * right_count / safe_total
    m2 = np.asarray(left['m2']) + np.asarray(right['m2']) + cross
    return {'count': total, 'mean': mean, 'm2': m2}


def finalize_channel_scaler(state: dict, epsilon: float = 1e-8) -> dict:
    count = np.asarray(state['count'], dtype=np.int64)
    if np.any(count <= 0):
        raise ValueError('Scaler count must be positive')
    mean = np.asarray(state['mean'], dtype=np.float64)
    std = np.sqrt(np.asarray(state['m2'], dtype=np.float64) / count)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Scaler mean/std must be finite')
    if np.any(std <= epsilon):
        raise ValueError(f'Scaler std must exceed {epsilon}')
    return {'count': count, 'mean': mean, 'std': std}


def apply_channel_zscore(values: np.ndarray, mean, std) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[-3] != 2:
        raise ValueError(f'Expected [...,2,H,W], got {values.shape}')
    broadcast_shape = [1] * values.ndim
    broadcast_shape[-3] = 2
    mean32 = np.asarray(mean, dtype=np.float32).reshape(broadcast_shape)
    std32 = np.asarray(std, dtype=np.float32).reshape(broadcast_shape)
    transformed = (values - mean32) / std32
    if not np.isfinite(transformed).all():
        raise ValueError('Scaled CWT contains non-finite values')
    return transformed.astype(np.float32, copy=False)


print('Welford and z-score helpers OK')"""
    ),
    code(
        """# 4) IO, Task 2 preflight, and atomic JSON
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


def validate_task2_manifest(cache_dir: Path) -> tuple[dict, dict, dict, str]:
    manifest_path = cache_dir / 'cwt_cache_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing Task 2 manifest: {manifest_path}')
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('cache_version') != CACHE_VERSION:
        raise ValueError('Task 2 cache version mismatch')
    if manifest.get('cache_kind') != CACHE_KIND:
        raise ValueError('Task 2 cache kind mismatch')
    if manifest.get('label_version') != LABEL_VERSION:
        raise ValueError('Task 2 label version mismatch')
    if not np.isclose(
        float(manifest.get('frozen_irrms_threshold')),
        FROZEN_IRRMS_THRESHOLD,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError('Task 2 frozen threshold mismatch')
    entries = manifest.get('entries', [])
    if len(entries) != 6:
        raise ValueError('Task 2 manifest must contain six entries')
    entry_by_bearing = {entry['bearing_id']: entry for entry in entries}
    if set(entry_by_bearing) != set(BEARINGS) or len(entry_by_bearing) != 6:
        raise ValueError('Task 2 bearing set mismatch or duplicate')

    cache_paths = {}
    for bearing in BEARINGS:
        entry = entry_by_bearing[bearing]
        expected_shape = [EXPECTED_N_FILES[bearing], 2, 128, 128]
        if entry.get('shape') != expected_shape or entry.get('dtype') != 'float16':
            raise ValueError(f'{bearing}: Task 2 shape/dtype mismatch')
        if int(entry.get('fpt_index')) != EXPECTED_FPT[bearing]:
            raise ValueError(f'{bearing}: Task 2 FPT mismatch')
        cache_path = cache_dir / f'{bearing}_cwt_unscaled.npz'
        if not cache_path.exists():
            raise FileNotFoundError(f'Missing Task 2 cache: {cache_path}')
        actual_sha256 = sha256_file(cache_path)
        if actual_sha256 != entry.get('cache_sha256'):
            raise ValueError(f'{bearing}: Task 2 cache SHA-256 mismatch')
        cache_paths[bearing] = cache_path
    return manifest, entry_by_bearing, cache_paths, manifest_sha256


print('Task 2 preflight helpers OK')"""
    ),
    code(
        """# 5) Fit only from outer-training caches
def fit_fold_scaler(
    fold: dict,
    cache_paths: dict,
    entry_by_bearing: dict,
    task2_manifest_sha256: str,
) -> dict:
    fit_bearings = list(fold['fit_bearings'])
    excluded = {fold['inner_val_bearing'], fold['outer_test_bearing']}
    if len(fit_bearings) != 4 or set(fit_bearings) & excluded:
        raise ValueError('Scaler provenance must contain exactly four train bearings')
    state = empty_channel_moments(2)
    for bearing in fit_bearings:
        with np.load(cache_paths[bearing], allow_pickle=False) as data:
            cwt = data['cwt_log_power']
            expected_shape = (EXPECTED_N_FILES[bearing], 2, 128, 128)
            if cwt.shape != expected_shape or cwt.dtype != np.float16:
                raise ValueError(f'{bearing}: cache array contract mismatch')
            for start in range(0, len(cwt), CHUNK_FILES):
                stop = min(start + CHUNK_FILES, len(cwt))
                state = merge_channel_moments(
                    state, batch_channel_moments(cwt[start:stop])
                )
    stats = finalize_channel_scaler(state, epsilon=SCALER_EPSILON)
    return {
        'scaler_version': SCALER_VERSION,
        'scaler_kind': SCALER_KIND,
        'ddof': SCALER_DDOF,
        'epsilon': SCALER_EPSILON,
        'fold_index': int(fold['fold_index']),
        'fit_bearings': fit_bearings,
        'inner_val_bearing': fold['inner_val_bearing'],
        'outer_test_bearing': fold['outer_test_bearing'],
        'channel_axis': 1,
        'channel_names': CHANNEL_NAMES,
        'pixel_count_per_channel': stats['count'].astype(int).tolist(),
        'mean': stats['mean'].tolist(),
        'std': stats['std'].tolist(),
        'task2_manifest_sha256': task2_manifest_sha256,
        'fit_cache_sha256': {
            bearing: entry_by_bearing[bearing]['cache_sha256']
            for bearing in fit_bearings
        },
        'frozen_irrms_threshold': FROZEN_IRRMS_THRESHOLD,
        'label_version': LABEL_VERSION,
    }


print('Train-only scaler fitter OK')"""
    ),
    code(
        """# 6) Post-fit probes for non-training bearings
def probe_nonfit_cache(
    cache_paths: dict,
    bearings: list[str],
    mean: list[float],
    std: list[float],
) -> dict:
    details = {}
    all_finite = True
    for bearing in bearings:
        with np.load(cache_paths[bearing], allow_pickle=False) as data:
            cwt = data['cwt_log_power']
            indices = [0, len(cwt) // 2, len(cwt) - 1]
            scaled = apply_channel_zscore(cwt[indices], mean, std)
            finite = bool(np.isfinite(scaled).all())
            shape_ok = scaled.shape == (3, 2, 128, 128)
            all_finite = all_finite and finite and shape_ok
            details[bearing] = {
                'indices': indices,
                'finite': finite,
                'shape_ok': shape_ok,
            }
    return {'all_finite': bool(all_finite), 'details': details}


print('Post-fit probe helper OK')"""
    ),
    code(
        """# 7) Preflight, fit six scalers, then probe held-out caches
FOLDS = build_fold_schedule(BEARINGS)
display(pd.DataFrame(FOLDS))

task2_manifest, entry_by_bearing, cache_paths, task2_manifest_sha256 = (
    validate_task2_manifest(CACHE_DIR)
)

audit_rows = []
fold_records = []
for fold in FOLDS:
    fold_index = int(fold['fold_index'])
    print('\\n' + '=' * 80)
    print('FOLD', fold_index, fold)
    scaler = fit_fold_scaler(
        fold, cache_paths, entry_by_bearing, task2_manifest_sha256
    )
    scaler_path = OUT_DIR / f'fold_{fold_index}_scaler.json'
    atomic_write_json(scaler_path, scaler)
    scaler_sha256 = sha256_file(scaler_path)

    probe_results = probe_nonfit_cache(
        cache_paths,
        [fold['inner_val_bearing'], fold['outer_test_bearing']],
        scaler['mean'],
        scaler['std'],
    )
    fit_set = set(fold['fit_bearings'])
    flags = {
        'train_count_is_four': len(fit_set) == 4,
        'sets_are_disjoint': not fit_set.intersection({
            fold['inner_val_bearing'], fold['outer_test_bearing']
        }),
        'outer_not_in_fit': fold['outer_test_bearing'] not in fit_set,
        'inner_not_in_fit': fold['inner_val_bearing'] not in fit_set,
        'std_is_valid': bool(np.all(np.asarray(scaler['std']) > SCALER_EPSILON)),
        'postfit_probe_finite': bool(probe_results['all_finite']),
    }
    if not all(flags.values()):
        raise RuntimeError(f'Fold {fold_index} audit failed: {flags}')

    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        audit_rows.append({
            'fold_index': fold_index,
            'channel_index': channel_index,
            'channel_name': channel_name,
            'fit_bearings': '|'.join(fold['fit_bearings']),
            'inner_val_bearing': fold['inner_val_bearing'],
            'outer_test_bearing': fold['outer_test_bearing'],
            'pixel_count': scaler['pixel_count_per_channel'][channel_index],
            'mean': scaler['mean'][channel_index],
            'std': scaler['std'][channel_index],
            'scaler_path': str(scaler_path),
            'scaler_sha256': scaler_sha256,
            **flags,
        })
    fold_records.append({
        **fold,
        'scaler_path': str(scaler_path),
        'scaler_sha256': scaler_sha256,
        'postfit_probes': probe_results,
        'audit_flags': flags,
    })

audit_df = pd.DataFrame(audit_rows)
audit_path = OUT_DIR / 'fold_scaler_audit.csv'
audit_df.to_csv(audit_path, index=False)

fold_manifest = {
    'version': 'fpt-fold-scalers-v1',
    'cache_dir': str(CACHE_DIR),
    'task2_manifest_sha256': task2_manifest_sha256,
    'canonical_bearings': BEARINGS,
    'scaler_version': SCALER_VERSION,
    'scaler_kind': SCALER_KIND,
    'ddof': SCALER_DDOF,
    'epsilon': SCALER_EPSILON,
    'channel_names': CHANNEL_NAMES,
    'frozen_irrms_threshold': FROZEN_IRRMS_THRESHOLD,
    'label_version': LABEL_VERSION,
    'official_data_policy': 'closed; learning bearings only',
    'folds': fold_records,
    'audit_csv': str(audit_path),
    'audit_csv_sha256': sha256_file(audit_path),
}
fold_manifest_path = OUT_DIR / 'fold_manifest.json'
atomic_write_json(fold_manifest_path, fold_manifest)

display(audit_df)
print('Saved:', fold_manifest_path)
print('Task 3 complete: fold-specific train-only scalers; no model training.')"""
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

output = Path(__file__).resolve().parents[2] / "notebooks" / "05_cv_folds_and_scalers.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
