"""Generate the Kaggle notebook that caches unscaled CWT log-power images."""

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
        """# Task 2 — cache unscaled CWT log-power once

Notebook này chỉ tạo cache CWT cho sáu PRONOSTIA learning bearings. Nó đọc
nhãn Task 1 đã freeze tại Q99 và không tính lại FPT.

Không có scaler, normalization theo bearing, model training hoặc model
selection trong notebook này. Mọi scaling học được phải chờ outer fold và chỉ
fit trên outer-training bearings."""
    ),
    code(
        """# 1) Imports
import gc
import hashlib
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from IPython.display import Image, display
from skimage.transform import resize

print('Imports OK')"""
    ),
    code(
        """# 2) Frozen paths and transform contract
RAW_DIR = Path('/kaggle/input/datasets/longvu274/train-dataset/')
LABEL_DIR = Path('/kaggle/input/causal-irrms-fpt-reference/')
OUT_DIR = Path('/kaggle/working/cwt_unscaled_cache/')
OUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR = OUT_DIR / 'visualizations'
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [
    'bearing1_1', 'bearing1_2',
    'bearing2_1', 'bearing2_2',
    'bearing3_1', 'bearing3_2',
]

DATA_POINTS = 2560
SAMPLING_FREQ = 25600.0
FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
LABEL_VERSION = 'causal-irrms-fpt-reference-v3-q99'
CACHE_VERSION = 'cwt-unscaled-log-power-v2'
CACHE_KIND = 'unscaled_log_power'
CWT_WAVELET = 'morl'
CWT_SCALES = np.geomspace(1, 512, 128)
IMAGE_SHAPE = (128, 128)
LOG_POWER_EPS = 1e-3
CACHE_DTYPE = np.float16
CWT_FREQUENCIES_HZ = pywt.scale2frequency(
    CWT_WAVELET, CWT_SCALES
) * SAMPLING_FREQ
NYQUIST_FREQUENCY_HZ = SAMPLING_FREQ / 2.0
ACQUISITION_DURATION_SECONDS = DATA_POINTS / SAMPLING_FREQ
OVERWRITE = False

print('RAW_DIR  :', RAW_DIR)
print('LABEL_DIR:', LABEL_DIR)
print('OUT_DIR  :', OUT_DIR)"""
    ),
    code(
        """# 3) Scale-frequency audit (diagnostic only; frozen scales do not change)
scale_frequency_df = pd.DataFrame({
    'scale': CWT_SCALES,
    'frequency_hz': CWT_FREQUENCIES_HZ,
    'above_nyquist': CWT_FREQUENCIES_HZ > NYQUIST_FREQUENCY_HZ,
})
N_SCALES_ABOVE_NYQUIST = int(scale_frequency_df['above_nyquist'].sum())
display(scale_frequency_df.head(10))
display(scale_frequency_df.tail(10))
print('CWT frequency range (Hz):',
      float(CWT_FREQUENCIES_HZ.min()), 'to',
      float(CWT_FREQUENCIES_HZ.max()))
print('Nyquist frequency (Hz):', NYQUIST_FREQUENCY_HZ)
print('Scales above Nyquist:', N_SCALES_ABOVE_NYQUIST)
if N_SCALES_ABOVE_NYQUIST:
    print('AUDIT WARNING: low CWT scales exceed Nyquist; transform contract remains frozen.')"""
    ),
    code(
        """# 4) Frozen-label validation (pure contract)
def validate_label_alignment(
    y_state,
    file_index,
    bearing_id,
    fpt_index,
    frozen_threshold,
    label_version,
    expected_bearing,
):
    y_state = np.asarray(y_state)
    file_index = np.asarray(file_index)
    bearing_id = np.asarray(bearing_id).astype(str)
    n_files = len(y_state)

    if y_state.ndim != 1 or n_files == 0:
        raise ValueError('y_state must be a non-empty 1-D array')
    if not np.array_equal(file_index, np.arange(n_files)):
        raise ValueError('file_index must equal zero-based arange(n_files)')
    if bearing_id.shape != (n_files,) or not np.all(bearing_id == expected_bearing):
        raise ValueError('bearing_id does not match expected bearing')
    if not set(np.unique(y_state)).issubset({0, 1}):
        raise ValueError('y_state must be binary')
    if not np.isclose(
        float(frozen_threshold), FROZEN_IRRMS_THRESHOLD, rtol=0.0, atol=1e-12
    ):
        raise ValueError('frozen threshold does not match Task 1 Q99')
    if str(label_version) != LABEL_VERSION:
        raise ValueError('label version does not match frozen Task 1 contract')

    positive = np.flatnonzero(y_state == 1)
    if len(positive) == 0 or int(positive[0]) != int(fpt_index):
        raise ValueError('first positive label must equal frozen FPT index')
    expected = (np.arange(n_files) >= int(fpt_index)).astype(np.uint8)
    if not np.array_equal(y_state.astype(np.uint8), expected):
        raise ValueError('labels must be Healthy before FPT and Degraded from FPT')
    return {'n_files': int(n_files), 'fpt_index': int(fpt_index)}


print('Frozen-label contract OK')"""
    ),
    code(
        """# 4) CWT transform — deliberately no scaling
def cwt_log_power_image(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or len(signal) != DATA_POINTS:
        raise ValueError(f'Expected one signal with {DATA_POINTS} points')
    if not np.isfinite(signal).all():
        raise ValueError('Signal contains non-finite values')

    coef, _ = pywt.cwt(
        signal,
        CWT_SCALES,
        CWT_WAVELET,
        sampling_period=1.0 / SAMPLING_FREQ,
    )
    log_power = np.log2(np.abs(coef) ** 2 + LOG_POWER_EPS)
    image = resize(
        log_power,
        IMAGE_SHAPE,
        preserve_range=True,
        anti_aliasing=True,
    )
    image = np.asarray(image, dtype=np.float32)
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError('Invalid CWT image')
    return image


def cwt_two_channel_record(record: np.ndarray) -> np.ndarray:
    record = np.asarray(record, dtype=np.float32)
    if record.shape != (2, DATA_POINTS):
        raise ValueError(f'Expected [2, {DATA_POINTS}], got {record.shape}')
    return np.stack(
        [cwt_log_power_image(record[channel]) for channel in range(2)],
        axis=0,
    )


print('CWT transform OK')"""
    ),
    code(
        """# 5) IO helpers and deterministic signatures
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(bearing: str) -> Path:
    return RAW_DIR / f'{bearing}.pkz'


def label_path(bearing: str) -> Path:
    return LABEL_DIR / f'{bearing}_fpt_labels.npz'


def load_raw_records(bearing: str) -> np.ndarray:
    path = raw_path(bearing)
    if not path.exists():
        raise FileNotFoundError(f'Missing raw bearing: {path}')
    with open(path, 'rb') as handle:
        df = pkl.load(handle)
    required = {'horiz accel', 'vert accel'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    if len(df) % DATA_POINTS != 0:
        raise ValueError(f'{path}: row count is not divisible by {DATA_POINTS}')
    n_files = len(df) // DATA_POINTS
    horizontal = df['horiz accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    vertical = df['vert accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    records = np.stack([horizontal, vertical], axis=1)
    if not np.isfinite(records).all():
        raise ValueError(f'{bearing}: raw records contain non-finite values')
    return records


def load_frozen_labels(bearing: str) -> dict:
    path = label_path(bearing)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing frozen labels: {path}. Publish/attach Task 1 output first.'
        )
    with np.load(path, allow_pickle=False) as data:
        required = {
            'y_state', 'file_index', 'bearing_id', 'fpt_index',
            'frozen_irrms_threshold', 'label_version',
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f'{path} missing arrays: {sorted(missing)}')
        labels = {name: data[name].copy() for name in required}
    validate_label_alignment(
        y_state=labels['y_state'],
        file_index=labels['file_index'],
        bearing_id=labels['bearing_id'],
        fpt_index=int(labels['fpt_index']),
        frozen_threshold=float(labels['frozen_irrms_threshold']),
        label_version=str(labels['label_version']),
        expected_bearing=bearing,
    )
    return labels


print('IO helpers OK')"""
    ),
    code(
        """# 7) Cache statistics and strict reuse validation
def finalize_streaming_stats(
    value_count: int,
    value_sum: float,
    value_sum_sq: float,
    value_min: float,
    value_max: float,
) -> dict:
    if value_count <= 0:
        raise ValueError('Cannot finalize empty CWT statistics')
    mean = float(value_sum / value_count)
    variance = max(float(value_sum_sq / value_count - mean ** 2), 0.0)
    return {
        'cwt_min': float(value_min),
        'cwt_max': float(value_max),
        'cwt_mean': mean,
        'cwt_std': float(np.sqrt(variance)),
    }


def _scalar(data, name: str):
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f'{name} must be a scalar, got shape {value.shape}')
    return value.item()


def validate_existing_cache(cache_path: Path, bearing: str) -> dict:
    required = {
        'cwt_log_power', 'y_state', 'file_index', 'bearing_id', 'fpt_index',
        'frozen_irrms_threshold', 'label_version', 'cache_kind', 'cache_version',
        'cwt_wavelet', 'cwt_scales', 'cwt_frequencies_hz',
        'nyquist_frequency_hz', 'image_shape', 'storage_dtype',
        'raw_sha256', 'label_sha256', 'cwt_min', 'cwt_max', 'cwt_mean', 'cwt_std',
    }
    with np.load(cache_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f'{bearing}: existing cache missing arrays: {sorted(missing)}')

        cwt = data['cwt_log_power']
        y_state = data['y_state']
        file_index = data['file_index']
        bearing_id = data['bearing_id'].astype(str)
        n_files = int(cwt.shape[0]) if cwt.ndim == 4 else 0
        expected_shape = (n_files, 2, *IMAGE_SHAPE)
        if cwt.shape != expected_shape:
            raise ValueError(
                f'{bearing}: existing cache shape {cwt.shape} != {expected_shape}'
            )
        if cwt.dtype != np.dtype(CACHE_DTYPE):
            raise ValueError(f'{bearing}: existing cache dtype mismatch: {cwt.dtype}')
        if bearing_id.shape != (n_files,) or not np.all(bearing_id == bearing):
            raise ValueError(f'{bearing}: existing cache bearing_id mismatch')

        fpt_index = int(_scalar(data, 'fpt_index'))
        frozen_threshold = float(_scalar(data, 'frozen_irrms_threshold'))
        label_version = str(_scalar(data, 'label_version'))
        validate_label_alignment(
            y_state=y_state,
            file_index=file_index,
            bearing_id=bearing_id,
            fpt_index=fpt_index,
            frozen_threshold=frozen_threshold,
            label_version=label_version,
            expected_bearing=bearing,
        )
        current_labels = load_frozen_labels(bearing)
        if not np.array_equal(y_state, current_labels['y_state']):
            raise ValueError(f'{bearing}: existing cache y_state differs from frozen labels')
        if not np.array_equal(file_index, current_labels['file_index']):
            raise ValueError(f'{bearing}: existing cache file_index differs from frozen labels')
        if fpt_index != int(current_labels['fpt_index']):
            raise ValueError(f'{bearing}: existing cache FPT differs from frozen labels')

        scalar_expectations = {
            'cache_kind': CACHE_KIND,
            'cache_version': CACHE_VERSION,
            'cwt_wavelet': CWT_WAVELET,
            'storage_dtype': str(np.dtype(CACHE_DTYPE)),
        }
        for name, expected in scalar_expectations.items():
            actual = str(_scalar(data, name))
            if actual != expected:
                raise ValueError(
                    f'{bearing}: existing cache {name} mismatch: {actual} != {expected}'
                )
        if not np.isclose(
            float(_scalar(data, 'nyquist_frequency_hz')),
            NYQUIST_FREQUENCY_HZ,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f'{bearing}: existing cache Nyquist mismatch')
        if not np.array_equal(data['image_shape'], np.asarray(IMAGE_SHAPE, dtype=np.int32)):
            raise ValueError(f'{bearing}: existing cache image_shape mismatch')
        if not np.array_equal(data['cwt_scales'], CWT_SCALES.astype(np.float32)):
            raise ValueError(f'{bearing}: existing cache scales mismatch')
        if not np.array_equal(
            data['cwt_frequencies_hz'], CWT_FREQUENCIES_HZ.astype(np.float32)
        ):
            raise ValueError(f'{bearing}: existing cache frequencies mismatch')

        current_raw_sha256 = sha256_file(raw_path(bearing))
        current_label_sha256 = sha256_file(label_path(bearing))
        if str(_scalar(data, 'raw_sha256')) != current_raw_sha256:
            raise ValueError(f'{bearing}: existing cache raw_sha256 mismatch')
        if str(_scalar(data, 'label_sha256')) != current_label_sha256:
            raise ValueError(f'{bearing}: existing cache label_sha256 mismatch')

        probe_indices = sorted({0, n_files // 2, n_files - 1})
        for index in probe_indices:
            if not np.isfinite(cwt[index]).all():
                raise ValueError(f'{bearing}: non-finite existing cache at index {index}')

        result = {
            'bearing_id': bearing,
            'n_files': n_files,
            'shape': list(cwt.shape),
            'dtype': str(cwt.dtype),
            'fpt_index': fpt_index,
            'raw_sha256': current_raw_sha256,
            'label_sha256': current_label_sha256,
            'cache_sha256': sha256_file(cache_path),
            'cache_path': str(cache_path),
            'status': 'existing_validated',
        }
        for name in ('cwt_min', 'cwt_max', 'cwt_mean', 'cwt_std'):
            result[name] = float(_scalar(data, name))
    return result


print('Cache validation helpers OK')"""
    ),
    code(
        """# 8) Build and atomically save one bearing cache
def build_bearing_cache(bearing: str) -> dict:
    cache_path = OUT_DIR / f'{bearing}_cwt_unscaled.npz'
    temp_path = OUT_DIR / f'{bearing}_cwt_unscaled.tmp.npz'
    if cache_path.exists() and not OVERWRITE:
        print('VALIDATE existing cache:', cache_path)
        return validate_existing_cache(cache_path, bearing)

    records = load_raw_records(bearing)
    labels = load_frozen_labels(bearing)
    n_files = len(records)
    if n_files != len(labels['y_state']):
        raise ValueError(
            f'{bearing}: raw/label length mismatch {n_files} != {len(labels["y_state"])}'
        )

    cwt_cache = np.empty(
        (n_files, 2, *IMAGE_SHAPE),
        dtype=CACHE_DTYPE,
    )
    value_count = 0
    value_sum = 0.0
    value_sum_sq = 0.0
    value_min = np.inf
    value_max = -np.inf
    for index in range(n_files):
        image = cwt_two_channel_record(records[index]).astype(CACHE_DTYPE)
        if not np.isfinite(image).all():
            raise RuntimeError(
                f'{bearing}: non-finite CWT image after {CACHE_DTYPE} cast at {index}'
            )
        cwt_cache[index] = image
        values = image.astype(np.float64)
        value_count += int(values.size)
        value_sum += float(values.sum(dtype=np.float64))
        value_sum_sq += float(np.square(values).sum(dtype=np.float64))
        value_min = min(value_min, float(values.min()))
        value_max = max(value_max, float(values.max()))
        if index % 100 == 0 or index + 1 == n_files:
            print(f'{bearing}: {index + 1}/{n_files}')

    expected_shape = (n_files, 2, *IMAGE_SHAPE)
    if cwt_cache.shape != expected_shape:
        raise RuntimeError(
            f'{bearing}: unexpected cache shape {cwt_cache.shape}, expected {expected_shape}'
        )
    stats = finalize_streaming_stats(
        value_count, value_sum, value_sum_sq, value_min, value_max
    )
    raw_sha256 = sha256_file(raw_path(bearing))
    label_sha256 = sha256_file(label_path(bearing))

    np.savez_compressed(
        temp_path,
        cwt_log_power=cwt_cache,
        y_state=labels['y_state'],
        file_index=labels['file_index'],
        bearing_id=labels['bearing_id'],
        fpt_index=labels['fpt_index'],
        frozen_irrms_threshold=labels['frozen_irrms_threshold'],
        label_version=labels['label_version'],
        cache_kind=np.array(CACHE_KIND),
        cache_version=np.array(CACHE_VERSION),
        cwt_wavelet=np.array(CWT_WAVELET),
        cwt_scales=CWT_SCALES.astype(np.float32),
        cwt_frequencies_hz=CWT_FREQUENCIES_HZ.astype(np.float32),
        nyquist_frequency_hz=np.float64(NYQUIST_FREQUENCY_HZ),
        image_shape=np.asarray(IMAGE_SHAPE, dtype=np.int32),
        storage_dtype=np.array(str(np.dtype(CACHE_DTYPE))),
        raw_sha256=np.array(raw_sha256),
        label_sha256=np.array(label_sha256),
        cwt_min=np.float64(stats['cwt_min']),
        cwt_max=np.float64(stats['cwt_max']),
        cwt_mean=np.float64(stats['cwt_mean']),
        cwt_std=np.float64(stats['cwt_std']),
    )
    temp_path.replace(cache_path)

    result = {
        'bearing_id': bearing,
        'n_files': int(n_files),
        'shape': list(cwt_cache.shape),
        'dtype': str(cwt_cache.dtype),
        'fpt_index': int(labels['fpt_index']),
        'raw_sha256': raw_sha256,
        'label_sha256': label_sha256,
        'cache_sha256': sha256_file(cache_path),
        'cache_path': str(cache_path),
        'status': 'created',
    }
    result.update(stats)
    del records, labels, cwt_cache
    gc.collect()
    return result


print('Bearing cache builder OK')"""
    ),
    code(
        """# 9) Reproducible visual audit from the saved float16 cache
def visualize_bearing_cache(cache_path: Path, bearing: str) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        cwt = data['cwt_log_power']
        n_files = int(cwt.shape[0])
        fpt_index = int(np.asarray(data['fpt_index']).item())
        selected_indices = [0, fpt_index, n_files - 1]
        examples = cwt[selected_indices].astype(np.float32)

    vmin, vmax = np.percentile(examples, [1.0, 99.0])
    if not np.isfinite([vmin, vmax]).all():
        raise RuntimeError(f'{bearing}: non-finite visualization percentiles')
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    channel_names = ['Horizontal', 'Vertical']
    state_names = ['Start', 'Frozen FPT', 'End']
    frequency_rows = np.array([0, 32, 64, 96, IMAGE_SHAPE[0] - 1], dtype=int)
    frequency_labels = [f'{CWT_FREQUENCIES_HZ[row]:.0f}' for row in frequency_rows]
    x_ticks = [0, (IMAGE_SHAPE[1] - 1) / 2, IMAGE_SHAPE[1] - 1]
    x_labels = [
        '0.00',
        f'{ACQUISITION_DURATION_SECONDS / 2:.2f}',
        f'{ACQUISITION_DURATION_SECONDS:.2f}',
    ]
    plotted = None
    for row in range(2):
        for column in range(3):
            axis = axes[row, column]
            plotted = axis.imshow(
                examples[column, row],
                aspect='auto',
                origin='upper',
                cmap='viridis',
                vmin=float(vmin),
                vmax=float(vmax),
            )
            axis.set_title(
                f'{state_names[column]} | index={selected_indices[column]}'
            )
            axis.set_xticks(x_ticks, x_labels)
            axis.set_xlabel('Time within acquisition (s)')
            axis.set_yticks(frequency_rows, frequency_labels)
            if column == 0:
                axis.set_ylabel(f'{channel_names[row]}\\nPseudo-frequency (Hz)')
    fig.suptitle(
        f'{bearing}: unscaled float16 CWT log-power | frozen FPT={fpt_index}',
        fontsize=14,
    )
    fig.colorbar(plotted, ax=axes.ravel().tolist(), label='log2 power')
    figure_path = VISUALIZATION_DIR / f'{bearing}_cwt_examples.png'
    fig.savefig(
        figure_path,
        dpi=150,
        bbox_inches='tight',
        metadata={'Software': 'Task 2 CWT cache audit'},
    )
    plt.close(fig)
    display(Image(filename=str(figure_path)))
    return {
        'visualization_path': str(figure_path),
        'visualization_sha256': sha256_file(figure_path),
        'visualized_indices': selected_indices,
        'visualization_vmin_p01': float(vmin),
        'visualization_vmax_p99': float(vmax),
    }


print('CWT visualization helper OK')"""
    ),
    code(
        """# 10) Run all six learning bearings and save manifest
manifest_rows = []
for bearing in BEARINGS:
    print('\\n' + '=' * 80)
    print('CWT CACHE:', bearing)
    cache_result = build_bearing_cache(bearing)
    cache_result.update(
        visualize_bearing_cache(Path(cache_result['cache_path']), bearing)
    )
    manifest_rows.append(cache_result)

manifest = {
    'cache_version': CACHE_VERSION,
    'cache_kind': CACHE_KIND,
    'bearings': BEARINGS,
    'raw_data_points': DATA_POINTS,
    'sampling_frequency_hz': SAMPLING_FREQ,
    'acquisition_duration_seconds': ACQUISITION_DURATION_SECONDS,
    'wavelet': CWT_WAVELET,
    'scales': CWT_SCALES.tolist(),
    'cwt_frequencies_hz': CWT_FREQUENCIES_HZ.tolist(),
    'nyquist_frequency_hz': NYQUIST_FREQUENCY_HZ,
    'n_scales_above_nyquist': N_SCALES_ABOVE_NYQUIST,
    'image_shape': list(IMAGE_SHAPE),
    'log_power_epsilon': LOG_POWER_EPS,
    'storage_dtype': str(np.dtype(CACHE_DTYPE)),
    'scaling_applied': False,
    'scaling_policy': (
        'unscaled log-power cache; cast to float32 and fit scaler only on '
        'outer-training bearings inside each fold'
    ),
    'official_test_policy': 'closed; not used in this notebook',
    'visualization_policy': (
        'display-only percentile normalization on indices [0, FPT, N-1]; '
        'never used by cache generation, scaling, or modeling'
    ),
    'frozen_irrms_threshold': FROZEN_IRRMS_THRESHOLD,
    'label_version': LABEL_VERSION,
    'entries': manifest_rows,
}
manifest_path = OUT_DIR / 'cwt_cache_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

display(pd.DataFrame(manifest_rows))
print('Saved:', manifest_path)
print('Task 2 complete: unscaled cache only. Do not fit a scaler here.')"""
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

output = Path(__file__).resolve().parents[2] / "notebooks" / "02_prepare_cwt_cache_kaggle.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
