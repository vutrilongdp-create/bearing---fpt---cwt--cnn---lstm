"""Generate the fast Kaggle notebook for RMS/EWMA/FPT/HI analysis only."""

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
        """# PRONOSTIA/FEMTO — V-HI2: fast RMS → EWMA → FPT → HI audit

Notebook này chỉ đánh giá nhãn HI, **không tạo ảnh CWT**. FPT phải thỏa:

1. RMS-EWMA vượt ngưỡng 3-sigma tại 5 mẫu liên tiếp.
2. Trong cửa sổ 30 mẫu từ ứng viên FPT, ít nhất 20 mẫu vượt ngưỡng.
3. Mẫu cuối cửa sổ vẫn vượt ngưỡng.

Sau khi nhãn ổn định, CWT sẽ được ghép lại trong notebook riêng."""
    ),
    code(
        """# 1) Imports
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})
print('Imports OK — HI-only mode')"""
    ),
    code(
        """# 2) Paths and frozen V-HI2 configuration
MAIN_DIR = Path('/kaggle/input/datasets/longvu274/train-dataset/')
OUT_DIR = Path('/kaggle/working/hi_fpt_vhi2/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_BEARINGS = ['bearing1_1', 'bearing1_2', 'bearing2_1', 'bearing2_2', 'bearing3_1']
VAL_BEARINGS = ['bearing3_2']

DATA_POINTS = 2560
SAMPLE_INTERVAL_SECONDS = 10.0
HEALTHY_FILES = 200
EWMA_ALPHA = 0.08
HI_SIGMA_SPAN = 6.0
FPT_CONSECUTIVE = 5
FPT_CONFIRM_WINDOW = 30
FPT_CONFIRM_REQUIRED = 20
EPS = 1e-8

RUN_FULL_PIPELINE = True

print('MAIN_DIR:', MAIN_DIR)
print('OUT_DIR :', OUT_DIR)
print('Train:', TRAIN_BEARINGS)
print('Validation:', VAL_BEARINGS)"""
    ),
    code(
        """# 3) Load RMS directly without retaining the full vibration array
def bearing_path(bearing_name: str) -> Path:
    return MAIN_DIR / f'{bearing_name}.pkz'


def load_bearing_df(bearing_name: str) -> pd.DataFrame:
    path = bearing_path(bearing_name)
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Check MAIN_DIR.')
    with open(path, 'rb') as f:
        df = pkl.load(f)
    required = {'horiz accel', 'vert accel'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    if len(df) % DATA_POINTS != 0:
        raise ValueError(f'{path}: {len(df)} rows is not divisible by {DATA_POINTS}')
    return df


def rms_series_from_df(df: pd.DataFrame) -> np.ndarray:
    n_files = len(df) // DATA_POINTS
    horiz = df['horiz accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    vert = df['vert accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    # RMS magnitude of both acceleration channels, equivalent to the CWT notebook.
    rms_squared = np.mean(horiz * horiz + vert * vert, axis=1)
    rms = np.sqrt(rms_squared).astype(np.float32)
    if not np.isfinite(rms).all():
        raise ValueError('Non-finite RMS values found')
    return rms


def load_rms_series(bearing_name: str) -> np.ndarray:
    return rms_series_from_df(load_bearing_df(bearing_name))


print('RMS loader OK')"""
    ),
    code(
        """# 4) Causal EWMA and confirmed FPT (5 consecutive + 20/30)
def ewma_causal(x: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError('x must be a non-empty one-dimensional series')
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def find_confirmed_fpt(
    values: np.ndarray,
    threshold: float,
    start_index: int,
    consecutive: int = FPT_CONSECUTIVE,
    confirm_window: int = FPT_CONFIRM_WINDOW,
    confirm_required: int = FPT_CONFIRM_REQUIRED,
):
    values = np.asarray(values)
    consecutive = int(consecutive)
    confirm_window = int(confirm_window)
    confirm_required = int(confirm_required)
    if consecutive < 1:
        raise ValueError('consecutive must be at least 1')
    if confirm_window < consecutive:
        raise ValueError('confirm_window must be at least consecutive')
    if not 1 <= confirm_required <= confirm_window:
        raise ValueError('confirm_required must be in [1, confirm_window]')

    above = values > threshold
    first = max(0, int(start_index))
    last_start = len(values) - confirm_window
    for i in range(first, last_start + 1):
        first_run_ok = np.all(above[i:i + consecutive])
        confirmation_ok = np.count_nonzero(above[i:i + confirm_window]) >= confirm_required
        window_ends_above = bool(above[i + confirm_window - 1])
        if first_run_ok and confirmation_ok and window_ends_above:
            return i
    return None


print('FPT detector OK')"""
    ),
    code(
        """# 5) Build current baseline HI (z-score version is intentionally deferred)
def build_hi_from_rms(rms_values: np.ndarray) -> dict:
    rms_values = np.asarray(rms_values, dtype=np.float32)
    if rms_values.ndim != 1 or len(rms_values) < FPT_CONFIRM_WINDOW:
        raise ValueError(f'Need at least {FPT_CONFIRM_WINDOW} RMS samples')

    n_healthy = min(HEALTHY_FILES, max(5, len(rms_values) // 3))
    baseline = rms_values[:n_healthy]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline))
    sigma_safe = max(sigma, EPS)
    threshold = mu + 3.0 * sigma_safe
    smooth = ewma_causal(rms_values)

    fpt_idx = find_confirmed_fpt(smooth, threshold, n_healthy)
    fpt_found = fpt_idx is not None

    damage = np.maximum(0.0, smooth - threshold) / (HI_SIGMA_SPAN * sigma_safe)
    hi = np.clip(damage, 0.0, 1.0).astype(np.float32)
    if fpt_found:
        hi[:fpt_idx] = 0.0
    else:
        hi[:] = 0.0

    return {
        'hi': hi,
        'rms': rms_values,
        'rms_ewma': smooth,
        'baseline_mu': mu,
        'baseline_sigma': sigma,
        'threshold': float(threshold),
        'healthy_files_used': int(n_healthy),
        'fpt_index': int(fpt_idx) if fpt_found else None,
        'fpt_found': bool(fpt_found),
    }


print('HI builder OK')"""
    ),
    code(
        """# 6) Process one bearing and save lightweight arrays/plot
def process_bearing(bearing: str, split: str) -> dict:
    rms = load_rms_series(bearing)
    info = build_hi_from_rms(rms)
    n = len(rms)
    idx = np.arange(n, dtype=np.int32)

    np.savez_compressed(
        OUT_DIR / f'{bearing}_hi_vhi2.npz',
        y_hi=info['hi'],
        y_rms=info['rms'],
        y_rms_ewma=info['rms_ewma'],
        file_index=idx,
        elapsed_seconds=idx.astype(np.float32) * SAMPLE_INTERVAL_SECONDS,
        bearing_id=np.full(n, bearing),
        split=np.full(n, split),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(idx, info['rms'], lw=0.8, alpha=0.5, label='RMS')
    axes[0].plot(idx, info['rms_ewma'], lw=1.6, label='RMS EWMA')
    axes[0].axhline(info['threshold'], color='red', ls='--', label='3-sigma threshold')
    if info['fpt_found']:
        axes[0].axvline(
            info['fpt_index'], color='purple', ls=':',
            label=f"FPT={info['fpt_index']} (5 + 20/30)",
        )
    axes[0].set_title(f'RMS + confirmed FPT — {bearing}')
    axes[0].set_xlabel('File index')
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(idx, info['hi'], color='#1D9E75', lw=1.7)
    axes[1].fill_between(idx, 0, info['hi'], color='#1D9E75', alpha=0.12)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title('HI label — current 3-sigma span baseline')
    axes[1].set_xlabel('File index')
    axes[1].set_ylabel('HI')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    plot_path = OUT_DIR / f'{bearing}_hi_vhi2.png'
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    hi = info['hi']
    meta = {
        'bearing': bearing,
        'split': split,
        'n_files': int(n),
        'fpt_index': info['fpt_index'],
        'fpt_found': info['fpt_found'],
        'fpt_ratio': float(info['fpt_index'] / max(n - 1, 1)) if info['fpt_found'] else None,
        'baseline_mu': info['baseline_mu'],
        'baseline_sigma': info['baseline_sigma'],
        'threshold': info['threshold'],
        'hi_zero_ratio': float(np.mean(hi == 0.0)),
        'hi_middle_ratio': float(np.mean((hi > 0.1) & (hi < 0.9))),
        'hi_high_ratio': float(np.mean(hi >= 0.9)),
        'hi_max': float(np.max(hi)),
        'plot': str(plot_path),
    }
    print(json.dumps(meta, indent=2))
    return meta


print('Bearing processor OK')"""
    ),
    code(
        """# 7) Run all 5 train bearings, then validation once
all_meta = []
if RUN_FULL_PIPELINE:
    for bearing in TRAIN_BEARINGS:
        print('\\n' + '=' * 72)
        print('TRAIN:', bearing)
        all_meta.append(process_bearing(bearing, 'train'))

    # Do not tune parameters after inspecting this validation bearing.
    for bearing in VAL_BEARINGS:
        print('\\n' + '=' * 72)
        print('VALIDATION (evaluate once after freeze):', bearing)
        all_meta.append(process_bearing(bearing, 'validation'))
else:
    print('RUN_FULL_PIPELINE=False — no bearing processed.')"""
    ),
    code(
        """# 8) Machine-readable comparison table and frozen configuration
if RUN_FULL_PIPELINE:
    summary_df = pd.DataFrame(all_meta)
    summary_path = OUT_DIR / 'hi_fpt_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    config = {
        'version': 'V-HI2-hi-only',
        'train_bearings': TRAIN_BEARINGS,
        'validation_bearings': VAL_BEARINGS,
        'healthy_files': HEALTHY_FILES,
        'ewma_alpha': EWMA_ALPHA,
        'hi_sigma_span': HI_SIGMA_SPAN,
        'fpt_consecutive': FPT_CONSECUTIVE,
        'fpt_confirm_window': FPT_CONFIRM_WINDOW,
        'fpt_confirm_required': FPT_CONFIRM_REQUIRED,
        'cwt_generated': False,
    }
    (OUT_DIR / 'hi_fpt_config.json').write_text(
        json.dumps(config, indent=2), encoding='utf-8'
    )
    display(summary_df)
    print('Saved:', summary_path)
    print('Done — inspect FPT/HI before restoring CWT.')"""
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

output = Path(__file__).resolve().parents[2] / "notebooks" / "hi_fpt_prepare_kaggle_vhi2.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
