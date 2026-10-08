"""Create a standalone Kaggle notebook for CWT -> HI CNN data preparation."""

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
        """# PRONOSTIA / FEMTO — Leakage-safe CWT + 3-Sigma HI dataset for CNN

Notebook này viết lại sạch phần chuẩn bị dữ liệu cho CNN:

- **x**: ảnh CWT log-power từ tín hiệu rung động, shape `[N, 2, 128, 128]`.
- **y**: Health Indicator `HI ∈ [0,1]`, sinh từ chuỗi RMS bằng ngưỡng 3-sigma/FPT.
- **Split**: 5 bearing train, 1 bearing validation/test nội bộ.
- **Chống leakage**: scaler CWT chỉ fit trên 5 train bearings; validation chỉ transform bằng scaler đó. HI không dùng min/max tương lai của bearing.

Mục tiêu của CNN sau notebook này:

```text
CWT image -> HI
```

Lưu ý: HI là nhãn suy thoái, chưa phải RUL. Nếu muốn RUL, cần thêm bước mapping `HI -> RUL` hoặc mô hình sequence riêng."""
    ),
    code(
        """# ─── 1) Imports ─────────────────────────────────────────────────────────────
import os
import json
import pickle as pkl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize

from scipy.signal import butter, filtfilt, hilbert

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

print("Imports OK")"""
    ),
    code(
        """# ─── 2) Paths & configuration ─────────────────────────────────────────────
# Kaggle input folder containing bearing1_1.pkz, bearing1_2.pkz, ...
MAIN_DIR = Path('/kaggle/input/datasets/longvu274/train-dataset/')
OUT_DIR = Path('/kaggle/working/cwt_hi_dataset/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 train bearings + 1 validation/test nội bộ
TRAIN_BEARINGS = ['bearing1_1', 'bearing1_2', 'bearing2_1', 'bearing2_2', 'bearing3_1']
VAL_BEARINGS = ['bearing3_2']
ALL_BEARINGS = TRAIN_BEARINGS + VAL_BEARINGS

# Signal / CWT parameters
SAMPLING_FREQ = 25600
DATA_POINTS = 2560
N_SCALES = 128
IMG_SIZE = 128
WAVELET = 'morl'
SAMPLE_INTERVAL_SECONDS = 10.0

# HI parameters
HEALTHY_FILES = 200          # baseline đầu đời dùng cho 3-sigma
EWMA_ALPHA = 0.08            # causal smoothing
HI_SIGMA_SPAN = 6.0          # HI=1 khi RMS_EWMA vượt threshold khoảng 6 sigma
FPT_CONSECUTIVE = 5          # xác nhận FPT khi vượt ngưỡng đủ 5 mẫu liên tiếp
EPS = 1e-8

# Signal mode:
# - 'raw': dùng trực tiếp 2 kênh acceleration. Ít giả định, không cần chọn band.
# - 'envelope_fixed_band': fixed bandpass + Hilbert envelope; vẫn leakage-safe vì band cố định.
SIGNAL_MODE = 'raw'
FIXED_BAND_HZ = (500.0, 10000.0)

# Output dtype. float16 giảm dung lượng rất nhiều và vẫn hợp lý cho ảnh CNN.
X_DTYPE = np.float16

# Chạy thử trước khi chạy toàn bộ dataset.
# Lần đầu: giữ RUN_FULL_PIPELINE=False và Run All để kiểm tra 1 bearing.
# Sau khi smoke test đạt: đổi thành True rồi Run All lần nữa.
SMOKE_BEARING = 'bearing1_1'
SMOKE_TEST_FILES = 50
RUN_FULL_PIPELINE = False

print("Config OK")
print("MAIN_DIR:", MAIN_DIR)
print("OUT_DIR :", OUT_DIR)
print(f"Smoke test: {SMOKE_BEARING}, first {SMOKE_TEST_FILES} files")
print("RUN_FULL_PIPELINE:", RUN_FULL_PIPELINE)"""
    ),
    code(
        """# ─── 3) IO helpers ───────────────────────────────────────────────────────
def bearing_path(bearing_name: str) -> Path:
    return MAIN_DIR / f'{bearing_name}.pkz'


def load_bearing_df(bearing_name: str) -> pd.DataFrame:
    path = bearing_path(bearing_name)
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Check MAIN_DIR or Kaggle dataset mount.')
    with open(path, 'rb') as f:
        df = pkl.load(f)
    required = {'horiz accel', 'vert accel'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing required columns: {sorted(missing)}')
    if len(df) % DATA_POINTS != 0:
        raise ValueError(f'{path} has {len(df)} rows, not divisible by {DATA_POINTS}')
    return df


def bearing_signals(df: pd.DataFrame) -> np.ndarray:
    \"\"\"Return raw signals as [N_files, 2, 2560].\"\"\"
    n_files = len(df) // DATA_POINTS
    horiz = df['horiz accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    vert = df['vert accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    values = np.stack([horiz, vert], axis=1)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite vibration values found')
    return values


def load_bearing_signals(bearing_name: str) -> np.ndarray:
    return bearing_signals(load_bearing_df(bearing_name))


print("IO helpers OK")"""
    ),
    code(
        """# ─── 4) Signal processing ────────────────────────────────────────────────
def apply_fixed_bandpass(signal_1d: np.ndarray, band_hz=FIXED_BAND_HZ) -> np.ndarray:
    low, high = band_hz
    nyquist = SAMPLING_FREQ / 2.0
    if not (0.0 < low < high < nyquist):
        raise ValueError(f'Invalid band {band_hz}; Nyquist={nyquist}')
    b, a = butter(4, [low / nyquist, high / nyquist], btype='bandpass')
    return filtfilt(b, a, signal_1d).astype(np.float32)


def signal_for_learning(two_channel_record: np.ndarray) -> np.ndarray:
    \"\"\"Return [2, 2560] signal used for both CWT and RMS.

    Không dùng kurtogram/band selection theo từng bearing để tránh nhìn thấy tương lai.
    Nếu cần envelope, dùng fixed band toàn cục, không fit trên validation.
    \"\"\"
    values = np.asarray(two_channel_record, dtype=np.float32)
    if values.shape != (2, DATA_POINTS):
        raise ValueError(f'Expected [2,{DATA_POINTS}], got {values.shape}')
    if SIGNAL_MODE == 'raw':
        return values
    if SIGNAL_MODE == 'envelope_fixed_band':
        env = []
        for ch in range(2):
            filtered = apply_fixed_bandpass(values[ch])
            env.append(np.abs(hilbert(filtered)).astype(np.float32))
        return np.stack(env, axis=0)
    raise ValueError(f'Unsupported SIGNAL_MODE={SIGNAL_MODE!r}')


def rms_for_record(two_channel_record: np.ndarray) -> float:
    \"\"\"RMS đại diện của một file, gộp 2 kênh bằng RMS magnitude.\"\"\"
    values = signal_for_learning(two_channel_record)
    magnitude = np.sqrt(values[0] ** 2 + values[1] ** 2)
    return float(np.sqrt(np.mean(magnitude ** 2)))


def rms_series(records: np.ndarray) -> np.ndarray:
    return np.array([rms_for_record(record) for record in records], dtype=np.float32)


print("Signal processing OK")"""
    ),
    code(
        """# ─── 5) CWT image generation ─────────────────────────────────────────────
def compute_cwt_channel(signal_1d: np.ndarray) -> np.ndarray:
    scales = np.geomspace(1, 512, N_SCALES)
    coef, _ = pywt.cwt(
        signal_1d,
        scales,
        WAVELET,
        sampling_period=1.0 / SAMPLING_FREQ,
    )
    power = np.abs(coef) ** 2
    log_power = np.log2(power + 1e-3)
    image = resize(
        log_power,
        (IMG_SIZE, IMG_SIZE),
        anti_aliasing=True,
        mode='reflect',
        preserve_range=True,
    )
    return image.astype(np.float32)


def compute_cwt_record(two_channel_record: np.ndarray) -> np.ndarray:
    values = signal_for_learning(two_channel_record)
    return np.stack(
        [compute_cwt_channel(values[0]), compute_cwt_channel(values[1])],
        axis=0,
    ).astype(np.float32)


def update_cwt_minmax(records: np.ndarray, current_min=np.inf, current_max=-np.inf):
    \"\"\"First pass: fit global CWT min/max from train bearings only.\"\"\"
    g_min = float(current_min)
    g_max = float(current_max)
    for i, record in enumerate(records):
        image = compute_cwt_record(record)
        g_min = min(g_min, float(image.min()))
        g_max = max(g_max, float(image.max()))
        if (i + 1) % 250 == 0 or (i + 1) == len(records):
            print(f'    scaler pass {i + 1}/{len(records)}', end='\\r')
    print()
    return g_min, g_max


def scale_cwt(image: np.ndarray, scaler: dict) -> np.ndarray:
    scaled = (image - scaler['min']) / (scaler['max'] - scaler['min'] + EPS)
    return np.clip(scaled, 0.0, 1.0).astype(X_DTYPE)


print("CWT functions OK")"""
    ),
    code(
        """# ─── 6) 3-sigma FPT and leakage-safe HI labels ───────────────────────────
def ewma_causal(x: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def find_persistent_fpt(
    values: np.ndarray,
    threshold: float,
    start_index: int,
    consecutive: int = FPT_CONSECUTIVE,
):
    \"\"\"Return the first index of a persistent threshold exceedance, or None.\"\"\"
    values = np.asarray(values)
    consecutive = int(consecutive)
    if consecutive < 1:
        raise ValueError('consecutive must be at least 1')

    first = max(0, int(start_index))
    last_start = len(values) - consecutive
    for i in range(first, last_start + 1):
        if np.all(values[i:i + consecutive] > threshold):
            return i
    return None


def build_hi_from_rms(
    rms_values: np.ndarray,
    healthy_files: int = HEALTHY_FILES,
    alpha: float = EWMA_ALPHA,
    sigma_span: float = HI_SIGMA_SPAN,
) -> dict:
    \"\"\"Create HI labels from RMS without using future min/max.

    Baseline and threshold use only the early healthy segment of the same bearing.
    HI uses a fixed sigma span, not the bearing's future maximum.
    \"\"\"
    rms_values = np.asarray(rms_values, dtype=np.float32)
    if rms_values.ndim != 1 or len(rms_values) < 5:
        raise ValueError('rms_values must be a one-dimensional series with at least 5 points')

    n_healthy = min(int(healthy_files), max(5, len(rms_values) // 3))
    baseline = rms_values[:n_healthy]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline))
    sigma_safe = max(sigma, EPS)
    threshold = mu + 3.0 * sigma_safe

    smooth = ewma_causal(rms_values, alpha=alpha)

    fpt_idx = find_persistent_fpt(
        smooth,
        threshold=threshold,
        start_index=n_healthy,
        consecutive=FPT_CONSECUTIVE,
    )
    fpt_found = fpt_idx is not None

    damage = np.maximum(0.0, smooth - threshold) / (sigma_span * sigma_safe)
    hi = np.clip(damage, 0.0, 1.0).astype(np.float32)
    if fpt_found:
        hi[:fpt_idx] = 0.0
    else:
        hi[:] = 0.0

    return {
        'hi': hi,
        'rms': rms_values.astype(np.float32),
        'rms_ewma': smooth.astype(np.float32),
        'baseline_mu': mu,
        'baseline_sigma': sigma,
        'threshold': float(threshold),
        'healthy_files_used': int(n_healthy),
        'fpt_index': int(fpt_idx) if fpt_found else None,
        'fpt_found': bool(fpt_found),
        'fpt_consecutive': int(FPT_CONSECUTIVE),
    }


print("HI functions OK")"""
    ),
    code(
        """# ─── 7) Smoke test: one bearing before the full pipeline ─────────────────
def run_one_bearing_smoke_test(
    bearing: str = SMOKE_BEARING,
    max_files: int = SMOKE_TEST_FILES,
) -> dict:
    print('\\n' + '=' * 72)
    print(f'SMOKE TEST: {bearing}, first {max_files} files')
    print('=' * 72)

    all_records = load_bearing_signals(bearing)
    n_test = min(int(max_files), len(all_records))
    if n_test < 5:
        raise ValueError('SMOKE_TEST_FILES must be at least 5')
    records = all_records[:n_test]

    # Smoke scaler is temporary and used only to validate the CWT code path.
    # It is NOT reused by the full pipeline.
    smoke_min, smoke_max = update_cwt_minmax(records)
    smoke_scaler = {'min': smoke_min, 'max': smoke_max}

    rms = rms_series(records)
    # For a short smoke subset, reduce the baseline while keeping it causal.
    smoke_healthy = min(HEALTHY_FILES, max(5, n_test // 3))
    hi_info = build_hi_from_rms(rms, healthy_files=smoke_healthy)

    sample_indices = sorted(set([0, n_test // 2, n_test - 1]))
    sample_cwt = np.stack(
        [scale_cwt(compute_cwt_record(records[i]), smoke_scaler) for i in sample_indices],
        axis=0,
    )

    assert records.shape == (n_test, 2, DATA_POINTS)
    assert sample_cwt.shape == (len(sample_indices), 2, IMG_SIZE, IMG_SIZE)
    assert np.isfinite(sample_cwt).all()
    assert np.isfinite(rms).all()
    assert np.isfinite(hi_info['hi']).all()
    assert 0.0 <= float(hi_info['hi'].min())
    assert float(hi_info['hi'].max()) <= 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    idx = np.arange(n_test)
    axes[0].plot(idx, rms, lw=1.2, alpha=0.65, label='RMS')
    axes[0].plot(idx, hi_info['rms_ewma'], lw=1.8, label='RMS EWMA')
    axes[0].axhline(
        hi_info['threshold'],
        color='red',
        ls='--',
        label='3-sigma threshold',
    )
    if hi_info['fpt_found']:
        axes[0].axvline(
            hi_info['fpt_index'],
            color='purple',
            ls=':',
            label=f"FPT={hi_info['fpt_index']} (5 consecutive)",
        )
    axes[0].set_title(f'RMS smoke test — {bearing}')
    axes[0].set_xlabel('File index')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(idx, hi_info['hi'], color='#1D9E75', lw=2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('Causal HI label')
    axes[1].set_xlabel('File index')
    axes[1].set_ylabel('HI')
    axes[1].grid(alpha=0.3)

    axes[2].imshow(sample_cwt[-1, 0], aspect='auto', cmap='RdYlBu_r')
    axes[2].set_title(f'CWT sample — file {sample_indices[-1]}')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    smoke_plot = OUT_DIR / f'{bearing}_smoke_test.png'
    fig.savefig(smoke_plot, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    result = {
        'bearing': bearing,
        'n_files_tested': n_test,
        'records_shape': list(records.shape),
        'sample_cwt_shape': list(sample_cwt.shape),
        'cwt_min': float(sample_cwt.min()),
        'cwt_max': float(sample_cwt.max()),
        'rms_min': float(rms.min()),
        'rms_max': float(rms.max()),
        'hi_min': float(hi_info['hi'].min()),
        'hi_max': float(hi_info['hi'].max()),
        'fpt_index': hi_info['fpt_index'],
        'fpt_found': bool(hi_info['fpt_found']),
        'fpt_consecutive': int(hi_info['fpt_consecutive']),
        'plot': str(smoke_plot),
    }
    print('\\nSMOKE TEST PASSED')
    print(json.dumps(result, indent=2))
    return result


smoke_result = run_one_bearing_smoke_test()

if RUN_FULL_PIPELINE:
    print('\\nRUN_FULL_PIPELINE=True: continuing to all 5 train + 1 validation bearings.')
else:
    print('\\nFull pipeline is skipped.')
    print('Review the smoke plot and values above.')
    print('If they are reasonable, set RUN_FULL_PIPELINE=True in Cell 2 and Run All again.')"""
    ),
    code(
        """# ─── 8) Fit CWT scaler on TRAIN bearings only ───────────────────────────
def fit_train_cwt_scaler(train_bearings):
    g_min, g_max = np.inf, -np.inf
    details = {}
    for bearing in train_bearings:
        print(f'  Fitting CWT scaler from {bearing} ...')
        records = load_bearing_signals(bearing)
        g_min, g_max = update_cwt_minmax(records, g_min, g_max)
        details[bearing] = {
            'n_files': int(records.shape[0]),
            'partial_min': float(g_min),
            'partial_max': float(g_max),
        }
    if not np.isfinite(g_min) or not np.isfinite(g_max) or g_max <= g_min:
        raise ValueError(f'Invalid scaler min/max: {g_min}, {g_max}')
    return {'min': float(g_min), 'max': float(g_max), 'fit_bearings': list(train_bearings), 'details': details}


cwt_scaler = None
if RUN_FULL_PIPELINE:
    cwt_scaler = fit_train_cwt_scaler(TRAIN_BEARINGS)
    scaler_path = OUT_DIR / 'cwt_scaler_train_only.json'
    scaler_path.write_text(json.dumps(cwt_scaler, indent=2), encoding='utf-8')
    print('Saved scaler:', scaler_path)
    print(cwt_scaler)
else:
    print('Skipped full train-only scaler fitting.')"""
    ),
    code(
        """# ─── 9) Build per-bearing CWT-HI arrays ─────────────────────────────────
def build_one_bearing_dataset(bearing: str, split: str, cwt_scaler: dict, out_dir: Path = OUT_DIR) -> dict:
    print('\\n' + '=' * 72)
    print(f'Processing {bearing} [{split}]')
    print('=' * 72)

    records = load_bearing_signals(bearing)
    n_files = records.shape[0]
    print(f'  records: {records.shape}')

    rms = rms_series(records)
    hi_info = build_hi_from_rms(rms)
    y_hi = hi_info['hi']

    x = np.empty((n_files, 2, IMG_SIZE, IMG_SIZE), dtype=X_DTYPE)
    for i, record in enumerate(records):
        x[i] = scale_cwt(compute_cwt_record(record), cwt_scaler)
        if (i + 1) % 250 == 0 or (i + 1) == n_files:
            print(f'  CWT {i + 1}/{n_files}', end='\\r')
    print()

    file_index = np.arange(n_files, dtype=np.int32)
    elapsed_seconds = (file_index * SAMPLE_INTERVAL_SECONDS).astype(np.float32)
    bearing_ids = np.array([bearing] * n_files)
    splits = np.array([split] * n_files)

    out_path = out_dir / f'{bearing}_cwt_hi.npz'
    np.savez_compressed(
        out_path,
        x=x,
        y_hi=y_hi.astype(np.float32),
        y_rms=rms.astype(np.float32),
        y_rms_ewma=hi_info['rms_ewma'].astype(np.float32),
        file_index=file_index,
        elapsed_seconds=elapsed_seconds,
        bearing_id=bearing_ids,
        split=splits,
    )

    meta = {
        'bearing': bearing,
        'split': split,
        'n_files': int(n_files),
        'path': str(out_path),
        'x_shape': list(x.shape),
        'x_dtype': str(x.dtype),
        'y_hi_min': float(y_hi.min()),
        'y_hi_max': float(y_hi.max()),
        'y_hi_mean': float(y_hi.mean()),
        'rms_min': float(rms.min()),
        'rms_max': float(rms.max()),
        'fpt_index': hi_info['fpt_index'],
        'fpt_found': bool(hi_info['fpt_found']),
        'fpt_consecutive': int(hi_info['fpt_consecutive']),
        'healthy_files_used': int(hi_info['healthy_files_used']),
        'baseline_mu': float(hi_info['baseline_mu']),
        'baseline_sigma': float(hi_info['baseline_sigma']),
        'threshold': float(hi_info['threshold']),
    }
    print(f'  Saved: {out_path}')
    print(f\"  HI range: [{meta['y_hi_min']:.3f}, {meta['y_hi_max']:.3f}], FPT={meta['fpt_index']}/{n_files}\")
    return meta


print("Dataset builder OK")"""
    ),
    code(
        """# ─── 10) Run all 5 train + 1 validation bearing ─────────────────────────
all_meta = []

if RUN_FULL_PIPELINE:
    for bearing in TRAIN_BEARINGS:
        all_meta.append(build_one_bearing_dataset(bearing, 'train', cwt_scaler))

    for bearing in VAL_BEARINGS:
        # Important: no scaler fitting here. Validation only transforms with train scaler.
        all_meta.append(build_one_bearing_dataset(bearing, 'val', cwt_scaler))

    summary = {
        'purpose': 'CNN dataset: x=CWT image, y=HI from RMS 3-sigma',
        'signal_mode': SIGNAL_MODE,
        'fixed_band_hz': list(FIXED_BAND_HZ),
        'train_bearings': TRAIN_BEARINGS,
        'val_bearings': VAL_BEARINGS,
        'cwt_scaler': cwt_scaler,
        'hi_params': {
            'healthy_files': HEALTHY_FILES,
            'ewma_alpha': EWMA_ALPHA,
            'hi_sigma_span': HI_SIGMA_SPAN,
            'fpt_consecutive': FPT_CONSECUTIVE,
            'leakage_note': 'HI does not use future per-bearing min/max normalization.',
        },
        'artifacts': all_meta,
    }

    summary_path = OUT_DIR / 'cwt_hi_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\\nSaved summary:', summary_path)
else:
    print('Skipped full per-bearing dataset generation.')"""
    ),
    code(
        """# ─── 11) Merge train and validation sets ────────────────────────────────
def load_npz_fields(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def merge_split(split_name: str, bearings: list[str], out_dir: Path = OUT_DIR) -> Path:
    arrays = []
    for bearing in bearings:
        path = out_dir / f'{bearing}_cwt_hi.npz'
        arrays.append(load_npz_fields(path))

    merged = {}
    for key in arrays[0].keys():
        merged[key] = np.concatenate([a[key] for a in arrays], axis=0)

    out_path = out_dir / f'{split_name}_cwt_hi_dataset.npz'
    np.savez_compressed(out_path, **merged)
    print(f'{split_name}: saved {out_path}')
    print(f\"  x={merged['x'].shape} {merged['x'].dtype}\")
    print(f\"  y_hi=[{merged['y_hi'].min():.3f}, {merged['y_hi'].max():.3f}], mean={merged['y_hi'].mean():.3f}\")
    print(f\"  bearings={sorted(set(merged['bearing_id'].tolist()))}\")
    return out_path


train_dataset_path = None
val_dataset_path = None
if RUN_FULL_PIPELINE:
    train_dataset_path = merge_split('train', TRAIN_BEARINGS)
    val_dataset_path = merge_split('val', VAL_BEARINGS)
    print('Merged dataset files ready for CNN training.')
else:
    print('Skipped merged train/validation dataset generation.')"""
    ),
    code(
        """# ─── 12) Sanity plots: RMS, threshold, HI, CWT samples ──────────────────
def plot_bearing_sanity(bearing: str, out_dir: Path = OUT_DIR):
    path = out_dir / f'{bearing}_cwt_hi.npz'
    d = np.load(path, allow_pickle=True)
    x = d['x']
    y_hi = d['y_hi']
    y_rms = d['y_rms']
    y_rms_ewma = d['y_rms_ewma']
    idx = np.arange(len(y_hi))

    meta = next(item for item in all_meta if item['bearing'] == bearing)
    threshold = meta['threshold']
    fpt = meta['fpt_index']

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    axes[0].plot(idx, y_rms, lw=1.0, alpha=0.55, label='RMS')
    axes[0].plot(idx, y_rms_ewma, lw=1.5, label='RMS EWMA')
    axes[0].axhline(threshold, color='red', ls='--', lw=1.2, label='3-sigma threshold')
    if meta['fpt_found']:
        axes[0].axvline(fpt, color='purple', ls=':', lw=1.2, label=f'FPT={fpt} (5 consecutive)')
    axes[0].set_title(f'RMS + FPT — {bearing}')
    axes[0].set_xlabel('File index')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(idx, y_hi, color='#1D9E75', lw=2)
    axes[1].fill_between(idx, 0, y_hi, color='#1D9E75', alpha=0.1)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('HI label from RMS 3-sigma')
    axes[1].set_xlabel('File index')
    axes[1].set_ylabel('HI')
    axes[1].grid(alpha=0.3)

    sample_indices = [0, len(x)//2, len(x)-1]
    for s_i, sample_index in enumerate(sample_indices):
        # show horizontal channel only for compact sanity view
        axes[2].imshow(x[sample_index, 0], aspect='auto', cmap='RdYlBu_r', alpha=0.35 + 0.2*s_i)
    axes[2].set_title('CWT samples overlay: first/mid/last')
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.tight_layout()
    out_path = out_dir / f'{bearing}_sanity.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print('Saved plot:', out_path)


if RUN_FULL_PIPELINE:
    for bearing in ALL_BEARINGS:
        plot_bearing_sanity(bearing)
    print('Sanity plots complete.')
else:
    print('Skipped full sanity plots; the one-bearing smoke plot was already created.')"""
    ),
    code(
        """# ─── 13) Final artifact check ───────────────────────────────────────────
print('Output directory:', OUT_DIR)
for path in sorted(OUT_DIR.glob('*')):
    if path.is_file():
        print(f'{path.name:35s} {path.stat().st_size / 1e6:8.2f} MB')

if RUN_FULL_PIPELINE:
    train_data = np.load(train_dataset_path, allow_pickle=True)
    val_data = np.load(val_dataset_path, allow_pickle=True)

    print('\\nTrain x:', train_data['x'].shape, train_data['x'].dtype)
    print('Train y_hi:', train_data['y_hi'].shape, train_data['y_hi'].min(), train_data['y_hi'].max())
    print('Val x:', val_data['x'].shape, val_data['x'].dtype)
    print('Val y_hi:', val_data['y_hi'].shape, val_data['y_hi'].min(), val_data['y_hi'].max())

    print('\\nDone. Use train_cwt_hi_dataset.npz and val_cwt_hi_dataset.npz for CNN training.')
else:
    print('\\nSmoke-test mode complete.')
    print('Set RUN_FULL_PIPELINE=True in Cell 2, then Run All to build the full dataset.')"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[2] / "notebooks" / "cwt_hi_cnn_prepare_kaggle.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
