"""Generate the Kaggle notebook for causal-IRRMS FPT sensitivity audit."""

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
        """# Task 1 — causal IRRMS reference-FPT label audit

Notebook này tạo causal IRRMS cho sáu PRONOSTIA learning bearings, đánh giá một
tập threshold toàn cục, sau đó ghi nhãn FPT bằng ngưỡng Q99 đã freeze. Task 2/CWT
chỉ được chạy sau khi handoff Task 1 báo `ready_for_cwt=True`.

Protocol:

1. `RRMS = RMS / mean(RMS[0:100])`.
2. Causal IRRMS chỉ dùng cửa sổ trailing 30 file; warm-up `0..28` là `NaN`.
3. FPT là điểm đầu của 5 IRRMS liên tiếp vượt threshold.
4. Ứng viên bị loại nếu sau đó có 30 IRRMS hữu hạn liên tiếp `<= threshold`.

Đây là **causal IRRMS-based retrospective operational FPT reference**, không
phải FPT chuyên gia hoặc ground truth vật lý tuyệt đối."""
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
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})
print('Imports OK')"""
    ),
    code(
        """# 2) Paths and sensitivity parameters
MAIN_DIR = Path('/kaggle/input/datasets/longvu274/train-dataset/')
OUT_DIR = Path('/kaggle/working/causal_irrms_fpt_reference/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [
    'bearing1_1', 'bearing1_2',
    'bearing2_1', 'bearing2_2',
    'bearing3_1', 'bearing3_2',
]

DATA_POINTS = 2560
SAMPLING_FREQ = 25600.0
SAMPLE_INTERVAL_SECONDS = 10.0
HEALTHY_FILES = 100
IRRMS_WINDOW = 30
FPT_CONSECUTIVE = 5
RECOVERY_RUN = 30
SEARCH_START = 100
FIXED_IRRMS_THRESHOLDS = (1.05, 1.10, 1.15, 1.20)
BASELINE_QUANTILES = (0.99, 0.995)
FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
STD_DDOF = 0
THRESHOLD_OPERATOR = '>'
NEAR_END_MIN_DEGRADED_FILES = 30
EPS = 1e-8

RUN_FULL_PIPELINE = True

print('MAIN_DIR:', MAIN_DIR)
print('OUT_DIR :', OUT_DIR)
print('Bearings:', BEARINGS)"""
    ),
    code(
        """# 3) Core causal-IRRMS and FPT functions
def rms_two_channel(record: np.ndarray) -> float:
    record = np.asarray(record, dtype=np.float32)
    if record.ndim != 2 or record.shape[0] != 2:
        raise ValueError(f'Expected [2, n_points], got {record.shape}')
    value = np.sqrt(np.mean(record[0] ** 2 + record[1] ** 2))
    return float(value)


def relative_rms(
    rms: np.ndarray,
    healthy_files: int = HEALTHY_FILES,
):
    rms = np.asarray(rms, dtype=np.float32)
    if rms.ndim != 1 or healthy_files < 1 or len(rms) < healthy_files:
        raise ValueError(f'Need at least {healthy_files} RMS samples')
    rms_norm = float(np.mean(rms[:healthy_files]))
    if not np.isfinite(rms_norm) or rms_norm <= EPS:
        raise ValueError(f'Invalid healthy RMS norm: {rms_norm}')
    return (rms / rms_norm).astype(np.float32), rms_norm


def causal_irrms(
    rrms: np.ndarray,
    window: int = IRRMS_WINDOW,
) -> np.ndarray:
    rrms = np.asarray(rrms, dtype=np.float32)
    if rrms.ndim != 1 or window < 2:
        raise ValueError('rrms must be 1-D and window must be at least 2')
    if not np.isfinite(rrms).all():
        raise ValueError('rrms must contain only finite values')

    irrms = np.full(len(rrms), np.nan, dtype=np.float32)
    x = np.arange(window, dtype=np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    for t in range(window - 1, len(rrms)):
        values = rrms[t - window + 1:t + 1].astype(np.float64)
        slope = float(np.dot(x_centered, values - values.mean()) / denominator)
        intercept = float(values.mean() - slope * x.mean())
        trend = slope * x + intercept
        residual = values - trend
        residual_mean = float(residual.mean())
        residual_std = float(residual.std(ddof=STD_DDOF))
        current_residual = float(residual[-1])

        if residual_std <= EPS:
            value = float(trend[-1])
        else:
            lower = residual_mean - 3.0 * residual_std
            upper = residual_mean + 3.0 * residual_std
            if current_residual <= lower:
                value = float(values.mean())
            elif current_residual < upper:
                value = float(trend[-1])
            else:
                value = float(trend[-1] + upper)
        irrms[t] = value
    return irrms


def _first_run_start(mask: np.ndarray, run_length: int):
    mask = np.asarray(mask, dtype=bool)
    if run_length < 1:
        raise ValueError('run_length must be at least 1')
    run = 0
    for idx, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= run_length:
            return idx - run_length + 1
    return None


def _count_run_starts(mask: np.ndarray, run_length: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    count = 0
    run = 0
    counted = False
    for value in mask:
        if value:
            run += 1
            if run >= run_length and not counted:
                count += 1
                counted = True
        else:
            run = 0
            counted = False
    return count


def evaluate_irrms_fpt(
    irrms: np.ndarray,
    threshold: float,
    consecutive: int = FPT_CONSECUTIVE,
    recovery_run: int = RECOVERY_RUN,
    start_index: int = SEARCH_START,
):
    irrms = np.asarray(irrms, dtype=np.float32)
    if irrms.ndim != 1:
        raise ValueError('irrms must be one-dimensional')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    if consecutive < 1 or recovery_run < 1:
        raise ValueError('consecutive and recovery_run must be positive')

    finite = np.isfinite(irrms)
    candidate_mask = finite & (irrms > threshold)
    last_start = len(irrms) - consecutive
    rejected_recovery_count = 0
    i = max(0, int(start_index))
    while i <= last_start:
        if not np.all(candidate_mask[i:i + consecutive]):
            i += 1
            continue
        later = irrms[i + consecutive:]
        later_healthy = np.isfinite(later) & (later <= threshold)
        recovery_start = _first_run_start(later_healthy, recovery_run)
        if recovery_start is not None:
            rejected_recovery_count += 1
            i = i + consecutive + recovery_start + recovery_run
            continue
        return i, rejected_recovery_count
    return None, rejected_recovery_count


def find_irrms_fpt(
    irrms: np.ndarray,
    threshold: float,
    consecutive: int = FPT_CONSECUTIVE,
    recovery_run: int = RECOVERY_RUN,
    start_index: int = SEARCH_START,
):
    fpt, _ = evaluate_irrms_fpt(
        irrms,
        threshold,
        consecutive=consecutive,
        recovery_run=recovery_run,
        start_index=start_index,
    )
    return fpt


def baseline_irrms_quantiles(
    series_by_bearing: dict,
    start: int = IRRMS_WINDOW - 1,
    stop: int = HEALTHY_FILES,
):
    if not series_by_bearing or stop <= start:
        raise ValueError('Need bearing series and a non-empty baseline slice')
    expected = stop - start
    slices = []
    for bearing, values in series_by_bearing.items():
        values = np.asarray(values, dtype=np.float64)
        baseline = values[start:stop]
        finite_count = int(np.isfinite(baseline).sum())
        if len(baseline) != expected or finite_count != expected:
            raise ValueError(
                f'{bearing} must contribute exactly {expected} finite baseline IRRMS values'
            )
        slices.append(baseline)
    pooled = np.concatenate(slices)
    return {
        'q99': float(np.quantile(pooled, 0.99)),
        'q99_5': float(np.quantile(pooled, 0.995)),
    }


def build_boundary_flags(
    fpt_found: bool,
    fpt_idx,
    baseline_alarm_runs: int,
    degraded_count,
    search_start: int = SEARCH_START,
    near_end_min_degraded_files: int = NEAR_END_MIN_DEGRADED_FILES,
) -> str:
    flags = []
    if baseline_alarm_runs > 0:
        flags.append('baseline_crossing')
    if not fpt_found:
        flags.append('not_found')
    else:
        if int(fpt_idx) == int(search_start):
            flags.append('search_start_hit')
        if int(degraded_count) < int(near_end_min_degraded_files):
            flags.append('near_end')
    return 'ok' if not flags else ';'.join(flags)


print('Core causal IRRMS/FPT functions OK')"""
    ),
    code(
        """# 4) Dataset IO and vectorized RMS
def bearing_path(bearing: str) -> Path:
    return MAIN_DIR / f'{bearing}.pkz'


def load_bearing_df(bearing: str) -> pd.DataFrame:
    path = bearing_path(bearing)
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Check MAIN_DIR.')
    with open(path, 'rb') as handle:
        df = pkl.load(handle)
    required = {'horiz accel', 'vert accel'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing columns: {sorted(missing)}')
    if len(df) % DATA_POINTS != 0:
        raise ValueError(f'{path}: row count {len(df)} is not divisible by {DATA_POINTS}')
    return df


def rms_series_from_df(df: pd.DataFrame) -> np.ndarray:
    n_files = len(df) // DATA_POINTS
    horizontal = df['horiz accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    vertical = df['vert accel'].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    rms = np.sqrt(np.mean(horizontal ** 2 + vertical ** 2, axis=1)).astype(np.float32)
    if not np.isfinite(rms).all():
        raise ValueError('Non-finite RMS values found')
    return rms


def extract_bearing_features(bearing: str) -> dict:
    rms = rms_series_from_df(load_bearing_df(bearing))
    rrms, rms_norm = relative_rms(rms)
    irrms = causal_irrms(rrms)
    file_index = np.arange(len(rms), dtype=np.int32)
    feature_path = OUT_DIR / f'{bearing}_irrms_features.npz'
    np.savez_compressed(
        feature_path,
        rms=rms,
        rrms=rrms,
        irrms=irrms,
        file_index=file_index,
        elapsed_seconds=file_index.astype(np.float32) * SAMPLE_INTERVAL_SECONDS,
        bearing_id=np.full(len(rms), bearing),
    )
    return {
        'bearing': bearing,
        'rms': rms,
        'rrms': rrms,
        'irrms': irrms,
        'rms_norm_first_100': float(rms_norm),
        'feature_path': str(feature_path),
    }


print('Dataset IO and feature extraction OK')"""
    ),
    code(
        """# 5) First pass: compute causal IRRMS for all six learning bearings
features_by_bearing = {}
if RUN_FULL_PIPELINE:
    for bearing in BEARINGS:
        print('EXTRACTING:', bearing)
        features_by_bearing[bearing] = extract_bearing_features(bearing)
else:
    print('RUN_FULL_PIPELINE=False — no bearing processed.')"""
    ),
    code(
        """# 6) Build one global, deduplicated threshold candidate table
threshold_candidates = []
baseline_quantiles = {}
if RUN_FULL_PIPELINE:
    baseline_quantiles = baseline_irrms_quantiles({
        bearing: payload['irrms']
        for bearing, payload in features_by_bearing.items()
    })

    sources_by_value = {}
    for value in FIXED_IRRMS_THRESHOLDS:
        sources_by_value.setdefault(float(value), []).append('fixed')
    for name, value in baseline_quantiles.items():
        sources_by_value.setdefault(float(value), []).append(name)

    for value in sorted(sources_by_value):
        threshold_candidates.append({
            'threshold': float(value),
            'source': ';'.join(sources_by_value[value]),
        })

    candidate_df = pd.DataFrame(threshold_candidates)
    candidate_path = OUT_DIR / 'irrms_threshold_candidates.csv'
    candidate_df.to_csv(candidate_path, index=False)
    display(candidate_df)
    print('Saved:', candidate_path)"""
    ),
    code(
        """# 7) Evaluate and plot one bearing-threshold pair
def evaluate_pair(bearing: str, payload: dict, threshold: float, source: str) -> dict:
    rms = payload['rms']
    rrms = payload['rrms']
    irrms = payload['irrms']
    file_index = np.arange(len(rms), dtype=np.int32)
    fpt, rejected_count = evaluate_irrms_fpt(
        irrms,
        threshold=threshold,
        start_index=SEARCH_START,
    )
    fpt_found = fpt is not None
    baseline_mask = np.isfinite(irrms[:HEALTHY_FILES]) & (
        irrms[:HEALTHY_FILES] > threshold
    )
    baseline_run_count = _count_run_starts(baseline_mask, FPT_CONSECUTIVE)

    y_state = None
    if fpt_found:
        y_state = np.zeros(len(rms), dtype=np.uint8)
        y_state[fpt:] = 1

    threshold_token = f'{threshold:.8f}'.replace('.', 'p')
    plot_path = OUT_DIR / f'{bearing}_irrms_threshold_{threshold_token}_audit.png'
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.3))
    axes[0].plot(file_index, rrms, lw=1.0, label='RRMS')
    axes[0].plot(file_index, irrms, lw=1.2, label='Causal IRRMS (30)')
    axes[0].axhline(threshold, color='red', ls='--', label=f'T={threshold:.6g}')
    if fpt_found:
        axes[0].axvline(fpt, color='purple', ls=':', label=f'FPT={fpt}')
    axes[0].set_title(f'{bearing} — {source}')
    axes[0].set_xlabel('File index (zero-based)')
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(file_index, irrms, lw=1.2, color='#E67E22', label='Causal IRRMS')
    axes[1].axhline(threshold, color='red', ls='--', label=f'T={threshold:.6g}')
    axes[1].axvline(SEARCH_START, color='gray', ls=':', label='search start=100')
    if fpt_found:
        axes[1].axvline(fpt, color='purple', ls=':', label=f'FPT={fpt}')
    axes[1].set_title('Threshold audit')
    axes[1].set_xlabel('File index (zero-based)')
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    if y_state is None:
        axes[2].text(0.5, 0.5, 'No valid FPT', ha='center', va='center', transform=axes[2].transAxes)
    else:
        axes[2].step(file_index, y_state, where='post', color='#1D9E75', lw=1.8)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0, 1], ['Healthy', 'Degraded'])
    axes[2].set_title('Candidate retrospective label')
    axes[2].set_xlabel('File index (zero-based)')
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    healthy_count = int(fpt) if fpt_found else None
    degraded_count = int(len(rms) - fpt) if fpt_found else None
    boundary_flag = build_boundary_flags(
        fpt_found=fpt_found,
        fpt_idx=fpt,
        baseline_alarm_runs=baseline_run_count,
        degraded_count=degraded_count,
    )
    return {
        'bearing_id': bearing,
        'n_files': int(len(rms)),
        'rms_norm_first_100': payload['rms_norm_first_100'],
        'threshold_name': source,
        'threshold_value': float(threshold),
        'fpt_index': int(fpt) if fpt_found else None,
        'fpt_seconds': int(fpt * SAMPLE_INTERVAL_SECONDS) if fpt_found else None,
        'fpt_found': bool(fpt_found),
        'boundary_flag': boundary_flag,
        'baseline_crossing': bool(baseline_run_count > 0),
        'search_start_hit': bool(fpt == SEARCH_START) if fpt_found else False,
        'near_end': bool(
            fpt_found and degraded_count < NEAR_END_MIN_DEGRADED_FILES
        ),
        'not_found': bool(not fpt_found),
        'healthy_count': healthy_count,
        'degraded_count': degraded_count,
        'healthy_ratio': float(healthy_count / len(rms)) if fpt_found else None,
        'degraded_ratio': float(degraded_count / len(rms)) if fpt_found else None,
        'baseline_alarm_runs': int(baseline_run_count),
        'rejected_recovery_count': int(rejected_count),
        'algorithm': 'causal_irrms_fpt_reference',
        'healthy_files': int(HEALTHY_FILES),
        'irrms_window': int(IRRMS_WINDOW),
        'search_start': int(SEARCH_START),
        'consecutive': int(FPT_CONSECUTIVE),
        'recovery_run': int(RECOVERY_RUN),
        'std_ddof': int(STD_DDOF),
        'threshold_operator': THRESHOLD_OPERATOR,
        'near_end_min_degraded_files': int(NEAR_END_MIN_DEGRADED_FILES),
        'plot_path': str(plot_path),
    }


print('Pair evaluator OK')"""
    ),
    code(
        """# 8) Second pass: full sensitivity matrix
sensitivity_rows = []
if RUN_FULL_PIPELINE:
    for candidate in threshold_candidates:
        threshold = candidate['threshold']
        source = candidate['source']
        for bearing in BEARINGS:
            print(f'EVALUATING {bearing}: T={threshold:.8g} ({source})')
            sensitivity_rows.append(evaluate_pair(
                bearing,
                features_by_bearing[bearing],
                threshold,
                source,
            ))

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_path = OUT_DIR / 'irrms_fpt_sensitivity.csv'
    sensitivity_df.to_csv(sensitivity_path, index=False)
    display(sensitivity_df)
    print('Saved:', sensitivity_path)"""
    ),
    code(
        """# 9) Freeze Q99 labels and write the Task 1 handoff
threshold_frozen = True
ready_for_cwt = False

if RUN_FULL_PIPELINE:
    frozen_rows = []
    for bearing in BEARINGS:
        payload = features_by_bearing[bearing]
        irrms = payload['irrms']
        fpt, rejected_count = evaluate_irrms_fpt(
            irrms,
            threshold=FROZEN_IRRMS_THRESHOLD,
            start_index=SEARCH_START,
        )
        baseline_mask = np.isfinite(irrms[:HEALTHY_FILES]) & (
            irrms[:HEALTHY_FILES] > FROZEN_IRRMS_THRESHOLD
        )
        baseline_alarm_runs = _count_run_starts(baseline_mask, FPT_CONSECUTIVE)
        if fpt is None:
            raise RuntimeError(f'{bearing}: frozen threshold produced no FPT')
        if fpt == SEARCH_START or baseline_alarm_runs > 0:
            raise RuntimeError(
                f'{bearing}: frozen-label boundary failure: '
                f'fpt={fpt}, baseline_alarm_runs={baseline_alarm_runs}'
            )

        y_state = (np.arange(len(irrms)) >= fpt).astype(np.uint8)
        file_index = np.arange(len(irrms), dtype=np.int32)
        label_path = OUT_DIR / f'{bearing}_fpt_labels.npz'
        temp_path = OUT_DIR / f'{bearing}_fpt_labels.tmp.npz'
        np.savez_compressed(
            temp_path,
            rms=payload['rms'],
            rrms=payload['rrms'],
            irrms=irrms,
            y_state=y_state,
            file_index=file_index,
            elapsed_seconds=file_index.astype(np.float32) * SAMPLE_INTERVAL_SECONDS,
            bearing_id=np.full(len(irrms), bearing),
            fpt_index=np.int32(fpt),
            frozen_irrms_threshold=np.float64(FROZEN_IRRMS_THRESHOLD),
            label_version=np.array('causal-irrms-fpt-reference-v3-q99'),
        )
        temp_path.replace(label_path)
        frozen_rows.append({
            'bearing_id': bearing,
            'n_files': int(len(irrms)),
            'fpt_index': int(fpt),
            'fpt_seconds': int(fpt * SAMPLE_INTERVAL_SECONDS),
            'healthy_count': int(fpt),
            'degraded_count': int(len(irrms) - fpt),
            'degraded_ratio': float((len(irrms) - fpt) / len(irrms)),
            'baseline_alarm_runs': int(baseline_alarm_runs),
            'rejected_recovery_count': int(rejected_count),
            'labels_path': str(label_path),
        })

    frozen_summary_df = pd.DataFrame(frozen_rows)
    frozen_summary_path = OUT_DIR / 'fpt_reference_frozen_summary.csv'
    frozen_summary_df.to_csv(frozen_summary_path, index=False)
    ready_for_cwt = bool(
        len(frozen_summary_df) == len(BEARINGS)
        and (frozen_summary_df['baseline_alarm_runs'] == 0).all()
    )

    config = {
        'version': 'causal-irrms-fpt-reference-v3-q99',
        'bearings': BEARINGS,
        'data_points_per_file': DATA_POINTS,
        'sampling_frequency_hz': SAMPLING_FREQ,
        'sample_interval_seconds': SAMPLE_INTERVAL_SECONDS,
        'index_convention': 'zero-based file index',
        'healthy_files': HEALTHY_FILES,
        'irrms_window': IRRMS_WINDOW,
        'fpt_consecutive': FPT_CONSECUTIVE,
        'recovery_run': RECOVERY_RUN,
        'search_start': SEARCH_START,
        'std_ddof': STD_DDOF,
        'threshold_operator': THRESHOLD_OPERATOR,
        'near_end_min_degraded_files': NEAR_END_MIN_DEGRADED_FILES,
        'fixed_thresholds': list(FIXED_IRRMS_THRESHOLDS),
        'baseline_quantiles': baseline_quantiles,
        'threshold_candidates': threshold_candidates,
        'frozen_irrms_threshold': FROZEN_IRRMS_THRESHOLD,
        'frozen_threshold_source': 'pooled healthy-baseline Q99',
        'threshold_frozen': threshold_frozen,
        'ready_for_cwt': ready_for_cwt,
        'label_type': 'causal IRRMS-based retrospective operational FPT reference',
    }
    config_path = OUT_DIR / 'fpt_reference_frozen_config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    (OUT_DIR / 'irrms_fpt_sensitivity_config.json').write_text(
        json.dumps(config, indent=2), encoding='utf-8'
    )

    eligibility_df = sensitivity_df.groupby(
        ['threshold_name', 'threshold_value'], as_index=False
    ).agg(
        n_bearings=('bearing_id', 'nunique'),
        all_fpt_found=('fpt_found', 'all'),
        any_not_found=('not_found', 'any'),
        any_search_start_hit=('search_start_hit', 'any'),
        any_baseline_crossing=('baseline_crossing', 'any'),
        any_near_end=('near_end', 'any'),
        total_baseline_alarm_runs=('baseline_alarm_runs', 'sum'),
        total_rejected_recovery=('rejected_recovery_count', 'sum'),
        mean_degraded_ratio=('degraded_ratio', 'mean'),
        min_degraded_count=('degraded_count', 'min'),
    )
    eligibility_df['auto_screen_pass'] = (
        eligibility_df['all_fpt_found']
        & ~eligibility_df['any_search_start_hit']
        & ~eligibility_df['any_baseline_crossing']
    )
    eligibility_df['manual_review_required'] = True
    eligibility_df = eligibility_df.sort_values(
        ['auto_screen_pass', 'threshold_value'],
        ascending=[False, True],
    )
    eligibility_path = OUT_DIR / 'irrms_threshold_auto_screen.csv'
    eligibility_df.to_csv(eligibility_path, index=False)
    display(eligibility_df)

    print('Saved:', config_path)
    print('Saved:', frozen_summary_path)
    print('Saved:', eligibility_path)
    print('THRESHOLD_FROZEN:', threshold_frozen)
    print('READY_FOR_CWT:', ready_for_cwt)
    if not ready_for_cwt:
        raise RuntimeError('Frozen Task 1 handoff failed; do not run Task 2')
    print('Task 1 labels are frozen. Task 2 may consume only these label artifacts.')"""
    ),
    code(
        """# 10) Report-ready visualizations for Task 1
import zipfile

VIS_DIR = OUT_DIR / 'task1_fpt_visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

RMS_LINE_COLOR = '#34495E'
RRMS_LINE_COLOR = '#2E86C1'
IRRMS_LINE_COLOR = '#E67E22'
THRESHOLD_COLOR = '#5F6368'
FPT_LINE_COLOR = '#D62728'
FPT_LINESTYLE = '-.'
SEARCH_START_COLOR = '#7F8C8D'
BASELINE_SHADE_COLOR = '#B0B7BC'
RMS_LINEWIDTH = 1.25
RRMS_LINEWIDTH = 1.45
IRRMS_LINEWIDTH = 2.05
THRESHOLD_LINEWIDTH = 1.7
FPT_LINEWIDTH = 2.6


def _axis_limit_with_margin(values: np.ndarray, margin: float = 0.08):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        pad = max(abs(hi), 1.0) * margin
    else:
        pad = (hi - lo) * margin
    return lo - pad, hi + pad


def make_bearing_task1_visual(
    bearing: str,
    payload: dict,
    fpt_index: int,
    threshold: float = FROZEN_IRRMS_THRESHOLD,
    zoom_radius: int = 180,
):
    rms = payload['rms']
    rrms = payload['rrms']
    irrms = payload['irrms']
    file_index = np.arange(len(rms), dtype=np.int32)
    zoom_left = max(0, int(fpt_index) - int(zoom_radius))
    zoom_right = min(len(rms) - 1, int(fpt_index) + int(zoom_radius))

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.8), sharex=False)
    fig.suptitle(f'{bearing} | FPT={fpt_index} | Q99={threshold:.6f}', y=0.995)

    axes[0].plot(file_index, rms, color=RMS_LINE_COLOR, lw=RMS_LINEWIDTH, label='RMS')
    axes[0].axvspan(0, HEALTHY_FILES - 1, color=BASELINE_SHADE_COLOR, alpha=0.18, label='baseline 0-99')
    axes[0].axvline(fpt_index, color=FPT_LINE_COLOR, ls=FPT_LINESTYLE, lw=FPT_LINEWIDTH, label='FPT')
    axes[0].set_title('Raw RMS')
    axes[0].set_ylabel('RMS')
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='upper left', ncol=3)

    axes[1].plot(file_index, rrms, color=RRMS_LINE_COLOR, lw=RRMS_LINEWIDTH, alpha=0.82, label='RRMS')
    axes[1].plot(file_index, irrms, color=IRRMS_LINE_COLOR, lw=IRRMS_LINEWIDTH, label='IRRMS')
    axes[1].axhline(threshold, color=THRESHOLD_COLOR, ls='--', lw=THRESHOLD_LINEWIDTH, label='Q99')
    axes[1].axvline(SEARCH_START, color=SEARCH_START_COLOR, ls=':', lw=1.4, label='start=100')
    axes[1].axvline(fpt_index, color=FPT_LINE_COLOR, ls=FPT_LINESTYLE, lw=FPT_LINEWIDTH, label='FPT')
    axes[1].set_title('Health indicator')
    axes[1].set_ylabel('RRMS / IRRMS')
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='upper left', ncol=5)

    zoom_slice = slice(zoom_left, zoom_right + 1)
    axes[2].plot(file_index[zoom_slice], rrms[zoom_slice], color=RRMS_LINE_COLOR, lw=RRMS_LINEWIDTH, alpha=0.82, label='RRMS')
    axes[2].plot(file_index[zoom_slice], irrms[zoom_slice], color=IRRMS_LINE_COLOR, lw=IRRMS_LINEWIDTH, label='IRRMS')
    axes[2].axhline(threshold, color=THRESHOLD_COLOR, ls='--', lw=THRESHOLD_LINEWIDTH, label='Q99')
    axes[2].axvline(fpt_index, color=FPT_LINE_COLOR, ls=FPT_LINESTYLE, lw=FPT_LINEWIDTH, label='FPT')
    axes[2].set_xlim(zoom_left, zoom_right)
    ylim = _axis_limit_with_margin(np.r_[rrms[zoom_slice], irrms[zoom_slice], threshold])
    if ylim is not None:
        axes[2].set_ylim(*ylim)
    axes[2].set_title(f'FPT zoom: files {zoom_left}-{zoom_right}')
    axes[2].set_xlabel('File index (zero-based)')
    axes[2].set_ylabel('RRMS / IRRMS')
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc='upper left', ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png_path = VIS_DIR / f'{bearing}_task1_fpt_visual.png'
    pdf_path = VIS_DIR / f'{bearing}_task1_fpt_visual.pdf'
    fig.savefig(png_path, dpi=220, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return png_path, pdf_path


visual_rows = []
if RUN_FULL_PIPELINE:
    frozen_fpt_by_bearing = dict(zip(
        frozen_summary_df['bearing_id'],
        frozen_summary_df['fpt_index'].astype(int),
    ))

    for bearing in BEARINGS:
        png_path, pdf_path = make_bearing_task1_visual(
            bearing=bearing,
            payload=features_by_bearing[bearing],
            fpt_index=frozen_fpt_by_bearing[bearing],
        )
        visual_rows.append({
            'bearing_id': bearing,
            'fpt_index': int(frozen_fpt_by_bearing[bearing]),
            'threshold': float(FROZEN_IRRMS_THRESHOLD),
            'png_path': str(png_path),
            'pdf_path': str(pdf_path),
        })

    fig, axes = plt.subplots(3, 2, figsize=(15, 10.5), sharex=False)
    axes = axes.ravel()
    for ax, bearing in zip(axes, BEARINGS):
        payload = features_by_bearing[bearing]
        irrms = payload['irrms']
        rrms = payload['rrms']
        file_index = np.arange(len(irrms), dtype=np.int32)
        fpt_index = frozen_fpt_by_bearing[bearing]
        ax.plot(file_index, rrms, color=RRMS_LINE_COLOR, lw=1.05, alpha=0.72, label='RRMS')
        ax.plot(file_index, irrms, color=IRRMS_LINE_COLOR, lw=1.55, label='IRRMS')
        ax.axhline(FROZEN_IRRMS_THRESHOLD, color=THRESHOLD_COLOR, ls='--', lw=1.25, label='Q99')
        ax.axvline(fpt_index, color=FPT_LINE_COLOR, ls=FPT_LINESTYLE, lw=2.0, label='FPT')
        ax.set_title(f'{bearing} | FPT={fpt_index}')
        ax.set_xlabel('File index')
        ax.set_ylabel('HI')
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f'Task 1 FPT reference | Q99={FROZEN_IRRMS_THRESHOLD:.6f}', y=0.995)
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    overview_png = VIS_DIR / 'task1_fpt_overview.png'
    overview_pdf = VIS_DIR / 'task1_fpt_overview.pdf'
    fig.savefig(overview_png, dpi=220, bbox_inches='tight')
    fig.savefig(overview_pdf, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    visual_rows.append({
        'bearing_id': 'ALL',
        'fpt_index': None,
        'threshold': float(FROZEN_IRRMS_THRESHOLD),
        'png_path': str(overview_png),
        'pdf_path': str(overview_pdf),
    })

    visual_summary_df = pd.DataFrame(visual_rows)
    visual_summary_path = VIS_DIR / 'task1_fpt_visualization_summary.csv'
    visual_summary_df.to_csv(visual_summary_path, index=False)

    zip_path = OUT_DIR / 'task1_fpt_visualizations.zip'
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(VIS_DIR.glob('*')):
            if path.is_file():
                zf.write(path, arcname=path.name)

    display(visual_summary_df)
    print('Saved visual summary:', visual_summary_path)
    print('Saved visualization zip:', zip_path)
    print('Visualization files:', len(list(VIS_DIR.glob('*'))))"""
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

output = Path(__file__).resolve().parents[2] / "notebooks" / "01_rrms_fpt_reference_kaggle.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.name)
