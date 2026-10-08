"""Generate a Kaggle notebook for IRRMS method audit and paper figures.

This notebook reruns the frozen Task 1 IRRMS/FPT method for documentation,
figures, and manifest checks. It is separate from the canonical label notebook
so paper artifacts do not accidentally change frozen downstream inputs.
"""

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
        """# Task 1b - IRRMS method audit and paper figures

This notebook reruns the frozen causal-IRRMS FPT reference method for paper
tables and figures. It does not change the frozen Task 1 label protocol.

Outputs:

- per-bearing RMS/RRMS/IRRMS CSV files;
- per-bearing IRRMS plots with the frozen Q99 threshold and FPT;
- a six-bearing overview figure;
- a method audit manifest and downloadable zip."""
    ),
    code(
        """# 1) Imports and frozen protocol constants
import hashlib
import json
import os
import pickle as pkl
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

RAW_DIR = Path(os.environ.get(
    "FPT_TASK1_RAW_DIR",
    "/kaggle/input/datasets/longvu274/train-dataset/",
))
OUT_DIR = Path("/kaggle/working/irrms_paper_audit/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEARINGS = [
    "bearing1_1", "bearing1_2",
    "bearing2_1", "bearing2_2",
    "bearing3_1", "bearing3_2",
]
FROZEN_FPT = {
    "bearing1_1": 1871,
    "bearing1_2": 826,
    "bearing2_1": 151,
    "bearing2_2": 198,
    "bearing3_1": 493,
    "bearing3_2": 1597,
}

DATA_POINTS = 2560
SAMPLING_FREQ = 25600.0
SAMPLE_INTERVAL_SECONDS = 10.0
HEALTHY_FILES = 100
IRRMS_WINDOW = 30
FPT_CONSECUTIVE = 5
RECOVERY_RUN = 30
SEARCH_START = 100
STD_DDOF = 0
THRESHOLD_OPERATOR = ">"
EPS = 1e-8
FROZEN_IRRMS_THRESHOLD = 1.2637933790683746
LABEL_VERSION = "causal-irrms-fpt-reference-v3-q99"
VISUALIZATION_VERSION = "irrms-paper-audit-v1"

print("RAW_DIR:", RAW_DIR)
print("OUT_DIR:", OUT_DIR)
print("Frozen threshold:", FROZEN_IRRMS_THRESHOLD)"""
    ),
    code(
        """# 2) Core frozen IRRMS and FPT functions
def rms_two_channel(record: np.ndarray) -> float:
    record = np.asarray(record, dtype=np.float32)
    if record.ndim != 2 or record.shape[0] != 2:
        raise ValueError(f"Expected [2, n_points], got {record.shape}")
    return float(np.sqrt(np.mean(record[0] ** 2 + record[1] ** 2)))


def relative_rms(rms: np.ndarray, healthy_files: int = HEALTHY_FILES):
    rms = np.asarray(rms, dtype=np.float32)
    if rms.ndim != 1 or len(rms) < healthy_files:
        raise ValueError(f"Need at least {healthy_files} RMS samples")
    rms_norm = float(np.mean(rms[:healthy_files]))
    if not np.isfinite(rms_norm) or rms_norm <= EPS:
        raise ValueError(f"Invalid healthy RMS norm: {rms_norm}")
    return (rms / rms_norm).astype(np.float32), rms_norm


def causal_irrms(rrms: np.ndarray, window: int = IRRMS_WINDOW) -> np.ndarray:
    rrms = np.asarray(rrms, dtype=np.float32)
    if rrms.ndim != 1 or window < 2:
        raise ValueError("rrms must be 1-D and window must be at least 2")
    if not np.isfinite(rrms).all():
        raise ValueError("rrms must contain only finite values")

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
            if current_residual < lower:
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
        raise ValueError("run_length must be at least 1")
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
        raise ValueError("irrms must be one-dimensional")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

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


def baseline_irrms_quantiles(series_by_bearing: dict[str, np.ndarray]) -> dict[str, float]:
    slices = []
    start = IRRMS_WINDOW - 1
    stop = HEALTHY_FILES
    expected = stop - start
    for bearing, values in series_by_bearing.items():
        baseline = np.asarray(values, dtype=np.float64)[start:stop]
        if len(baseline) != expected or int(np.isfinite(baseline).sum()) != expected:
            raise ValueError(f"{bearing} must contribute {expected} finite baseline IRRMS values")
        slices.append(baseline)
    pooled = np.concatenate(slices)
    return {
        "q99": float(np.quantile(pooled, 0.99)),
        "q99_5": float(np.quantile(pooled, 0.995)),
        "n_values": int(len(pooled)),
    }


print("Core method functions OK")"""
    ),
    code(
        """# 3) IO helpers
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(bearing: str) -> Path:
    return RAW_DIR / f"{bearing}.pkz"


def load_bearing_records(bearing: str) -> np.ndarray:
    if bearing not in FROZEN_FPT:
        raise ValueError(f"Unknown learning bearing: {bearing}")
    path = raw_path(bearing)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw bearing file: {path}")
    with open(path, "rb") as handle:
        frame = pkl.load(handle)
    required = {"horiz accel", "vert accel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(frame) % DATA_POINTS != 0:
        raise ValueError(f"{path}: row count is not divisible by {DATA_POINTS}")
    n_files = len(frame) // DATA_POINTS
    horizontal = frame["horiz accel"].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    vertical = frame["vert accel"].to_numpy(dtype=np.float32).reshape(n_files, DATA_POINTS)
    records = np.stack([horizontal, vertical], axis=1)
    if not np.isfinite(records).all():
        raise ValueError(f"{bearing}: raw records contain non-finite values")
    return records


def compute_bearing_series(bearing: str) -> dict:
    records = load_bearing_records(bearing)
    rms = np.asarray([rms_two_channel(record) for record in records], dtype=np.float32)
    rrms, rms_norm = relative_rms(rms)
    irrms = causal_irrms(rrms)
    fpt, rejected_count = evaluate_irrms_fpt(irrms, FROZEN_IRRMS_THRESHOLD)
    if fpt != FROZEN_FPT[bearing]:
        raise ValueError(f"{bearing}: recomputed FPT {fpt} != frozen {FROZEN_FPT[bearing]}")
    baseline_mask = np.isfinite(irrms[:HEALTHY_FILES]) & (irrms[:HEALTHY_FILES] > FROZEN_IRRMS_THRESHOLD)
    baseline_alarm_runs = _count_run_starts(baseline_mask, FPT_CONSECUTIVE)
    y_state = (np.arange(len(irrms)) >= fpt).astype(np.uint8)
    return {
        "bearing": bearing,
        "n_files": int(len(rms)),
        "file_index": np.arange(len(rms), dtype=np.int32),
        "rms": rms,
        "rrms": rrms,
        "irrms": irrms,
        "rms_norm_first_100": float(rms_norm),
        "fpt": int(fpt),
        "rejected_recovery_count": int(rejected_count),
        "baseline_alarm_runs": int(baseline_alarm_runs),
        "y_state": y_state,
        "raw_sha256": sha256_file(raw_path(bearing)),
    }


print("IO helpers OK")"""
    ),
    code(
        """# 4) Plot helpers
def save_per_bearing_plot(payload: dict) -> dict:
    bearing = payload["bearing"]
    fpt = payload["fpt"]
    file_index = payload["file_index"]
    rms = payload["rms"]
    rrms = payload["rrms"]
    irrms = payload["irrms"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(file_index, rms, lw=0.9, color="#2C3E50")
    axes[0].axvline(fpt, color="purple", ls=":", lw=1.5, label=f"FPT={fpt}")
    axes[0].set_ylabel("RMS")
    axes[0].set_title(f"{bearing}: raw RMS")
    axes[0].legend(loc="upper left")

    axes[1].plot(file_index, rrms, lw=0.8, alpha=0.55, label="RRMS")
    axes[1].plot(file_index, irrms, lw=1.2, label="causal IRRMS")
    axes[1].axhline(FROZEN_IRRMS_THRESHOLD, color="red", ls="--", lw=1.2, label=f"Q99={FROZEN_IRRMS_THRESHOLD:.6f}")
    axes[1].axvline(SEARCH_START, color="gray", ls=":", lw=1.0, label="search start=100")
    axes[1].axvline(fpt, color="purple", ls=":", lw=1.5, label=f"FPT={fpt}")
    axes[1].set_xlabel("File index (zero-based)")
    axes[1].set_ylabel("RRMS / IRRMS")
    axes[1].set_title("Causal IRRMS with frozen threshold")
    axes[1].legend(loc="upper left", ncol=2)

    png_path = OUT_DIR / f"{bearing}_irrms_paper_audit.png"
    pdf_path = OUT_DIR / f"{bearing}_irrms_paper_audit.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
        "plot_png_sha256": sha256_file(png_path),
        "plot_pdf_sha256": sha256_file(pdf_path),
    }


def save_overview_plot(series_by_bearing: dict[str, dict]) -> dict:
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=False, constrained_layout=True)
    for axis, bearing in zip(axes.ravel(), BEARINGS):
        payload = series_by_bearing[bearing]
        file_index = payload["file_index"]
        irrms = payload["irrms"]
        fpt = payload["fpt"]
        axis.plot(file_index, irrms, lw=1.0, color="#E67E22")
        axis.axhline(FROZEN_IRRMS_THRESHOLD, color="red", ls="--", lw=1.0)
        axis.axvline(fpt, color="purple", ls=":", lw=1.3)
        axis.axvline(SEARCH_START, color="gray", ls=":", lw=0.8)
        axis.set_title(f"{bearing} | FPT={fpt}")
        axis.set_xlabel("File index")
        axis.set_ylabel("causal IRRMS")
    fig.suptitle("Frozen causal-IRRMS FPT references across six learning bearings", fontsize=14)
    png_path = OUT_DIR / "all_bearings_irrms_fpt_overview.png"
    pdf_path = OUT_DIR / "all_bearings_irrms_fpt_overview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "overview_png": str(png_path),
        "overview_pdf": str(pdf_path),
        "overview_png_sha256": sha256_file(png_path),
        "overview_pdf_sha256": sha256_file(pdf_path),
    }


print("Plot helpers OK")"""
    ),
    code(
        """# 5) Run IRRMS method audit
series_by_bearing = {}
summary_rows = []

for bearing in BEARINGS:
    print("\\n" + "=" * 80)
    print("IRRMS:", bearing)
    payload = compute_bearing_series(bearing)
    plot_info = save_per_bearing_plot(payload)
    payload.update(plot_info)
    series_by_bearing[bearing] = payload

    table = pd.DataFrame({
        "file_index": payload["file_index"],
        "rms": payload["rms"],
        "rrms": payload["rrms"],
        "irrms": payload["irrms"],
        "y_state": payload["y_state"],
    })
    csv_path = OUT_DIR / f"{bearing}_irrms_series.csv"
    table.to_csv(csv_path, index=False)
    payload["series_csv"] = str(csv_path)
    payload["series_csv_sha256"] = sha256_file(csv_path)

    summary_rows.append({
        "bearing": bearing,
        "n_files": payload["n_files"],
        "fpt": payload["fpt"],
        "fpt_seconds": payload["fpt"] * SAMPLE_INTERVAL_SECONDS,
        "healthy_count": payload["fpt"],
        "degraded_count": payload["n_files"] - payload["fpt"],
        "rms_norm_first_100": payload["rms_norm_first_100"],
        "baseline_alarm_runs": payload["baseline_alarm_runs"],
        "rejected_recovery_count": payload["rejected_recovery_count"],
        "raw_sha256": payload["raw_sha256"],
        "series_csv": payload["series_csv"],
        "plot_png": payload["plot_png"],
    })
    display(Image(filename=payload["plot_png"]))

overview_info = save_overview_plot(series_by_bearing)
display(Image(filename=overview_info["overview_png"]))

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "irrms_fpt_reference_summary.csv"
summary_df.to_csv(summary_path, index=False)
display(summary_df)
print("Saved summary:", summary_path)"""
    ),
    code(
        """# 6) Method checks and manifest
quantiles = baseline_irrms_quantiles({
    bearing: payload["irrms"]
    for bearing, payload in series_by_bearing.items()
})
q99_delta = abs(quantiles["q99"] - FROZEN_IRRMS_THRESHOLD)
if q99_delta > 1e-7:
    raise RuntimeError(
        f"Pooled healthy-baseline Q99 changed: {quantiles['q99']} vs {FROZEN_IRRMS_THRESHOLD}"
    )

manifest_entries = []
for bearing in BEARINGS:
    payload = series_by_bearing[bearing]
    manifest_entries.append({
        "bearing": bearing,
        "n_files": payload["n_files"],
        "fpt": payload["fpt"],
        "frozen_fpt_expected": FROZEN_FPT[bearing],
        "fpt_match": payload["fpt"] == FROZEN_FPT[bearing],
        "rms_norm_first_100": payload["rms_norm_first_100"],
        "baseline_alarm_runs": payload["baseline_alarm_runs"],
        "rejected_recovery_count": payload["rejected_recovery_count"],
        "raw_sha256": payload["raw_sha256"],
        "series_csv": payload["series_csv"],
        "series_csv_sha256": payload["series_csv_sha256"],
        "plot_png": payload["plot_png"],
        "plot_png_sha256": payload["plot_png_sha256"],
        "plot_pdf": payload["plot_pdf"],
        "plot_pdf_sha256": payload["plot_pdf_sha256"],
    })

manifest = {
    "visualization_version": VISUALIZATION_VERSION,
    "label_version": LABEL_VERSION,
    "purpose": "paper_only_irrms_method_audit_and_figures",
    "raw_dir": str(RAW_DIR),
    "out_dir": str(OUT_DIR),
    "data_points_per_file": DATA_POINTS,
    "sampling_frequency_hz": SAMPLING_FREQ,
    "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
    "healthy_files": HEALTHY_FILES,
    "irrms_window": IRRMS_WINDOW,
    "fpt_consecutive": FPT_CONSECUTIVE,
    "recovery_run": RECOVERY_RUN,
    "search_start": SEARCH_START,
    "std_ddof": STD_DDOF,
    "threshold_operator": THRESHOLD_OPERATOR,
    "frozen_irrms_threshold": FROZEN_IRRMS_THRESHOLD,
    "pooled_baseline_quantiles": quantiles,
    "q99_delta_from_frozen": q99_delta,
    "frozen_fpt": FROZEN_FPT,
    "overview": overview_info,
    "entries": manifest_entries,
}

manifest_path = OUT_DIR / "irrms_paper_audit_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = Path("/kaggle/working/irrms_paper_audit.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT_DIR.glob("*")):
        if path.is_file():
            archive.write(path, arcname=path.name)

print("Q99:", quantiles["q99"])
print("Q99 delta from frozen:", q99_delta)
print("Manifest:", manifest_path)
print("Zip:", zip_path)
print("Zip sha256:", sha256_file(zip_path))"""
    ),
]


if __name__ == "__main__":
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).with_name("01b_irrms_paper_audit_kaggle.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.name} ({len(cells)} cells)")
