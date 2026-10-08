"""Generate a Kaggle notebook for paper-only CWT figure exports.

This notebook is intentionally separate from frozen Task 2 cache generation.
It recomputes selected CWT examples for publication figures without modifying
the training/evaluation cache artifacts.
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
        """# Task 2b - CWT paper figure export

This notebook recomputes selected Morlet CWT log-power images for paper figures.
It does **not** replace the frozen Task 2 cache used for model training or
evaluation.

Use this notebook when you need publication-quality panels such as before-FPT,
at-FPT, after-FPT, or a causal 16-frame sequence example."""
    ),
    code(
        """# 1) Imports and frozen visualization contract
import hashlib
import json
import os
import pickle as pkl
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from IPython.display import Image, display
from skimage.transform import resize

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

DATA_POINTS = 2560
SAMPLING_FREQ = 25600.0
ACQUISITION_DURATION_SECONDS = DATA_POINTS / SAMPLING_FREQ

CWT_WAVELET = "morl"
CWT_SCALES = np.geomspace(1, 512, 128).astype(np.float32)
IMAGE_SHAPE = (128, 128)
LOG_POWER_EPS = 1e-3
CWT_FREQUENCIES_HZ = (
    pywt.scale2frequency(CWT_WAVELET, CWT_SCALES) * SAMPLING_FREQ
).astype(np.float32)

LABEL_VERSION = "causal-irrms-fpt-reference-v3-q99"
VISUALIZATION_VERSION = "cwt-paper-figures-v1"
SEQUENCE_LENGTH = 16

FROZEN_FPT = {
    "bearing1_1": 1871,
    "bearing1_2": 826,
    "bearing2_1": 151,
    "bearing2_2": 198,
    "bearing3_1": 493,
    "bearing3_2": 1597,
}

RAW_DIR = Path(os.environ.get(
    "FPT_TASK2_RAW_DIR",
    "/kaggle/input/datasets/longvu274/train-dataset/",
))
LABEL_DIR = Path(os.environ.get(
    "FPT_TASK2_LABEL_DIR",
    "/kaggle/input/causal-irrms-fpt-reference/",
))
OUT_DIR = Path("/kaggle/working/cwt_paper_figures/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("RAW_DIR:", RAW_DIR)
print("LABEL_DIR:", LABEL_DIR)
print("OUT_DIR:", OUT_DIR)
print("CWT:", CWT_WAVELET, len(CWT_SCALES), "scales", float(CWT_SCALES[0]), "to", float(CWT_SCALES[-1]))"""
    ),
    code(
        """# 2) Figure plan
# Edit this list if you want different examples. Indices are zero-based file indices.
# Use offset_from_fpt for after-FPT examples so short bearings are clamped safely.
FIGURE_EXAMPLES = [
    {"bearing": "bearing3_1", "index": 450, "tag": "bearing3_1_before_fpt"},
    {"bearing": "bearing3_1", "index": 493, "tag": "bearing3_1_at_fpt"},
    {"bearing": "bearing3_1", "offset_from_fpt": 27, "tag": "bearing3_1_after_fpt"},
    {"bearing": "bearing1_1", "index": 100, "tag": "bearing1_1_early_healthy"},
    {"bearing": "bearing1_1", "index": 1871, "tag": "bearing1_1_at_fpt"},
    {"bearing": "bearing1_2", "index": 100, "tag": "bearing1_2_early_healthy"},
    {"bearing": "bearing1_2", "index": 826, "tag": "bearing1_2_at_fpt"},
]

SEQUENCE_EXAMPLES = [
    {"bearing": "bearing3_1", "end_index": 493, "tag": "bearing3_1_causal_sequence_at_fpt"},
    {"bearing": "bearing1_2", "end_index": 826, "tag": "bearing1_2_causal_sequence_at_fpt"},
]

THREE_TIMEPOINT_BEARINGS = list(FROZEN_FPT)

COLORMAP = "viridis"
DPI = 300
PERCENTILE_RANGE = (1.0, 99.0)
TITLE_FONTSIZE = 12
CHANNEL_TITLE_FONTSIZE = 11
SEQUENCE_INDEX_FONTSIZE = 7
COLORBAR_LABEL = "CWT log-power"

pd.DataFrame(FIGURE_EXAMPLES)"""
    ),
    code(
        """# 3) IO and validation helpers
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


def label_path(bearing: str) -> Path:
    return LABEL_DIR / f"{bearing}_fpt_labels.npz"


def validate_bearing_name(bearing: str) -> None:
    if bearing not in FROZEN_FPT:
        raise ValueError(f"Unknown learning bearing: {bearing}")


def validate_index(bearing: str, index: int, n_files: int) -> None:
    validate_bearing_name(bearing)
    if not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer")
    if index < 0 or index >= n_files:
        raise ValueError(f"{bearing}: index {index} outside [0, {n_files - 1}]")


def resolve_figure_index(item: dict, fpt_index: int, n_files: int) -> int:
    if "index" in item:
        index = int(item["index"])
    elif "offset_from_fpt" in item:
        offset = int(item["offset_from_fpt"])
        index = min(n_files - 1, fpt_index + offset)
    else:
        raise ValueError(f"{item['tag']}: expected index or offset_from_fpt")
    validate_index(str(item["bearing"]), index, n_files)
    return index


def three_timepoint_indices(bearing: str, fpt_index: int, n_files: int) -> list[tuple[str, int]]:
    timepoints = [
        ("Start", 0),
        ("Frozen FPT", int(fpt_index)),
        ("End", int(n_files - 1)),
    ]
    for _, index in timepoints:
        validate_index(bearing, index, n_files)
    return timepoints


def load_raw_records(bearing: str) -> np.ndarray:
    validate_bearing_name(bearing)
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


def load_and_validate_frozen_labels(bearing: str, n_files: int) -> dict:
    path = label_path(bearing)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen labels: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"y_state", "file_index", "bearing_id", "fpt_index", "label_version"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        y_state = data["y_state"].astype(np.uint8)
        file_index = data["file_index"].astype(np.int64)
        bearing_id = data["bearing_id"].astype(str)
        fpt_index = int(np.asarray(data["fpt_index"]).item())
        label_version = str(np.asarray(data["label_version"]).item())

    if len(y_state) != n_files:
        raise ValueError(f"{bearing}: label count {len(y_state)} != raw count {n_files}")
    if not np.array_equal(file_index, np.arange(n_files)):
        raise ValueError(f"{bearing}: label file_index must be zero-based arange")
    if bearing_id.shape != (n_files,) or not np.all(bearing_id == bearing):
        raise ValueError(f"{bearing}: label bearing_id mismatch")
    if fpt_index != FROZEN_FPT[bearing]:
        raise ValueError(f"{bearing}: FPT {fpt_index} != frozen {FROZEN_FPT[bearing]}")
    if label_version != LABEL_VERSION:
        raise ValueError(f"{bearing}: label version mismatch")
    expected = (np.arange(n_files) >= fpt_index).astype(np.uint8)
    if not np.array_equal(y_state, expected):
        raise ValueError(f"{bearing}: y_state does not match frozen FPT step labels")
    return {
        "fpt_index": fpt_index,
        "label_version": label_version,
        "n_files": n_files,
    }


print("IO helpers OK")"""
    ),
    code(
        """# 4) CWT transform functions
def cwt_log_power_image(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (DATA_POINTS,):
        raise ValueError(f"Expected signal shape ({DATA_POINTS},), got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite values")
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
    ).astype(np.float32)
    if image.shape != IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError("Invalid CWT image")
    return image


def cwt_two_channel_record(record: np.ndarray) -> np.ndarray:
    record = np.asarray(record, dtype=np.float32)
    if record.shape != (2, DATA_POINTS):
        raise ValueError(f"Expected record shape (2, {DATA_POINTS}), got {record.shape}")
    return np.stack([cwt_log_power_image(record[0]), cwt_log_power_image(record[1])], axis=0)


def compute_two_channel_examples(records: np.ndarray, indices: list[int]) -> np.ndarray:
    images = []
    for index in indices:
        images.append(cwt_two_channel_record(records[index]))
    return np.stack(images, axis=0).astype(np.float32)


print("CWT transform OK")"""
    ),
    code(
        """# 5) Plot helpers
def display_state_label(index: int, fpt_index: int) -> str:
    if index < fpt_index:
        return "pre-FPT"
    if index == fpt_index:
        return "FPT"
    return "post-FPT"


def robust_limits(images: np.ndarray, percentile_range=PERCENTILE_RANGE) -> tuple[float, float]:
    low, high = np.percentile(images.astype(np.float32), percentile_range)
    low = float(low)
    high = float(high)
    if not np.isfinite([low, high]).all():
        raise ValueError("Non-finite color limits")
    if high <= low:
        high = low + 1.0
    return low, high


def configure_cwt_axis(axis, channel_name: str) -> None:
    frequency_rows = np.array([0, 32, 64, 96, IMAGE_SHAPE[0] - 1], dtype=int)
    frequency_labels = [f"{CWT_FREQUENCIES_HZ[row]:.0f}" for row in frequency_rows]
    x_ticks = [0, (IMAGE_SHAPE[1] - 1) / 2, IMAGE_SHAPE[1] - 1]
    x_labels = ["0.00", f"{ACQUISITION_DURATION_SECONDS / 2:.2f}", f"{ACQUISITION_DURATION_SECONDS:.2f}"]
    axis.set_xticks(x_ticks, x_labels)
    axis.set_xlabel("Time (s)")
    axis.set_yticks(frequency_rows, frequency_labels)
    axis.set_ylabel(f"{channel_name}\\nPseudo-frequency (Hz)")


def save_two_channel_panel(image: np.ndarray, bearing: str, index: int, tag: str, fpt_index: int) -> dict:
    if image.shape != (2, *IMAGE_SHAPE):
        raise ValueError(f"Expected [2,128,128], got {image.shape}")
    vmin, vmax = robust_limits(image)
    channel_names = ["Horizontal", "Vertical"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    plotted = None
    for channel, axis in enumerate(axes):
        plotted = axis.imshow(
            image[channel],
            aspect="auto",
            origin="upper",
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(channel_names[channel], fontsize=CHANNEL_TITLE_FONTSIZE)
        configure_cwt_axis(axis, channel_names[channel])
    fig.suptitle(
        f"{bearing} | file {index} | {display_state_label(index, fpt_index)} | FPT={fpt_index}",
        fontsize=TITLE_FONTSIZE,
    )
    fig.colorbar(plotted, ax=axes.ravel().tolist(), label=COLORBAR_LABEL)
    png_path = OUT_DIR / f"{tag}.png"
    pdf_path = OUT_DIR / f"{tag}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "kind": "two_channel_panel",
        "tag": tag,
        "bearing": bearing,
        "index": int(index),
        "fpt_index": int(fpt_index),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "png_sha256": sha256_file(png_path),
        "pdf_sha256": sha256_file(pdf_path),
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def save_sequence_strip(images: np.ndarray, bearing: str, indices: list[int], tag: str, fpt_index: int) -> dict:
    if images.ndim != 4 or images.shape[1:] != (2, *IMAGE_SHAPE):
        raise ValueError(f"Expected [T,2,128,128], got {images.shape}")
    vmin, vmax = robust_limits(images)
    fig, axes = plt.subplots(2, len(indices), figsize=(1.35 * len(indices), 4.35), constrained_layout=True)
    channel_names = ["Horizontal", "Vertical"]
    plotted = None
    for row in range(2):
        for col, index in enumerate(indices):
            axis = axes[row, col]
            plotted = axis.imshow(
                images[col, row],
                aspect="auto",
                origin="upper",
                cmap=COLORMAP,
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(str(index), fontsize=SEQUENCE_INDEX_FONTSIZE)
            if col == 0:
                axis.set_ylabel(channel_names[row], fontsize=CHANNEL_TITLE_FONTSIZE)
    fig.suptitle(
        f"{bearing} | causal {len(indices)}-frame CWT sequence | end={indices[-1]} | FPT={fpt_index}",
        fontsize=TITLE_FONTSIZE,
    )
    fig.colorbar(plotted, ax=axes.ravel().tolist(), label=COLORBAR_LABEL, shrink=0.72)
    png_path = OUT_DIR / f"{tag}.png"
    pdf_path = OUT_DIR / f"{tag}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "kind": "causal_sequence_strip",
        "tag": tag,
        "bearing": bearing,
        "indices": [int(i) for i in indices],
        "end_index": int(indices[-1]),
        "fpt_index": int(fpt_index),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "png_sha256": sha256_file(png_path),
        "pdf_sha256": sha256_file(pdf_path),
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def save_three_timepoint_panel(
    images: np.ndarray,
    bearing: str,
    timepoints: list[tuple[str, int]],
    tag: str,
    fpt_index: int,
) -> dict:
    if images.shape != (3, 2, *IMAGE_SHAPE):
        raise ValueError(f"Expected [3,2,128,128], got {images.shape}")
    if len(timepoints) != 3:
        raise ValueError("Expected exactly three timepoints")
    vmin, vmax = robust_limits(images)
    channel_names = ["Horizontal", "Vertical"]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), constrained_layout=True)
    plotted = None
    for col, (state_name, index) in enumerate(timepoints):
        for row, channel_name in enumerate(channel_names):
            axis = axes[row, col]
            plotted = axis.imshow(
                images[col, row],
                aspect="auto",
                origin="upper",
                cmap=COLORMAP,
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                axis.set_title(f"{state_name} | index={index}", fontsize=CHANNEL_TITLE_FONTSIZE)
            configure_cwt_axis(axis, channel_name)
            if col > 0:
                axis.set_ylabel("")
    fig.suptitle(
        f"{bearing}: CWT log-power | frozen FPT={fpt_index}",
        fontsize=TITLE_FONTSIZE + 1,
    )
    fig.colorbar(plotted, ax=axes.ravel().tolist(), label=COLORBAR_LABEL, shrink=0.82)
    png_path = OUT_DIR / f"{tag}.png"
    pdf_path = OUT_DIR / f"{tag}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "kind": "three_timepoint_panel",
        "tag": tag,
        "bearing": bearing,
        "timepoints": [
            {"name": str(name), "index": int(index)}
            for name, index in timepoints
        ],
        "fpt_index": int(fpt_index),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "png_sha256": sha256_file(png_path),
        "pdf_sha256": sha256_file(pdf_path),
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


print("Plot helpers OK")"""
    ),
    code(
        """# 6) Generate requested paper figures
records_by_bearing = {}
labels_by_bearing = {}
manifest_entries = []

required_bearings = (
    {item["bearing"] for item in FIGURE_EXAMPLES}
    | {item["bearing"] for item in SEQUENCE_EXAMPLES}
    | set(THREE_TIMEPOINT_BEARINGS)
)

for bearing in sorted(required_bearings):
    print("Loading", bearing)
    records = load_raw_records(bearing)
    labels = load_and_validate_frozen_labels(bearing, len(records))
    records_by_bearing[bearing] = records
    labels_by_bearing[bearing] = labels
    print("  n_files:", len(records), "fpt:", labels["fpt_index"])

for bearing in THREE_TIMEPOINT_BEARINGS:
    records = records_by_bearing[bearing]
    fpt_index = labels_by_bearing[bearing]["fpt_index"]
    timepoints = three_timepoint_indices(bearing, fpt_index, len(records))
    indices = [index for _, index in timepoints]
    images = compute_two_channel_examples(records, indices)
    entry = save_three_timepoint_panel(
        images,
        bearing,
        timepoints,
        f"{bearing}_three_timepoint_cwt",
        fpt_index,
    )
    manifest_entries.append(entry)
    display(Image(filename=entry["png_path"]))

for item in FIGURE_EXAMPLES:
    bearing = item["bearing"]
    tag = item["tag"]
    records = records_by_bearing[bearing]
    fpt_index = labels_by_bearing[bearing]["fpt_index"]
    index = resolve_figure_index(item, fpt_index, len(records))
    image = cwt_two_channel_record(records[index])
    entry = save_two_channel_panel(image, bearing, index, tag, fpt_index)
    manifest_entries.append(entry)
    display(Image(filename=entry["png_path"]))

for item in SEQUENCE_EXAMPLES:
    bearing = item["bearing"]
    end_index = int(item["end_index"])
    tag = item["tag"]
    records = records_by_bearing[bearing]
    fpt_index = labels_by_bearing[bearing]["fpt_index"]
    validate_index(bearing, end_index, len(records))
    start_index = end_index - SEQUENCE_LENGTH + 1
    if start_index < 0:
        raise ValueError(f"{bearing}: sequence ending at {end_index} needs start >= 0")
    indices = list(range(start_index, end_index + 1))
    images = compute_two_channel_examples(records, indices)
    entry = save_sequence_strip(images, bearing, indices, tag, fpt_index)
    manifest_entries.append(entry)
    display(Image(filename=entry["png_path"]))

print("Generated", len(manifest_entries), "figure artifacts")"""
    ),
    code(
        """# 7) Save manifest and downloadable zip
manifest = {
    "visualization_version": VISUALIZATION_VERSION,
    "purpose": "paper_only_cwt_figures_not_for_training_or_evaluation",
    "cwt_wavelet": CWT_WAVELET,
    "n_scales": int(len(CWT_SCALES)),
    "scale_min": float(CWT_SCALES[0]),
    "scale_max": float(CWT_SCALES[-1]),
    "scale_spacing": "geometric",
    "image_shape": list(IMAGE_SHAPE),
    "log_power_transform": "log2(abs(coef)^2 + 1e-3)",
    "sampling_frequency_hz": SAMPLING_FREQ,
    "data_points_per_file": DATA_POINTS,
    "label_version": LABEL_VERSION,
    "sequence_length": SEQUENCE_LENGTH,
    "raw_dir": str(RAW_DIR),
    "label_dir": str(LABEL_DIR),
    "out_dir": str(OUT_DIR),
    "entries": manifest_entries,
}

manifest_path = OUT_DIR / "cwt_paper_figures_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = Path("/kaggle/working/cwt_paper_figures.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT_DIR.glob("*")):
        if path.is_file():
            archive.write(path, arcname=path.name)

display(pd.DataFrame(manifest_entries))
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
    out = Path(__file__).resolve().parents[2] / "notebooks" / "04_cwt_figures.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out.name} ({len(cells)} cells)")
