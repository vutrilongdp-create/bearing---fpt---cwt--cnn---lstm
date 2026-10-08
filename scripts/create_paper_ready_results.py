"""Create paper-ready tables and figures from frozen FPT result artifacts.

This script is presentation-only. It reads existing Task 5/6 outputs and writes
derived CSV/HTML/PNG/PDF files for the paper. It does not retrain models, change
thresholds, or modify the original result artifacts.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DIR = ROOT / "results" / "04_training_cnn_vs_cnn_lstm"
SECONDARY_DIR = ROOT / "results" / "05_threshold_calibration"
OUT_DIR = ROOT / "results" / "paper_ready_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPT_CONSECUTIVE = 5
FIXED_THRESHOLD = 0.5
MODELS = ["cnn", "cnn_lstm"]
MODEL_LABEL = {"cnn": "CWT-CNN", "cnn_lstm": "CWT-CNN-LSTM"}
MODEL_COLOR = {"cnn": "#2563EB", "cnn_lstm": "#F97316"}
WIN_COLOR = "#DCFCE7"
LOSE_COLOR = "#FFFFFF"
TIE_COLOR = "#FEF3C7"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def first_consecutive_index(values: np.ndarray, indices: np.ndarray, threshold: float) -> int | None:
    positive = np.asarray(values, dtype=np.float64) >= float(threshold)
    if len(positive) < FPT_CONSECUTIVE:
        return None
    for start in range(0, len(positive) - FPT_CONSECUTIVE + 1):
        if bool(np.all(positive[start : start + FPT_CONSECUTIVE])):
            return int(indices[start])
    return None


def event_metrics_from_predictions(path: Path, threshold: float) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        y_true = data["y_true"].astype(np.uint8)
        y_prob = data["y_prob"].astype(np.float64)
        target_index = data["target_index"].astype(np.int64)
        bearing_id = str(data["bearing_id"][0])

    positive_targets = target_index[y_true == 1]
    if len(positive_targets) == 0:
        raise ValueError(f"{path.name}: no positive labels found")
    reference_fpt = int(positive_targets[0])
    predicted_fpt = first_consecutive_index(y_prob, target_index, threshold)
    pre_fpt_positive_sample_count = int(np.sum((target_index < reference_fpt) & (y_prob >= threshold)))

    if predicted_fpt is None:
        signed_delay = None
        abs_error = None
        missed = True
        pre_fpt_alarm_run = False
    else:
        signed_delay = int(predicted_fpt - reference_fpt)
        abs_error = abs(signed_delay)
        missed = False
        pre_fpt_alarm_run = bool(predicted_fpt < reference_fpt)

    return {
        "bearing": bearing_id,
        "reference_fpt": reference_fpt,
        "predicted_fpt": "" if predicted_fpt is None else predicted_fpt,
        "signed_delay": "" if signed_delay is None else signed_delay,
        "abs_error": "" if abs_error is None else abs_error,
        "pre_fpt_positive_count": pre_fpt_positive_sample_count,
        "pre_fpt_alarm_run": pre_fpt_alarm_run,
        "missed": missed,
    }


def load_primary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in range(6):
        for model in MODELS:
            stem = f"fold_{fold}_{model}"
            prediction_path = PRIMARY_DIR / f"{stem}_predictions.npz"
            metric_path = PRIMARY_DIR / f"{stem}_metrics.json"
            if prediction_path.exists():
                metrics = event_metrics_from_predictions(prediction_path, FIXED_THRESHOLD)
            elif metric_path.exists():
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
                event = payload["event_metrics"]
                metrics = {
                    "bearing": payload["outer_test_bearing"],
                    "reference_fpt": int(event["reference_fpt"]),
                    "predicted_fpt": int(event["predicted_fpt"]) if event["predicted_fpt"] is not None else "",
                    "signed_delay": int(event["signed_delay_files"]) if event["signed_delay_files"] is not None else "",
                    "abs_error": int(event["abs_fpt_error_files"]) if event["abs_fpt_error_files"] is not None else "",
                    "pre_fpt_positive_count": int(event["pre_fpt_positive_sample_count"]),
                    "pre_fpt_alarm_run": bool(event["pre_fpt_alarm_run"]),
                    "missed": bool(event["missed_detection"]),
                }
            else:
                continue
            rows.append(
                {
                    "analysis": "primary_fixed_threshold_p_0_5",
                    "fold": fold,
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "threshold": FIXED_THRESHOLD,
                    **metrics,
                }
            )
    return sorted(rows, key=lambda row: (int(row["fold"]), str(row["model"])))


def load_secondary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(SECONDARY_DIR.glob("fold_*_*metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["model"])
        event = payload["event_metrics"]
        rows.append(
            {
                "analysis": "secondary_inner_val_calibrated",
                "fold": int(payload["fold_index"]),
                "model": model,
                "model_label": MODEL_LABEL[model],
                "threshold": float(payload["selected_threshold"]),
                "bearing": payload["outer_test_bearing"],
                "reference_fpt": int(event["reference_fpt"]),
                "predicted_fpt": int(event["predicted_fpt"]) if event["predicted_fpt"] is not None else "",
                "signed_delay": int(event["signed_delay_files"]) if event["signed_delay_files"] is not None else "",
                "abs_error": int(event["abs_fpt_error_files"]) if event["abs_fpt_error_files"] is not None else "",
                "pre_fpt_positive_count": int(event["pre_fpt_positive_sample_count"]),
                "pre_fpt_alarm_run": bool(event["pre_fpt_alarm_run"]),
                "missed": bool(event["missed_detection"]),
            }
        )
    return sorted(rows, key=lambda row: (int(row["fold"]), str(row["model"])))


def add_best_model(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_bearing: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_bearing.setdefault(str(row["bearing"]), {})[str(row["model"])] = row

    best_by_bearing: dict[str, str] = {}
    for bearing, model_rows in by_bearing.items():
        errors = {
            model: float(row["abs_error"])
            for model, row in model_rows.items()
            if row["abs_error"] != ""
        }
        if len(errors) < 2:
            best_by_bearing[bearing] = ""
        elif errors["cnn"] < errors["cnn_lstm"]:
            best_by_bearing[bearing] = "cnn"
        elif errors["cnn_lstm"] < errors["cnn"]:
            best_by_bearing[bearing] = "cnn_lstm"
        else:
            best_by_bearing[bearing] = "tie"

    result = []
    for row in rows:
        enriched = dict(row)
        best = best_by_bearing.get(str(row["bearing"]), "")
        enriched["best_model_by_abs_error"] = best
        enriched["is_best_for_bearing"] = bool(best in (str(row["model"]), "tie"))
        result.append(enriched)
    return result


def paired_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_bearing: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_bearing.setdefault(str(row["bearing"]), {})[str(row["model"])] = row

    result = []
    for bearing in sorted(by_bearing):
        cnn = by_bearing[bearing].get("cnn")
        lstm = by_bearing[bearing].get("cnn_lstm")
        if cnn is None or lstm is None:
            continue
        cnn_error = float(cnn["abs_error"])
        lstm_error = float(lstm["abs_error"])
        if cnn_error < lstm_error:
            best = "CWT-CNN"
        elif lstm_error < cnn_error:
            best = "CWT-CNN-LSTM"
        else:
            best = "Tie"
        result.append(
            {
                "bearing": bearing,
                "reference_fpt": cnn["reference_fpt"],
                "cnn_predicted_fpt": cnn["predicted_fpt"],
                "cnn_abs_error": cnn["abs_error"],
                "cnn_signed_delay": cnn["signed_delay"],
                "cnn_lstm_predicted_fpt": lstm["predicted_fpt"],
                "cnn_lstm_abs_error": lstm["abs_error"],
                "cnn_lstm_signed_delay": lstm["signed_delay"],
                "best_model_by_abs_error": best,
            }
        )
    return result


def aggregate_rows(rows: list[dict[str, object]], analysis: str) -> list[dict[str, object]]:
    result = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model and row["abs_error"] != ""]
        abs_errors = np.array([float(row["abs_error"]) for row in model_rows], dtype=np.float64)
        delays = np.array([float(row["signed_delay"]) for row in model_rows], dtype=np.float64)
        result.append(
            {
                "analysis": analysis,
                "model": model,
                "model_label": MODEL_LABEL[model],
                "n_folds": len(model_rows),
                "abs_error_mean": round(float(np.mean(abs_errors)), 3),
                "abs_error_median": round(float(np.median(abs_errors)), 3),
                "abs_error_std": round(float(np.std(abs_errors, ddof=0)), 3),
                "signed_delay_mean": round(float(np.mean(delays)), 3),
                "signed_delay_median": round(float(np.median(delays)), 3),
                "signed_delay_std": round(float(np.std(delays, ddof=0)), 3),
                "pre_fpt_alarm_runs": int(sum(bool(row["pre_fpt_alarm_run"]) for row in model_rows)),
                "missed_detections": int(sum(bool(row["missed"]) for row in model_rows)),
            }
        )
    return result


def html_table(path: Path, title: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:'Times New Roman',serif;margin:24px;color:#111827;}",
        "table{border-collapse:collapse;font-size:14px;}",
        "th,td{border:1px solid #CBD5E1;padding:6px 8px;text-align:center;}",
        "th{background:#E5E7EB;font-weight:bold;}",
        ".best{background:#DCFCE7;font-weight:bold;}",
        ".tie{background:#FEF3C7;font-weight:bold;}",
        ".left{text-align:left;}",
        "</style></head><body>",
        f"<h2>{html.escape(title)}</h2>",
        "<table><thead><tr>",
    ]
    for name in fieldnames:
        lines.append(f"<th>{html.escape(name)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>")
        for name in fieldnames:
            value = row.get(name, "")
            css = ""
            if name == "model_label":
                best = row.get("best_model_by_abs_error")
                model = row.get("model")
                if best == "tie":
                    css = " class='tie'"
                elif best == model:
                    css = " class='best'"
            if name == "best_model_by_abs_error":
                css = " class='best'"
            lines.append(f"<td{css}>{html.escape(str(value))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table>", "</body></html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_text(x: float, y: float, value: object, size: int = 14, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' "
        f"font-family='Times New Roman, Times, serif' font-size='{size}' "
        f"font-weight='{weight}' fill='#111827'>{html.escape(str(value))}</text>"
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = "#64748B", width: float = 1.2) -> str:
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
        f"stroke='{color}' stroke-width='{width}' />"
    )


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def draw_axes(
    body: list[str],
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    ymax: float,
    y_label: str,
    x_label: str,
) -> None:
    bottom = top + plot_height
    right = left + plot_width
    body.append(svg_line(left, top, left, bottom, "#334155", 1.4))
    body.append(svg_line(left, bottom, right, bottom, "#334155", 1.4))
    ticks = 5
    for i in range(ticks + 1):
        value = ymax * i / ticks
        y = bottom - plot_height * i / ticks
        body.append(svg_line(left - 5, y, right, y, "#CBD5E1", 0.8 if i else 1.2))
        body.append(svg_text(left - 10, y + 4, f"{value:.0f}", 11, "end"))
    body.append(svg_text(left + plot_width / 2, bottom + 70, x_label, 13, "middle"))
    body.append(
        f"<text x='{left - 58:.1f}' y='{top + plot_height / 2:.1f}' "
        "text-anchor='middle' font-family='Times New Roman, Times, serif' "
        "font-size='13' fill='#111827' transform='rotate(-90 "
        f"{left - 58:.1f} {top + plot_height / 2:.1f})'>{html.escape(y_label)}</text>"
    )


def plot_bearing_abs_errors(rows: list[dict[str, object]], analysis: str, path_stem: str) -> None:
    paired = paired_rows(rows)
    bearings = [row["bearing"] for row in paired]
    cnn_errors = [float(row["cnn_abs_error"]) for row in paired]
    lstm_errors = [float(row["cnn_lstm_abs_error"]) for row in paired]
    ymax = max(cnn_errors + lstm_errors) * 1.15 or 1.0

    width, height = 940, 500
    left, top, plot_width, plot_height = 92, 70, 790, 310
    bottom = top + plot_height
    body: list[str] = []
    body.append(svg_text(width / 2, 30, analysis, 18, "middle", "bold"))
    draw_axes(body, left, top, plot_width, plot_height, ymax, "Absolute FPT error (files)", "Outer-test bearing")

    group_width = plot_width / len(bearings)
    bar_width = group_width * 0.28
    for i, row in enumerate(paired):
        center = left + group_width * (i + 0.5)
        values = [float(row["cnn_abs_error"]), float(row["cnn_lstm_abs_error"])]
        xs = [center - bar_width * 0.6, center + bar_width * 0.6]
        models = ["CWT-CNN", "CWT-CNN-LSTM"]
        colors = [MODEL_COLOR["cnn"], MODEL_COLOR["cnn_lstm"]]
        for x, value, model_label, color in zip(xs, values, models, colors):
            bar_h = plot_height * value / ymax
            y = bottom - bar_h
            stroke = "#111827" if row["best_model_by_abs_error"] == model_label else "none"
            stroke_width = 2.0 if stroke != "none" else 0.0
            body.append(
                f"<rect x='{x - bar_width / 2:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_h:.1f}' "
                f"fill='{color}' stroke='{stroke}' stroke-width='{stroke_width}' />"
            )
        body.append(
            f"<text x='{center:.1f}' y='{bottom + 26:.1f}' text-anchor='end' "
            "font-family='Times New Roman, Times, serif' font-size='11' "
            f"fill='#111827' transform='rotate(-22 {center:.1f} {bottom + 26:.1f})'>{html.escape(bearings[i])}</text>"
        )

    legend_y = 440
    body.append(f"<rect x='310' y='{legend_y - 13}' width='14' height='14' fill='{MODEL_COLOR['cnn']}' />")
    body.append(svg_text(330, legend_y, MODEL_LABEL["cnn"], 12, "start"))
    body.append(f"<rect x='455' y='{legend_y - 13}' width='14' height='14' fill='{MODEL_COLOR['cnn_lstm']}' />")
    body.append(svg_text(475, legend_y, MODEL_LABEL["cnn_lstm"], 12, "start"))
    body.append(f"<rect x='650' y='{legend_y - 14}' width='16' height='16' fill='white' stroke='#111827' stroke-width='2' />")
    body.append(svg_text(672, legend_y, "lower error per bearing", 12, "start"))
    save_svg(OUT_DIR / f"{path_stem}.svg", width, height, body)


def plot_summary_errorbar(primary_summary: list[dict[str, object]], secondary_summary: list[dict[str, object]]) -> None:
    groups = [
        ("Primary: fixed P=0.5", primary_summary),
        ("Secondary: inner-val calibrated", secondary_summary),
    ]
    all_tops = [
        float(row["abs_error_mean"]) + float(row["abs_error_std"])
        for _, summary in groups
        for row in summary
    ]
    ymax = max(all_tops) * 1.2 or 1.0

    width, height = 900, 470
    left, top, plot_width, plot_height = 95, 70, 720, 285
    bottom = top + plot_height
    body: list[str] = []
    body.append(svg_text(width / 2, 30, "Mean absolute FPT error with standard deviation", 18, "middle", "bold"))
    draw_axes(body, left, top, plot_width, plot_height, ymax, "Mean absolute FPT error (files)", "Analysis")

    centers = [left + plot_width * 0.28, left + plot_width * 0.72]
    bar_width = 58
    for group_center, (group_title, summary) in zip(centers, groups):
        offsets = [-40, 40]
        for offset, row in zip(offsets, summary):
            model = str(row["model"])
            mean = float(row["abs_error_mean"])
            std = float(row["abs_error_std"])
            x = group_center + offset
            bar_h = plot_height * mean / ymax
            y = bottom - bar_h
            body.append(
                f"<rect x='{x - bar_width / 2:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_h:.1f}' "
                f"fill='{MODEL_COLOR[model]}' />"
            )
            err_top = bottom - plot_height * (mean + std) / ymax
            err_bottom = bottom - plot_height * max(mean - std, 0.0) / ymax
            body.append(svg_line(x, err_top, x, err_bottom, "#111827", 1.6))
            body.append(svg_line(x - 11, err_top, x + 11, err_top, "#111827", 1.6))
            body.append(svg_line(x - 11, err_bottom, x + 11, err_bottom, "#111827", 1.6))
        body.append(svg_text(group_center, bottom + 35, group_title, 12, "middle"))

    legend_y = 425
    body.append(f"<rect x='310' y='{legend_y - 13}' width='14' height='14' fill='{MODEL_COLOR['cnn']}' />")
    body.append(svg_text(330, legend_y, MODEL_LABEL["cnn"], 12, "start"))
    body.append(f"<rect x='455' y='{legend_y - 13}' width='14' height='14' fill='{MODEL_COLOR['cnn_lstm']}' />")
    body.append(svg_text(475, legend_y, MODEL_LABEL["cnn_lstm"], 12, "start"))
    body.append(svg_line(650, legend_y - 12, 650, legend_y + 12, "#111827", 1.6))
    body.append(svg_line(639, legend_y - 12, 661, legend_y - 12, "#111827", 1.6))
    body.append(svg_line(639, legend_y + 12, 661, legend_y + 12, "#111827", 1.6))
    body.append(svg_text(670, legend_y + 4, "standard deviation", 12, "start"))
    save_svg(OUT_DIR / "primary_secondary_abs_error_errorbar.svg", width, height, body)


def write_markdown_summary(
    primary_summary: list[dict[str, object]],
    secondary_summary: list[dict[str, object]],
) -> None:
    lines = [
        "# Paper-ready FPT result presentation",
        "",
        "This folder contains presentation-only tables and figures derived from existing result artifacts.",
        "No model was retrained, no prediction was edited, and no outer-test threshold tuning was performed.",
        "",
        "## Primary result",
        "",
        "Primary analysis uses the fixed probability threshold P=0.5 for both models.",
        "",
        "## Secondary analysis",
        "",
        "Secondary analysis uses thresholds selected from the inner-validation bearing only, then frozen before outer-test evaluation.",
        "",
        "## Key aggregate metrics",
        "",
        "| Analysis | Model | Mean abs. error | Median abs. error | Std abs. error | Missed | Pre-FPT alarm runs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary_summary + secondary_summary:
        lines.append(
            "| {analysis} | {model_label} | {abs_error_mean} | {abs_error_median} | "
            "{abs_error_std} | {missed_detections} | {pre_fpt_alarm_runs} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Recommended paper framing",
            "",
            "- Report the fixed P=0.5 results as the main comparison.",
            "- Report inner-validation threshold calibration as a secondary sensitivity analysis.",
            "- Use event-level FPT error before sample-level metrics when discussing model quality.",
            "- Highlight early false alarms explicitly instead of hiding difficult bearings.",
        ]
    )
    (OUT_DIR / "paper_ready_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    primary_rows = add_best_model(load_primary_rows())
    secondary_rows = add_best_model(load_secondary_rows())
    primary_summary = aggregate_rows(primary_rows, "primary_fixed_threshold_p_0_5")
    secondary_summary = aggregate_rows(secondary_rows, "secondary_inner_val_calibrated")

    detail_fields = [
        "analysis",
        "fold",
        "bearing",
        "model_label",
        "threshold",
        "reference_fpt",
        "predicted_fpt",
        "signed_delay",
        "abs_error",
        "pre_fpt_positive_count",
        "pre_fpt_alarm_run",
        "missed",
        "best_model_by_abs_error",
    ]
    paired_fields = [
        "bearing",
        "reference_fpt",
        "cnn_predicted_fpt",
        "cnn_abs_error",
        "cnn_signed_delay",
        "cnn_lstm_predicted_fpt",
        "cnn_lstm_abs_error",
        "cnn_lstm_signed_delay",
        "best_model_by_abs_error",
    ]
    summary_fields = [
        "analysis",
        "model_label",
        "n_folds",
        "abs_error_mean",
        "abs_error_median",
        "abs_error_std",
        "signed_delay_mean",
        "signed_delay_median",
        "signed_delay_std",
        "pre_fpt_alarm_runs",
        "missed_detections",
    ]

    write_csv(OUT_DIR / "primary_event_detail.csv", primary_rows, detail_fields)
    write_csv(OUT_DIR / "primary_event_paired_by_bearing.csv", paired_rows(primary_rows), paired_fields)
    write_csv(OUT_DIR / "primary_summary_mean_median_std.csv", primary_summary, summary_fields)
    write_csv(OUT_DIR / "secondary_event_detail.csv", secondary_rows, detail_fields)
    write_csv(OUT_DIR / "secondary_event_paired_by_bearing.csv", paired_rows(secondary_rows), paired_fields)
    write_csv(OUT_DIR / "secondary_summary_mean_median_std.csv", secondary_summary, summary_fields)
    write_csv(
        OUT_DIR / "combined_summary_mean_median_std.csv",
        primary_summary + secondary_summary,
        summary_fields,
    )

    html_table(
        OUT_DIR / "primary_event_detail_highlight.html",
        "Primary fixed-threshold event metrics (P=0.5)",
        primary_rows,
        detail_fields,
    )
    html_table(
        OUT_DIR / "secondary_event_detail_highlight.html",
        "Secondary inner-validation calibrated event metrics",
        secondary_rows,
        detail_fields,
    )

    plot_bearing_abs_errors(primary_rows, "Primary fixed-threshold FPT error", "primary_abs_error_by_bearing")
    plot_bearing_abs_errors(secondary_rows, "Secondary calibrated-threshold FPT error", "secondary_abs_error_by_bearing")
    plot_summary_errorbar(primary_summary, secondary_summary)
    write_markdown_summary(primary_summary, secondary_summary)

    manifest = {
        "purpose": "paper_ready_result_presentation_only",
        "primary_source": PRIMARY_DIR.relative_to(ROOT).as_posix(),
        "secondary_source": SECONDARY_DIR.relative_to(ROOT).as_posix(),
        "output_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "primary_rule": "fixed probability threshold P=0.5",
        "secondary_rule": "threshold selected from inner-validation only",
        "no_retraining": True,
        "no_original_result_modification": True,
        "generated_files": sorted(path.name for path in OUT_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "paper_ready_results_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
