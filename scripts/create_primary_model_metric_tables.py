"""Create primary CWT-CNN/CWT-CNN-LSTM metric tables for paper writing.

This script is presentation-only. It reads the existing Task 5 primary run in
``results/04_training_cnn_vs_cnn_lstm`` and writes compact CSV/HTML/Markdown tables. It does not train
models, change predictions, or tune thresholds.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results" / "04_training_cnn_vs_cnn_lstm"
OUT_DIR = ROOT / "results" / "metric_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = range(6)
MODELS = ("cnn", "cnn_lstm")
MODEL_LABEL = {"cnn": "CWT-CNN", "cnn_lstm": "CWT-CNN-LSTM"}
THRESHOLD = 0.5
FPT_CONSECUTIVE = 5


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def round_metric(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def roc_auc_score_rank(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            mean_rank = float(np.mean(np.arange(start + 1, end + 1, dtype=np.float64)))
            ranks[order[start:end]] = mean_rank
        start = end

    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def first_consecutive_index(values: np.ndarray, indices: np.ndarray, threshold: float) -> int | None:
    positive = np.asarray(values, dtype=np.float64) >= float(threshold)
    if len(positive) < FPT_CONSECUTIVE:
        return None
    for start in range(0, len(positive) - FPT_CONSECUTIVE + 1):
        if bool(np.all(positive[start : start + FPT_CONSECUTIVE])):
            return int(indices[start])
    return None


def load_prediction(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {
            "y_true": data["y_true"].astype(np.uint8),
            "y_prob": data["y_prob"].astype(np.float64),
            "target_index": data["target_index"].astype(np.int64),
            "bearing": str(data["bearing_id"][0]),
        }


def sample_metrics_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    auc_pr: float | None,
    roc_auc: float | None,
) -> dict[str, Any]:
    total = tn + fp + fn + tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, total)
    balanced_accuracy = (
        None
        if recall is None or specificity is None
        else (float(recall) + float(specificity)) / 2
    )
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * float(precision) * float(recall) / (float(precision) + float(recall))
    )

    return {
        "n_samples": total,
        "n_negative": tn + fp,
        "n_positive": fn + tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": round_metric(accuracy),
        "balanced_accuracy": round_metric(balanced_accuracy),
        "precision": round_metric(precision),
        "recall": round_metric(recall),
        "specificity": round_metric(specificity),
        "f1": round_metric(f1),
        "auc_pr": round_metric(auc_pr),
        "roc_auc": round_metric(roc_auc),
    }


def event_metrics_from_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    y_true = prediction["y_true"]
    y_prob = prediction["y_prob"]
    target_index = prediction["target_index"]
    positives = target_index[y_true == 1]
    if len(positives) == 0:
        raise ValueError(f"{prediction['bearing']}: no positive labels in prediction artifact")

    reference_fpt = int(positives[0])
    predicted_fpt = first_consecutive_index(y_prob, target_index, THRESHOLD)
    pre_fpt_positive = int(np.sum((target_index < reference_fpt) & (y_prob >= THRESHOLD)))
    if predicted_fpt is None:
        return {
            "reference_fpt": reference_fpt,
            "predicted_fpt": "",
            "signed_delay_files": "",
            "abs_fpt_error_files": "",
            "relative_error_reference_pct": "",
            "pre_fpt_positive_count": pre_fpt_positive,
            "pre_fpt_alarm_run": False,
            "missed_detection": True,
        }

    signed_delay = int(predicted_fpt - reference_fpt)
    abs_error = abs(signed_delay)
    return {
        "reference_fpt": reference_fpt,
        "predicted_fpt": predicted_fpt,
        "signed_delay_files": signed_delay,
        "abs_fpt_error_files": abs_error,
        "relative_error_reference_pct": round_metric(abs_error / reference_fpt * 100, 3),
        "pre_fpt_positive_count": pre_fpt_positive,
        "pre_fpt_alarm_run": bool(predicted_fpt < reference_fpt),
        "missed_detection": False,
    }


def event_metrics_from_json(metrics: dict[str, Any]) -> dict[str, Any]:
    event = metrics["event_metrics"]
    reference_fpt = int(event["reference_fpt"])
    abs_error = event.get("abs_fpt_error_files")
    return {
        "reference_fpt": reference_fpt,
        "predicted_fpt": "" if event.get("predicted_fpt") is None else int(event["predicted_fpt"]),
        "signed_delay_files": "" if event.get("signed_delay_files") is None else int(event["signed_delay_files"]),
        "abs_fpt_error_files": "" if abs_error is None else int(abs_error),
        "relative_error_reference_pct": ""
        if abs_error is None
        else round_metric(float(abs_error) / reference_fpt * 100, 3),
        "pre_fpt_positive_count": int(event["pre_fpt_positive_sample_count"]),
        "pre_fpt_alarm_run": bool(event["pre_fpt_alarm_run"]),
        "missed_detection": bool(event["missed_detection"]),
    }


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    sample_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for fold in FOLDS:
        for model in MODELS:
            stem = f"fold_{fold}_{model}"
            prediction_path = SOURCE_DIR / f"{stem}_predictions.npz"
            metrics_path = SOURCE_DIR / f"{stem}_metrics.json"
            confusion_path = SOURCE_DIR / f"{stem}_confusion_matrix.json"

            prediction = load_prediction(prediction_path)
            metrics_json = read_json(metrics_path)
            confusion_json = read_json(confusion_path)

            if prediction is None and metrics_json is None:
                warnings.append(f"{stem}: missing both prediction and metric JSON; skipped")
                continue

            if metrics_json is not None:
                bearing = str(metrics_json["outer_test_bearing"])
            elif prediction is not None:
                bearing = str(prediction["bearing"])
            else:
                bearing = ""

            if prediction is None:
                warnings.append(f"{stem}: missing predictions.npz; used saved metric/confusion JSON only")

            auc_pr: float | None = None
            roc_auc: float | None = None
            if metrics_json is not None:
                sample_json = metrics_json.get("sample_metrics", {})
                auc_pr = sample_json.get("auc_pr")
                roc_auc = sample_json.get("roc_auc")
            elif prediction is not None:
                auc_pr = average_precision(prediction["y_true"], prediction["y_prob"])
                roc_auc = roc_auc_score_rank(prediction["y_true"], prediction["y_prob"])

            if confusion_json is not None:
                tn = int(confusion_json["tn"])
                fp = int(confusion_json["fp"])
                fn = int(confusion_json["fn"])
                tp = int(confusion_json["tp"])
                confusion_source = "confusion_json"
            elif prediction is not None:
                y_pred = prediction["y_prob"] >= THRESHOLD
                y_true = prediction["y_true"].astype(bool)
                tn = int(np.sum(~y_true & ~y_pred))
                fp = int(np.sum(~y_true & y_pred))
                fn = int(np.sum(y_true & ~y_pred))
                tp = int(np.sum(y_true & y_pred))
                confusion_source = "prediction_npz"
            else:
                raise ValueError(f"{stem}: cannot compute sample metrics")

            base = {
                "analysis": "primary_fixed_threshold_p_0_5",
                "fold": fold,
                "outer_test_bearing": bearing,
                "model": model,
                "model_label": MODEL_LABEL[model],
                "threshold": THRESHOLD,
            }
            sample_rows.append(
                {
                    **base,
                    **sample_metrics_from_counts(tn, fp, fn, tp, auc_pr, roc_auc),
                    "prediction_source": "prediction_npz" if prediction is not None else "missing_prediction",
                    "metric_source": "metrics_json" if metrics_json is not None else "computed_from_prediction",
                    "confusion_source": confusion_source,
                }
            )

            if prediction is not None:
                event_metrics = event_metrics_from_prediction(prediction)
                event_source = "prediction_npz"
            elif metrics_json is not None:
                event_metrics = event_metrics_from_json(metrics_json)
                event_source = "metrics_json"
            else:
                continue

            event_rows.append({**base, **event_metrics, "event_source": event_source})

    sample_rows.sort(key=lambda row: (int(row["fold"]), str(row["model"])))
    event_rows.sort(key=lambda row: (int(row["fold"]), str(row["model"])))
    return sample_rows, event_rows, warnings


def summary_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": "", "median": "", "std": ""}
    return {
        "mean": round_metric(float(np.mean(array)), 6),
        "median": round_metric(float(np.median(array)), 6),
        "std": round_metric(float(np.std(array, ddof=0)), 6),
    }


def summarize(sample_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        samples = [row for row in sample_rows if row["model"] == model]
        events = [row for row in event_rows if row["model"] == model]
        row: dict[str, Any] = {
            "analysis": "primary_fixed_threshold_p_0_5",
            "model": model,
            "model_label": MODEL_LABEL[model],
            "n_folds": len(samples),
        }
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc_pr",
            "roc_auc",
        ]:
            stats = summary_stats([float(sample[metric]) for sample in samples if sample[metric] != ""])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_median"] = stats["median"]
            row[f"{metric}_std"] = stats["std"]

        abs_stats = summary_stats(
            [float(event["abs_fpt_error_files"]) for event in events if event["abs_fpt_error_files"] != ""]
        )
        delay_stats = summary_stats(
            [float(event["signed_delay_files"]) for event in events if event["signed_delay_files"] != ""]
        )
        row["abs_fpt_error_mean"] = abs_stats["mean"]
        row["abs_fpt_error_median"] = abs_stats["median"]
        row["abs_fpt_error_std"] = abs_stats["std"]
        row["signed_delay_mean"] = delay_stats["mean"]
        row["signed_delay_median"] = delay_stats["median"]
        row["signed_delay_std"] = delay_stats["std"]
        row["missed_detections"] = int(sum(bool(event["missed_detection"]) for event in events))
        row["pre_fpt_alarm_runs"] = int(sum(bool(event["pre_fpt_alarm_run"]) for event in events))
        rows.append(row)
    return rows


def write_html(path: Path, title: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:'Times New Roman',serif;margin:24px;color:#111827;}",
        "table{border-collapse:collapse;font-size:13px;}",
        "th,td{border:1px solid #CBD5E1;padding:5px 7px;text-align:center;}",
        "th{background:#E5E7EB;font-weight:bold;}",
        "td.left{text-align:left;}",
        "</style></head><body>",
        f"<h2>{html.escape(title)}</h2>",
        "<table><thead><tr>",
    ]
    for field in fieldnames:
        lines.append(f"<th>{html.escape(field)}</th>")
    lines.append("</tr></thead><tbody>")
    for row in rows:
        lines.append("<tr>")
        for field in fieldnames:
            lines.append(f"<td>{html.escape(str(row.get(field, '')))}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table>", "</body></html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(summary_rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# Primary model metric tables",
        "",
        "These tables are derived from existing `results/04_training_cnn_vs_cnn_lstm` artifacts only.",
        "No model was retrained, no prediction was edited, and the fixed threshold remains `P=0.5`.",
        "",
        "## Recommended use",
        "",
        "- Use event-level FPT metrics as the primary evidence.",
        "- Use sample-level metrics as supporting classification evidence.",
        "- Treat accuracy as secondary because healthy/degraded samples are imbalanced.",
        "",
        "## Aggregate summary",
        "",
        "| Model | Mean abs. FPT error | Median abs. FPT error | Mean F1 | Mean recall | Mean balanced acc. | Mean AUC-PR | Mean accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model_label} | {abs_fpt_error_mean} | {abs_fpt_error_median} | "
            "{f1_mean} | {recall_mean} | {balanced_accuracy_mean} | {auc_pr_mean} | {accuracy_mean} |".format(**row)
        )

    if warnings:
        lines.extend(["", "## Artifact warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Leakage note",
            "",
            "All metrics are computed from frozen outer-test artifacts after model selection.",
            "No threshold calibration or model selection is performed in this script.",
        ]
    )
    (OUT_DIR / "primary_model_metric_tables_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sample_rows, event_rows, warnings = collect_rows()
    summary_rows = summarize(sample_rows, event_rows)

    sample_fields = [
        "analysis",
        "fold",
        "outer_test_bearing",
        "model_label",
        "threshold",
        "n_samples",
        "n_negative",
        "n_positive",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc_pr",
        "roc_auc",
        "prediction_source",
        "metric_source",
        "confusion_source",
    ]
    event_fields = [
        "analysis",
        "fold",
        "outer_test_bearing",
        "model_label",
        "threshold",
        "reference_fpt",
        "predicted_fpt",
        "signed_delay_files",
        "abs_fpt_error_files",
        "relative_error_reference_pct",
        "pre_fpt_positive_count",
        "pre_fpt_alarm_run",
        "missed_detection",
        "event_source",
    ]
    summary_fields = [
        "analysis",
        "model_label",
        "n_folds",
        "accuracy_mean",
        "accuracy_median",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_median",
        "balanced_accuracy_std",
        "precision_mean",
        "precision_median",
        "precision_std",
        "recall_mean",
        "recall_median",
        "recall_std",
        "specificity_mean",
        "specificity_median",
        "specificity_std",
        "f1_mean",
        "f1_median",
        "f1_std",
        "auc_pr_mean",
        "auc_pr_median",
        "auc_pr_std",
        "roc_auc_mean",
        "roc_auc_median",
        "roc_auc_std",
        "abs_fpt_error_mean",
        "abs_fpt_error_median",
        "abs_fpt_error_std",
        "signed_delay_mean",
        "signed_delay_median",
        "signed_delay_std",
        "missed_detections",
        "pre_fpt_alarm_runs",
    ]

    write_csv(OUT_DIR / "primary_sample_level_metrics_by_fold.csv", sample_rows, sample_fields)
    write_csv(OUT_DIR / "primary_event_level_metrics_by_fold.csv", event_rows, event_fields)
    write_csv(OUT_DIR / "primary_model_metric_summary.csv", summary_rows, summary_fields)
    write_html(
        OUT_DIR / "primary_sample_level_metrics_by_fold.html",
        "Primary sample-level metrics by fold",
        sample_rows,
        sample_fields,
    )
    write_html(
        OUT_DIR / "primary_event_level_metrics_by_fold.html",
        "Primary event-level FPT metrics by fold",
        event_rows,
        event_fields,
    )
    write_html(
        OUT_DIR / "primary_model_metric_summary.html",
        "Primary model metric summary",
        summary_rows,
        summary_fields,
    )
    write_markdown(summary_rows, warnings)

    manifest = {
        "purpose": "primary_model_metric_tables_for_paper",
        "source_dir": SOURCE_DIR.relative_to(ROOT).as_posix(),
        "output_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "analysis": "primary_fixed_threshold_p_0_5",
        "threshold": THRESHOLD,
        "no_retraining": True,
        "no_prediction_editing": True,
        "sample_rows": len(sample_rows),
        "event_rows": len(event_rows),
        "summary_rows": len(summary_rows),
        "warnings": warnings,
        "generated_files": sorted(path.name for path in OUT_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "primary_model_metric_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
