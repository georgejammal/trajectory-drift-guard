#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from security_drift_common import write_json


POSITIVE = "failed_refusal"
NEGATIVE = "successful_refusal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select concrete drift/no-drift thresholds for heldout drift scores.")
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--layer-trace-csv", type=Path, default=None)
    parser.add_argument("--metric", default=None, help="Feature metric, e.g. cosine or dot.")
    parser.add_argument("--score-name", required=True)
    parser.add_argument("--layer", type=int, default=None, help="Required for layer-trace mode.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def load_scores(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.features_csv) == bool(args.layer_trace_csv):
        raise ValueError("Pass exactly one of --features-csv or --layer-trace-csv.")
    source = args.features_csv or args.layer_trace_csv
    rows = read_csv(source)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("drift_outcome") not in {POSITIVE, NEGATIVE}:
            continue
        if args.features_csv:
            if args.metric is not None and row.get("metric") != args.metric:
                continue
            if args.score_name not in row:
                raise KeyError(f"Missing score column {args.score_name}")
            score = row[args.score_name]
            descriptor = {"mode": "feature", "metric": row.get("metric"), "score_name": args.score_name}
        else:
            if args.layer is None:
                raise ValueError("--layer is required with --layer-trace-csv.")
            if int(row["layer"]) != args.layer:
                continue
            if args.score_name not in row:
                raise KeyError(f"Missing score column {args.score_name}")
            score = row[args.score_name]
            descriptor = {"mode": "layer", "layer": args.layer, "score_name": args.score_name}
        if score in {"", "None", None}:
            continue
        out.append(
            {
                "sample_id": row["sample_id"],
                "benchmark": row.get("benchmark"),
                "dataset_index": row.get("dataset_index"),
                "drift_outcome": row.get("drift_outcome"),
                "score": float(score),
                **descriptor,
            }
        )
    if not out:
        raise RuntimeError("No score rows selected.")
    return out


def candidate_thresholds(scores: list[float]) -> list[float]:
    values = sorted(set(scores))
    if len(values) == 1:
        eps = max(abs(values[0]), 1.0) * 1e-6
        return [values[0] - eps, values[0] + eps]
    thresholds = [values[0] - (values[1] - values[0]) * 0.5]
    thresholds.extend((lo + hi) * 0.5 for lo, hi in zip(values, values[1:]))
    thresholds.append(values[-1] + (values[-1] - values[-2]) * 0.5)
    return thresholds


def confusion(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = float(row["score"]) >= threshold
        gold = row["drift_outcome"] == POSITIVE
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "failed_n": tp + fn,
        "success_n": tn + fp,
        "predicted_drift_n": tp + fp,
        "predicted_no_drift_n": tn + fn,
        "tpr": tpr,
        "fpr": fpr,
        "specificity": tnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": tpr - fpr,
    }


def select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = candidate_thresholds([float(row["score"]) for row in rows])
    scored = [confusion(rows, threshold) for threshold in candidates]
    return sorted(
        scored,
        key=lambda row: (
            -float(row["youden_j"]),
            -float(row["f1"]),
            -float(row["precision"]),
            -float(row["threshold"]),
        ),
    )[0]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def describe_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for outcome in [POSITIVE, NEGATIVE]:
        values = [float(row["score"]) for row in rows if row["drift_outcome"] == outcome]
        out[outcome] = {
            "n": len(values),
            "mean": mean(values),
            "p25": percentile(values, 0.25),
            "p50": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
        }
    return out


def cross_validate(rows: list[dict[str, Any]], folds: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [row for row in rows if fold_id(str(row["sample_id"]), folds) != fold]
        valid = [row for row in rows if fold_id(str(row["sample_id"]), folds) == fold]
        if not train or not valid:
            continue
        selected = select_threshold(train)
        valid_metrics = confusion(valid, float(selected["threshold"]))
        fold_rows.append(
            {
                "fold": fold,
                "threshold": selected["threshold"],
                "train_balanced_accuracy": selected["balanced_accuracy"],
                "train_youden_j": selected["youden_j"],
                "valid_balanced_accuracy": valid_metrics["balanced_accuracy"],
                "valid_youden_j": valid_metrics["youden_j"],
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_specificity": valid_metrics["specificity"],
                "valid_tp": valid_metrics["tp"],
                "valid_fp": valid_metrics["fp"],
                "valid_tn": valid_metrics["tn"],
                "valid_fn": valid_metrics["fn"],
            }
        )
        for row in valid:
            pred = float(row["score"]) >= float(selected["threshold"])
            predictions.append(
                {
                    **row,
                    "fold": fold,
                    "threshold": selected["threshold"],
                    "predicted_drift": int(pred),
                }
            )
    aggregate = confusion_from_predictions(predictions)
    aggregate.pop("threshold", None)
    aggregate["evaluation"] = "out-of-fold predictions; each validation fold uses its own train-selected threshold"
    aggregate["threshold_mean"] = mean([float(row["threshold"]) for row in fold_rows])
    aggregate["threshold_median"] = percentile([float(row["threshold"]) for row in fold_rows], 0.5)
    aggregate["threshold_min"] = min((float(row["threshold"]) for row in fold_rows), default=None)
    aggregate["threshold_max"] = max((float(row["threshold"]) for row in fold_rows), default=None)
    return aggregate, fold_rows, predictions


def confusion_from_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    converted: list[dict[str, Any]] = []
    # Reuse the threshold confusion code by assigning threshold 0.5 to binary predictions.
    for row in rows:
        converted.append({**row, "score": float(row["predicted_drift"])})
    return confusion(converted, 0.5)


def main() -> None:
    args = parse_args()
    rows = load_scores(args)
    full = select_threshold(rows)
    cv, fold_rows, predictions = cross_validate(rows, args.folds)
    cv_mean_threshold = cv.get("threshold_mean")
    cv_mean_rule = confusion(rows, float(cv_mean_threshold)) if cv_mean_threshold is not None else None
    if cv_mean_rule is not None:
        cv_mean_rule["threshold_source"] = "mean of train-selected thresholds across cross-validation folds"
    summary = {
        "selected_rows": len(rows),
        "source": str(args.features_csv or args.layer_trace_csv),
        "metric": args.metric,
        "score_name": args.score_name,
        "layer": args.layer,
        "positive_label": POSITIVE,
        "negative_label": NEGATIVE,
        "rule": "predict drift iff score >= threshold",
        "threshold_selection": "maximize Youden J = TPR - FPR on calibration rows; tie-break by F1, precision, then higher threshold",
        "full_heldout_threshold": full,
        "cv_fold_calibrated_evaluation": cv,
        "cv_mean_threshold_rule_on_all_rows": cv_mean_rule,
        "score_distribution": describe_scores(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.output_prefix}_summary.json", summary)
    write_csv(args.output_dir / f"{args.output_prefix}_folds.csv", fold_rows)
    write_csv(args.output_dir / f"{args.output_prefix}_cv_predictions.csv", predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
