#!/usr/bin/env python3
"""Plot signed selected-MLP drift scores for benign, successful refusal, and failed refusal prompts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ROOT = Path("/home/georgejammal/projects/try_and_change/security_drift")
MALICIOUS_RUN_ROOT = ROOT / "outputs" / "anchored_intervention_heldout_baseline_abs_negabs"
BENIGN_RUN_ROOT = ROOT / "outputs" / "benign_window_drift_scores"
DEFAULT_OUT = ROOT / "plots" / "window_signed_drift_benign_success_failed.png"

MODELS = [
    ("gemma3_4b_it", "Gemma-3-4B-IT"),
    ("llama3p2_3b_instruct", "Llama-3.2-3B-Instruct"),
]

CATEGORY_LABELS = {
    "benign": "Benign",
    "successful_refusal": "Successful refusal",
    "failed_refusal": "Failed refusal",
}

COLORS = {
    "benign": "#4f7d45",
    "successful_refusal": "#2f6f9f",
    "failed_refusal": "#b2413b",
}


def rank_auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def load_malicious_scores(model: str) -> tuple[dict[str, list[float]], dict]:
    summary_path = MALICIOUS_RUN_ROOT / model / "anchored_window_drift_summary.json"
    score_path = MALICIOUS_RUN_ROOT / model / "anchored_window_drift_all_window_scores.csv"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    window = summary["best_signed_window"]
    start = int(window["window_start"])
    end = int(window["window_end"])
    score_name = str(window["score_name"])

    scores = {"successful_refusal": [], "failed_refusal": []}
    with score_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["condition"] != "baseline":
                continue
            if row["score_name"] != score_name:
                continue
            if int(row["window_start"]) != start or int(row["window_end"]) != end:
                continue
            outcome = row["drift_outcome"]
            if outcome in scores:
                scores[outcome].append(float(row["score"]))

    return scores, summary


def load_benign_scores(model: str, window: dict) -> list[float]:
    score_path = BENIGN_RUN_ROOT / model / "fixed_window_benign_drift_scores.csv"
    start = int(window["window_start"])
    end = int(window["window_end"])
    scores: list[float] = []
    with score_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["condition"] != "baseline":
                continue
            if row["score_name"] != "window_signed_deficit":
                continue
            if int(row["window_start"]) != start or int(row["window_end"]) != end:
                continue
            scores.append(float(row["score"]))
    return scores


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def fmt_desc(values: list[float]) -> str:
    stats = describe(values)
    if stats["n"] == 0:
        return "n=0"
    return f"n={stats['n']}, mean={stats['mean']:.4f}, median={stats['median']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--also-pdf", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(12.6, 4.6), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    random.seed(0)
    summary_rows = []

    for ax, (model, title) in zip(axes, MODELS):
        malicious_scores, summary = load_malicious_scores(model)
        window = summary["best_signed_window"]
        benign = load_benign_scores(model, window)
        success = malicious_scores["successful_refusal"]
        failed = malicious_scores["failed_refusal"]
        auc = rank_auc(failed, success)

        positions = [0, 1, 2]
        categories = ["benign", "successful_refusal", "failed_refusal"]
        data = [benign, success, failed]

        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, category in zip(box["boxes"], categories):
            patch.set_facecolor(COLORS[category])
            patch.set_alpha(0.22)

        for pos, category, values in zip(positions, categories, data):
            jitter = [pos + random.uniform(-0.11, 0.11) for _ in values]
            ax.scatter(
                jitter,
                values,
                s=16,
                alpha=0.62,
                linewidths=0,
                color=COLORS[category],
            )

        ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Benign", "Successful\nrefusal", "Failed\nrefusal"])
        ax.set_title(
            f"{title}\nwindow {window['window_start']}-{window['window_end']}, failed-vs-success AUC={auc:.3f}",
            fontsize=11,
        )
        ax.set_ylabel("Signed drift score")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)

        summary_rows.append(
            {
                "model": model,
                "window": f"{window['window_start']}-{window['window_end']}",
                "failed_vs_success_auc": auc,
                "benign": fmt_desc(benign),
                "successful": fmt_desc(success),
                "failed": fmt_desc(failed),
                "benign_stats": describe(benign),
                "successful_stats": describe(success),
                "failed_stats": describe(failed),
            }
        )

    fig.suptitle("Baseline selected-MLP drift scores: benign prompts and malicious refusal outcomes", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    if args.also_pdf:
        fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")

    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    print(args.output)
    print(summary_path)
    if args.also_pdf:
        print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
