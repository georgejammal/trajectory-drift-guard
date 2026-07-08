#!/usr/bin/env python3
"""Plot signed selected-MLP drift scores for failed vs successful refusals."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ROOT = Path("/home/georgejammal/projects/try_and_change/security_drift")
RUN_ROOT = ROOT / "outputs" / "anchored_intervention_heldout_baseline_abs_negabs"
DEFAULT_OUT = ROOT / "plots" / "window_signed_drift_failed_vs_success.png"

MODELS = [
    ("gemma3_4b_it", "Gemma-3-4B-IT"),
    ("llama3p2_3b_instruct", "Llama-3.2-3B-Instruct"),
]

OUTCOME_LABELS = {
    "successful_refusal": "Successful refusal",
    "failed_refusal": "Failed refusal",
}

COLORS = {
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


def load_scores(model: str) -> tuple[dict[str, list[float]], dict]:
    summary_path = RUN_ROOT / model / "anchored_window_drift_summary.json"
    score_path = RUN_ROOT / model / "anchored_window_drift_all_window_scores.csv"
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


def describe(values: list[float]) -> str:
    return f"n={len(values)}, mean={mean(values):.4f}, median={median(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--also-pdf", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(10.8, 4.4), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]

    random.seed(0)
    summary_rows = []

    for ax, (model, title) in zip(axes, MODELS):
        scores, summary = load_scores(model)
        success = scores["successful_refusal"]
        failed = scores["failed_refusal"]
        auc = rank_auc(failed, success)
        window = summary["best_signed_window"]

        positions = [0, 1]
        data = [success, failed]
        outcomes = ["successful_refusal", "failed_refusal"]

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
        for patch, outcome in zip(box["boxes"], outcomes):
            patch.set_facecolor(COLORS[outcome])
            patch.set_alpha(0.22)

        for pos, outcome, values in zip(positions, outcomes, data):
            jitter = [pos + random.uniform(-0.11, 0.11) for _ in values]
            ax.scatter(
                jitter,
                values,
                s=16,
                alpha=0.62,
                linewidths=0,
                color=COLORS[outcome],
            )

        ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Successful\nrefusal", "Failed\nrefusal"])
        ax.set_title(
            f"{title}\nwindow {window['window_start']}-{window['window_end']}, AUC={auc:.3f}",
            fontsize=11,
        )
        ax.set_ylabel("Signed drift score")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)

        summary_rows.append(
            {
                "model": model,
                "window": f"{window['window_start']}-{window['window_end']}",
                "auc": auc,
                "successful": describe(success),
                "failed": describe(failed),
            }
        )

    fig.suptitle("Baseline selected-MLP drift scores on held-out malicious prompts", fontsize=13)
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
