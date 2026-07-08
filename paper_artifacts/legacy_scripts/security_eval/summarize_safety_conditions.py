#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize safety condition outputs.")
    parser.add_argument("--run-root", type=Path, required=True, help="Directory containing condition subdirectories.")
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_condition(condition_dir: Path, baseline_asr: float | None) -> dict[str, Any] | None:
    cls_summary = load_json(condition_dir / "harmbench_classifier" / "summary.json")
    if cls_summary is None:
        return None
    native_summary = load_json(condition_dir / "native_summary.json")
    manifest = load_json(condition_dir / "manifest.json") or {}
    intervention = manifest.get("intervention", {})
    neuron_meta = intervention.get("neuron_metadata") or {}
    overall = cls_summary["overall"]
    row: dict[str, Any] = {
        "condition": condition_dir.name,
        "classifier_asr": overall.get("classifier_asr"),
        "success": overall.get("success"),
        "valid_total": overall.get("valid_total"),
        "mode": intervention.get("mode"),
        "scale_factor": intervention.get("scale_factor"),
        "top_k": neuron_meta.get("top_k"),
        "source_direction": neuron_meta.get("source_direction"),
        "score_weighting": neuron_meta.get("score_weighting"),
        "native_asr": (native_summary or {}).get("overall", {}).get("native_asr"),
    }
    if baseline_asr is not None and row["classifier_asr"] is not None:
        row["asr_reduction_pp"] = 100.0 * (baseline_asr - row["classifier_asr"])
        row["relative_asr_reduction_pct"] = (
            100.0 * (baseline_asr - row["classifier_asr"]) / baseline_asr if baseline_asr else None
        )
    for bench, bench_summary in cls_summary.get("by_benchmark", {}).items():
        row[f"{bench}_classifier_asr"] = bench_summary.get("classifier_asr")
        row[f"{bench}_success"] = bench_summary.get("success")
        row[f"{bench}_total"] = bench_summary.get("valid_total")
    return row


def main() -> None:
    args = parse_args()
    baseline_asr = None
    if args.baseline_summary:
        baseline = load_json(args.baseline_summary)
        if baseline:
            baseline_asr = baseline["overall"]["classifier_asr"]
    rows = []
    for condition_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        row = flatten_condition(condition_dir, baseline_asr)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda row: (row.get("classifier_asr") is None, row.get("classifier_asr", 1e9)))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"rows": rows[: args.top_n], "total_conditions": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
