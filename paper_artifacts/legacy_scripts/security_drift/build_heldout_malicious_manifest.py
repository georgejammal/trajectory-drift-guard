#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from security_drift_common import (
    ALIGNTREE_DATA,
    DATASET_MAP,
    SAMPLE_ROOT,
    count_by,
    read_json,
    sample_id,
    sha256_text,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT = SAMPLE_ROOT / "heldout_malware_promptinject_seed0_all_unused.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a malicious heldout manifest from AlignTree rows not used by first-100 evaluations."
    )
    parser.add_argument("--benchmarks", nargs="+", default=["malwaregen", "promptinject"], choices=sorted(DATASET_MAP))
    parser.add_argument("--unused-start-index", type=int, default=100)
    parser.add_argument("--per-benchmark", type=int, default=None, help="Optional random subset size per benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(benchmark: str, start_index: int) -> list[dict[str, Any]]:
    dataset_name = DATASET_MAP[benchmark]
    path = ALIGNTREE_DATA / f"{dataset_name}.json"
    data = read_json(path)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if idx < start_index:
            continue
        instruction = str(item["instruction"])
        rows.append(
            {
                "sample_id": sample_id(dataset_name, idx, instruction),
                "trace_group": "malicious",
                "benchmark": benchmark,
                "aligntree_dataset": dataset_name,
                "dataset_index": idx,
                "prompt": instruction,
                "instruction": instruction,
                "behavior": item.get("goal") or instruction,
                "target": item.get("target"),
                "prompt_sha256": sha256_text(instruction),
                "heldout_reason": f"index >= {start_index}; original first-100 evaluation used indices 0-{start_index - 1}",
                "metadata": {key: value for key, value in item.items() if key not in {"instruction", "target"}},
            }
        )
    return rows


def maybe_sample(rows: list[dict[str, Any]], benchmark: str, n: int | None, seed: int) -> list[dict[str, Any]]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(f"{seed}:heldout:{benchmark}")
    pool = list(rows)
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda row: int(row["dataset_index"]))


def main() -> None:
    args = parse_args()
    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    index_ranges: dict[str, dict[str, int | None]] = {}
    for benchmark in args.benchmarks:
        rows = load_rows(benchmark, args.unused_start_index)
        available_counts[benchmark] = len(rows)
        subset = maybe_sample(rows, benchmark, args.per_benchmark, args.seed)
        selected.extend(subset)
        indices = [int(row["dataset_index"]) for row in subset]
        index_ranges[benchmark] = {
            "min": min(indices) if indices else None,
            "max": max(indices) if indices else None,
        }
    selected.sort(key=lambda row: (row["benchmark"], int(row["dataset_index"])))
    write_jsonl(args.output_jsonl, selected)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(args.output_jsonl),
        "sample_count": len(selected),
        "benchmarks": args.benchmarks,
        "unused_start_index": args.unused_start_index,
        "per_benchmark": args.per_benchmark,
        "seed": args.seed,
        "available_unused_counts": available_counts,
        "selected_counts_by_benchmark": count_by(selected, "benchmark"),
        "selected_index_ranges": index_ranges,
        "source_dir": str(ALIGNTREE_DATA),
    }
    write_json(args.output_jsonl.with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
