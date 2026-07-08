#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT = SAMPLE_ROOT / "first100_pair_promptinject_malwaregen_autodan.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the first-100 per malicious benchmark manifest used by the final security evaluation."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["pair", "promptinject", "malwaregen", "autodan"],
        choices=sorted(DATASET_MAP),
    )
    parser.add_argument("--n-per-benchmark", type=int, default=100)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_first_n(benchmark: str, n: int) -> list[dict[str, Any]]:
    dataset_name = DATASET_MAP[benchmark]
    path = ALIGNTREE_DATA / f"{dataset_name}.json"
    data = read_json(path)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data[:n]):
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
                "eval_split": f"first_{n}_per_benchmark",
                "metadata": {key: value for key, value in item.items() if key not in {"instruction", "target"}},
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        rows.extend(load_first_n(benchmark, args.n_per_benchmark))
    rows.sort(key=lambda row: (args.benchmarks.index(str(row["benchmark"])), int(row["dataset_index"])))
    write_jsonl(args.output_jsonl, rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_jsonl": str(args.output_jsonl),
        "sample_count": len(rows),
        "benchmarks": args.benchmarks,
        "n_per_benchmark": args.n_per_benchmark,
        "counts_by_benchmark": count_by(rows, "benchmark"),
        "source_dir": str(ALIGNTREE_DATA),
    }
    write_json(args.output_jsonl.with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
