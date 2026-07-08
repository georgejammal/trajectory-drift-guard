#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from security_drift_common import dtype_from_name, read_jsonl, write_json, write_jsonl


PROJECT_ROOT = Path("/home/georgejammal/projects")
DRIFTING_SECURITY_SCRIPTS = PROJECT_ROOT / "drifting" / "security" / "scripts"
sys.path.insert(0, str(DRIFTING_SECURITY_SCRIPTS))

from evaluate_harmbench_classifier_transformers import classify_rows, summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge completion JSONL with the Llama HarmBench classifier and preserve security-drift metadata."
    )
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier-path", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def enrich_labels(labels: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample = {str(row["sample_id"]): row for row in rows}
    enriched: list[dict[str, Any]] = []
    for label in labels:
        source = by_sample.get(str(label["sample_id"]), {})
        enriched.append(
            {
                **label,
                "model_alias": source.get("model_alias"),
                "trace_group": source.get("trace_group"),
                "aligntree_dataset": source.get("aligntree_dataset"),
                "dataset_index": source.get("dataset_index"),
                "completion_chars": len(str(source.get("completion", ""))),
                "behavior": source.get("behavior"),
            }
        )
    return enriched


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.completions.parent / "llama_harmbench_judge"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.completions)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No completion rows selected.")

    print(f"[load] classifier tokenizer={args.classifier_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] classifier model={args.classifier_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.classifier_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    raw_labels = classify_rows(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=args.batch_size,
        num_tokens=args.num_tokens,
    )
    labels = enrich_labels(raw_labels, rows)
    write_jsonl(output_dir / "labels.jsonl", labels)
    summary = summarize(labels)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "judge": "HarmBench-Llama-2-13b-cls",
            "classifier_path": args.classifier_path,
            "completions": str(args.completions),
            "sample_count": len(rows),
            "batch_size": args.batch_size,
            "completion_token_clip": args.num_tokens,
            "label_semantics": {
                "1": "attack success; model failed to refuse",
                "0": "non-success; model refused or avoided the harmful behavior",
                "-1": "invalid classifier output",
            },
        },
    )
    del model
    del tokenizer
    gc.collect()
    print(summary, flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
