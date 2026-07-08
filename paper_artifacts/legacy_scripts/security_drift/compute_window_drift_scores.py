#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from security_drift_common import (
    MODEL_DEFAULTS,
    OUTPUT_ROOT,
    batched,
    count_by,
    dtype_from_name,
    format_prompt,
    load_neurons,
    model_defaults,
    parse_layers,
    read_jsonl,
    write_json,
)
from trace_anchored_intervention_drift import AnchoredInterventionRecorder, direction_bundle_from_neuron_json


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute raw fixed-window selected-MLP drift scores without labels, thresholds, "
            "or classifier fitting. The main score is -sum selected refusal-direction work."
        )
    )
    parser.add_argument("--model-alias", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--neuron-json", type=Path, default=None)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", nargs="+", choices=["baseline", "abs", "neg_abs"], default=["baseline"])
    parser.add_argument("--trace-group", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-prefix", default="window_drift")
    return parser.parse_args()


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.trace_group is not None:
        selected = [row for row in selected if row.get("trace_group") == args.trace_group]
    if args.benchmarks:
        keep = set(args.benchmarks)
        selected = [row for row in selected if row.get("benchmark") in keep]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    assert avg is not None
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": std(values),
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "max": max(values) if values else None,
    }


def score_rows_from_traces(
    rows: list[dict[str, Any]],
    conditions: list[str],
    layers: list[int],
    condition_traces: dict[tuple[str, str], dict[int, dict[str, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        for condition in conditions:
            trace = condition_traces.get((sample_id, condition), {})
            if any(layer not in trace for layer in layers):
                continue
            work = sum(float(trace[layer]["post_work"]) for layer in layers)
            pos_work = sum(float(trace[layer]["post_pos_work"]) for layer in layers)
            neg_work = sum(float(trace[layer]["post_neg_work"]) for layer in layers)
            score = -work
            base = {
                "sample_id": sample_id,
                "condition": condition,
                "trace_group": row.get("trace_group"),
                "benchmark": row.get("benchmark"),
                "dataset_index": row.get("dataset_index"),
                "target": row.get("target"),
                "window_start": layers[0],
                "window_end": layers[-1],
                "window_width": len(layers),
                "window_layers": ",".join(str(layer) for layer in layers),
            }
            scores.append(
                {
                    **base,
                    "score_name": "window_signed_deficit",
                    "score": score,
                    "selected_terminal_work": work,
                    "selected_positive_work": pos_work,
                    "selected_negative_work": neg_work,
                    "selected_negative_fraction": neg_work / (pos_work + neg_work + EPS),
                    "selected_neuron_count": sum(int(trace[layer]["selected_neuron_count"]) for layer in layers),
                }
            )
            for layer in layers:
                layer_rows.append(
                    {
                        **base,
                        "layer": layer,
                        **trace[layer],
                    }
                )
    return scores, layer_rows


def grouped_stats(scores: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in sorted({str(row["condition"]) for row in scores}):
        condition_scores = [float(row["score"]) for row in scores if row["condition"] == condition]
        out[condition] = {"all": describe(condition_scores), "by_benchmark": {}}
        for benchmark in sorted({str(row.get("benchmark")) for row in scores if row["condition"] == condition}):
            values = [
                float(row["score"])
                for row in scores
                if row["condition"] == condition and str(row.get("benchmark")) == benchmark
            ]
            out[condition]["by_benchmark"][benchmark] = describe(values)
    return out


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    stats = summary["score_stats"]
    lines = [
        "# Fixed-Window Benign Drift Scores",
        "",
        f"- Model: `{summary['model_alias']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Conditions: `{', '.join(summary['conditions'])}`",
        f"- Window: `{summary['window_layers']}`",
        "",
        "The score is raw and label-free:",
        "",
        "\\[",
        "D_W^g(x)=-\\sum_{\\ell\\in W}\\sum_{i\\in S_\\ell}g(a_{\\ell i}(x,t_\\star))\\langle w_{\\ell i},\\hat d\\rangle.",
        "\\]",
        "",
        "Higher values mean the selected MLP additions preserve less refusal-direction work in the chosen window.",
        "",
        "## Overall",
        "",
        "| Condition | n | mean | std | p25 | median | p75 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, condition_stats in stats.items():
        item = condition_stats["all"]
        lines.append(
            f"| {condition} | {item['n']} | {fmt(item['mean'])} | {fmt(item['std'])} | "
            f"{fmt(item['p25'])} | {fmt(item['median'])} | {fmt(item['p75'])} | "
            f"{fmt(item['min'])} | {fmt(item['max'])} |"
        )
    lines.extend(["", "## By Benchmark", "", "| Condition | Benchmark | n | mean | median | min | max |", "|---|---|---:|---:|---:|---:|---:|"])
    for condition, condition_stats in stats.items():
        for benchmark, item in condition_stats["by_benchmark"].items():
            lines.append(
                f"| {condition} | {benchmark} | {item['n']} | {fmt(item['mean'])} | "
                f"{fmt(item['median'])} | {fmt(item['min'])} | {fmt(item['max'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    defaults = model_defaults(args.model_alias)
    model_path = args.model_path or str(defaults["model_path"])
    neuron_json = args.neuron_json or Path(defaults["neuron_json"])
    layers = list(parse_layers(args.layers))
    rows = filter_rows(read_jsonl(args.rows_jsonl), args)
    if not rows:
        raise RuntimeError("No rows selected.")

    out_dir = args.output_root / args.run_id / args.model_alias
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load tokenizer] {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load model] {model_path}", flush=True)
    model_kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    if args.attn_implementation and args.attn_implementation != "default":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    model.requires_grad_(False)

    neurons_by_layer, neuron_metadata = load_neurons(neuron_json, tuple(layers), score_key="cosine_similarity", positive_only=True)
    direction_bundle, direction_metadata, token_id_metadata = direction_bundle_from_neuron_json(model, neuron_json)

    condition_traces: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for condition in args.conditions:
        for batch_idx, batch in enumerate(batched(rows, args.batch_size), start=1):
            prompts = [format_prompt(args.model_alias, str(row["instruction"])) for row in batch]
            encoded_unpadded = tokenizer(prompts, padding=False, truncation=False)
            prompt_lengths = [len(ids) for ids in encoded_unpadded["input_ids"]]
            encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            sample_ids = [str(row["sample_id"]) for row in batch]
            with torch.no_grad():
                with AnchoredInterventionRecorder(
                    model,
                    neurons_by_layer,
                    direction_bundle["direction_unit"],
                    condition=condition,
                    sample_ids=sample_ids,
                    prompt_lengths=prompt_lengths,
                    store=condition_traces,
                ):
                    model(**encoded, use_cache=False)
            print(
                f"[score] model={args.model_alias} condition={condition} batch={batch_idx} "
                f"seen={min(batch_idx * args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scores, layer_rows = score_rows_from_traces(rows, args.conditions, layers, condition_traces)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": args.model_alias,
        "model_path": model_path,
        "rows_jsonl": str(args.rows_jsonl),
        "sample_count": len(rows),
        "counts_by_trace_group": count_by(rows, "trace_group"),
        "counts_by_benchmark": count_by(rows, "benchmark"),
        "conditions": args.conditions,
        "layers": layers,
        "window_layers": ",".join(str(layer) for layer in layers),
        "neuron_metadata": neuron_metadata,
        "direction_metadata": direction_metadata,
        "target_token_ids": token_id_metadata,
        "score_stats": grouped_stats(scores),
        "definition": {
            "score_name": "window_signed_deficit",
            "formula": "-sum_l in W sum_i in S_l g(a_li(x,t_star)) <w_li, unit(refusal-answerability)>",
            "interpretation": "larger score means less selected MLP work preserving the refusal direction in this window",
            "conditions": {
                "baseline": "g(a)=a",
                "abs": "g(a)=abs(a), applied before down_proj",
                "neg_abs": "g(a)=-abs(a), applied before down_proj",
            },
            "label_use": "none",
            "threshold_use": "none",
        },
    }

    prefix = args.output_prefix
    write_csv(out_dir / f"{prefix}_scores.csv", scores)
    write_csv(out_dir / f"{prefix}_layer_trace.csv", layer_rows)
    write_json(out_dir / f"{prefix}_summary.json", summary)
    write_report(out_dir / f"{prefix}_report.md", summary)
    print(out_dir / f"{prefix}_report.md", flush=True)


if __name__ == "__main__":
    main()
