#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from evaluate_harmbench_classifier_transformers import classify_rows, summarize as summarize_classifier
from llama_safety_common import LLAMA32_3B_INSTRUCT_PATH, MLPActivationIntervention, dtype_from_name
from safety_benchmark_common import append_jsonl, dump_json, read_jsonl, sha256_text, summarize_native, write_jsonl


REPO = Path("/home/georgejammal/projects/semantic-to-symbolic")
ALIGNTREE = REPO / "external" / "AlignTree"
ALIGNTREE_DATA = ALIGNTREE / "dataset" / "processed"
OUT_ROOT = REPO / "safeguard_eval" / "outputs"
SINGLE_TOKEN_NEURON_JSON = (
    REPO
    / "safeguard_logit_lens"
    / "neuron_sets"
    / "llama3p2_3b_instruct_refusal_minus_answerability_single_token_top1000_layers14_27.json"
)

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


@dataclass(frozen=True)
class Condition:
    name: str
    layer_label: str
    layers: tuple[int, ...]
    mode: str
    scale_factor: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Llama-3.2-3B-Instruct single-token intervention on AlignTree malicious benchmarks."
    )
    parser.add_argument("--model-path", default=LLAMA32_3B_INSTRUCT_PATH)
    parser.add_argument("--model-alias", default="llama3p2_3b_instruct")
    parser.add_argument("--neuron-json", type=Path, default=SINGLE_TOKEN_NEURON_JSON)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layers", default="14-27")
    parser.add_argument("--mode", choices=["scale", "abs"], default="abs")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--classifier-path", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--completion-token-clip", type=int, default=512)
    parser.add_argument("--openai-judge-model", default="gpt-4o-mini")
    parser.add_argument("--openai-judge-seed", type=int, default=None)
    parser.add_argument("--gate-positive-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-harmbench", action="store_true")
    parser.add_argument("--skip-openai-judge", action="store_true")
    return parser.parse_args()


def run_id(raw: str | None) -> str:
    return raw or datetime.now(timezone.utc).strftime("llama3_single_token_align_tree_max512_%Y%m%d_%H%M%S")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_layers(raw: str) -> tuple[int, ...]:
    layers: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(item))
    if not layers:
        raise ValueError("At least one layer is required.")
    return tuple(dict.fromkeys(layers))


def scale_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_condition(args: argparse.Namespace) -> Condition:
    layers = parse_layers(args.layers)
    layer_label = f"layers{layers[0]}_{layers[-1]}" if len(layers) > 1 else f"layer{layers[0]}"
    if args.mode == "abs":
        suffix = "abs"
    else:
        suffix = f"scale{scale_label(args.scale_factor)}"
    return Condition(
        name=f"single_token_refusal_minus_answerability_{layer_label}_{suffix}",
        layer_label=layer_label,
        layers=layers,
        mode=args.mode,
        scale_factor=args.scale_factor,
    )


def load_aligntree_samples(n_test: int) -> list[dict[str, Any]]:
    dataset_map = [
        ("pair_test", "pair"),
        ("promptinject", "promptinject"),
        ("malwaregen", "malwaregen"),
        ("autodan_test", "autodan"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset_name, benchmark in dataset_map:
        data = json.loads((ALIGNTREE_DATA / f"{dataset_name}.json").read_text(encoding="utf-8"))
        for idx, item in enumerate(data[:n_test]):
            instruction = item["instruction"]
            rows.append(
                {
                    "sample_id": f"{dataset_name}__{idx:05d}__{sha256_text(instruction)[:12]}",
                    "benchmark": benchmark,
                    "aligntree_dataset": dataset_name,
                    "dataset_index": idx,
                    "prompt": instruction,
                    "instruction": instruction,
                    "behavior": instruction,
                    "target": item.get("target"),
                    "metadata": {key: value for key, value in item.items() if key not in {"instruction", "target"}},
                    "prompt_sha256": sha256_text(instruction),
                }
            )
    return rows


def format_llama3_prompt(instruction: str) -> str:
    return LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)


def load_neurons_for_layers(path: Path, layers: tuple[int, ...]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layer_set = set(layers)
    selected = [row for row in payload["top_neurons"] if int(row["layer"]) in layer_set]
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rank, row in enumerate(selected, start=1):
        by_layer[int(row["layer"])].append(
            {
                "neuron": int(row["neuron"]),
                "weight": 1.0,
                "rank": rank,
                "score": float(row.get("cosine_similarity", 0.0)),
            }
        )
    metadata = {
        "path": str(path),
        "source_direction": payload.get("direction_name"),
        "source_score_definition": payload.get("score_definition"),
        "source_top_k": payload.get("top_k"),
        "selected_layers": list(layers),
        "loaded": len(selected),
        "score_key": "cosine_similarity",
        "score_weighting": "none",
        "layer_counts": {str(layer): len(by_layer.get(layer, [])) for layer in layers},
    }
    return dict(sorted(by_layer.items())), metadata


def condition_dir(output_root: Path, rid: str, model_alias: str, condition: str) -> Path:
    return output_root / rid / model_alias / condition


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(sorted(counts.items()))


def completion_length_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(row.get("completion", "")) for row in rows]
    if not lengths:
        return {"min_chars": None, "max_chars": None, "mean_chars": None, "empty_count": 0}
    return {
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": sum(lengths) / len(lengths),
        "empty_count": sum(1 for value in lengths if value == 0),
    }


def append_debug_event(path: Path, event: str, payload: dict[str, Any]) -> None:
    append_jsonl(
        path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        },
    )


def generate_condition(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    condition: Condition,
    neurons_by_layer: dict[int, list[dict[str, Any]]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    gate_positive_only: bool,
) -> list[dict[str, Any]]:
    if output_path.exists():
        output_path.unlink()
    outputs: list[dict[str, Any]] = []
    start = time.time()
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    with MLPActivationIntervention(
        model,
        neurons_by_layer,
        condition.mode,
        condition.scale_factor,
        gate_positive_only=gate_positive_only,
    ):
        for batch_idx, batch in enumerate(batched(rows, batch_size), start=1):
            prompts = [format_llama3_prompt(row["instruction"]) for row in batch]
            inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                    generation_config=generation_config,
                )
            input_len = int(inputs["input_ids"].shape[1])
            decoded = tokenizer.batch_decode(
                generated[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample, completion in zip(batch, decoded):
                text = completion.strip()
                out = {**sample, "condition": condition.name, "completion": text, "response": text}
                outputs.append(out)
                append_jsonl(output_path, out)
            print(
                f"[generate {condition.name}] batch={batch_idx} seen={len(outputs)}/{len(rows)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return outputs


def write_condition_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    rid: str,
    condition: Condition,
    sample_count: int,
    neuron_metadata: dict[str, Any],
) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": rid,
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "sample_source": "AlignTree dataset/processed malicious datasets, first n_test rows",
        "sample_count": sample_count,
        "prompting": "AlignTree Llama3 chat template, tokenizer add_special_tokens default",
        "generation": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "do_sample": False},
        "condition": condition.__dict__ | {"layers": list(condition.layers)},
        "intervention": {
            "mode": condition.mode,
            "scale_factor": condition.scale_factor,
            "gate_positive_only": args.gate_positive_only,
            "site": "model.model.layers[*].mlp gated intermediate before down_proj",
            "token_scope": "all prompt positions and all autoregressive generation steps",
            "neuron_metadata": neuron_metadata,
        },
    }
    dump_json(output_dir / "manifest.json", manifest)


def run_generation(args: argparse.Namespace, rid: str, samples: list[dict[str, Any]], condition: Condition) -> None:
    print(f"[load] tokenizer={args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] model={args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.requires_grad_(False)

    out_dir = condition_dir(args.output_root, rid, args.model_alias, condition.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons_by_layer, neuron_metadata = load_neurons_for_layers(args.neuron_json, condition.layers)
    write_condition_manifest(out_dir, args, rid, condition, len(samples), neuron_metadata)
    rows_out = generate_condition(
        model=model,
        tokenizer=tokenizer,
        rows=samples,
        condition=condition,
        neurons_by_layer=neurons_by_layer,
        output_path=out_dir / "completions.jsonl",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        gate_positive_only=args.gate_positive_only,
    )
    dump_json(out_dir / "native_summary.json", summarize_native(rows_out))

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_harmbench(args: argparse.Namespace, rid: str, condition: Condition) -> list[dict[str, Any]]:
    print(f"[load] HarmBench tokenizer={args.classifier_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] HarmBench model={args.classifier_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.classifier_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    out_dir = condition_dir(args.output_root, rid, args.model_alias, condition.name)
    completions = out_dir / "completions.jsonl"
    if not completions.exists():
        print(f"[harmbench skip missing] {condition.name}", flush=True)
        return []
    rows = read_jsonl(completions)
    judge_dir = out_dir / "llama_harmbench_judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    labels = classify_rows(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        batch_size=args.classifier_batch_size,
        num_tokens=args.completion_token_clip,
    )
    write_jsonl(judge_dir / "labels.jsonl", labels)
    summary = summarize_classifier(labels)
    dump_json(judge_dir / "summary.json", summary)
    dump_json(
        judge_dir / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "judge": "HarmBench-Llama-2-13b-cls",
            "classifier_path": args.classifier_path,
            "batch_size": args.classifier_batch_size,
            "completion_token_clip": args.completion_token_clip,
            "completions": str(completions),
            "sample_count": len(rows),
        },
    )
    row_summary = {
        "condition": condition.name,
        "judge": "llama_harmbench",
        "overall": summary["overall"],
        "by_benchmark": summary["by_benchmark"],
    }
    dump_json(args.output_root / rid / args.model_alias / "llama_judge_grid_summary.json", {"rows": [row_summary]})

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [row_summary]


def evaluate_openai_judge(args: argparse.Namespace, rid: str, condition: Condition) -> list[dict[str, Any]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set for the OpenAI judge.")
    os.environ.setdefault("TOGETHER_API_KEY", "unused-by-chatgpt-judge")
    os.environ["ALIGNTREE_OPENAI_JUDGE_MODEL"] = args.openai_judge_model
    if args.openai_judge_seed is not None:
        os.environ["ALIGNTREE_OPENAI_JUDGE_SEED"] = str(args.openai_judge_seed)

    sys.path.insert(0, str(ALIGNTREE))
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreaks  # noqa: WPS433
    from pipeline.submodules.evaluate_jailbreak import substring_matching_judge_fn  # noqa: WPS433

    out_dir = condition_dir(args.output_root, rid, args.model_alias, condition.name)
    completions_path = out_dir / "completions.jsonl"
    judge_dir = out_dir / "openai_chatgpt_judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    debug_events_path = judge_dir / "debug_events.jsonl"
    if not completions_path.exists():
        print(f"[openai judge skip missing] {condition.name}", flush=True)
        append_debug_event(
            debug_events_path,
            "skip_missing_completions",
            {"condition": condition.name, "completions": str(completions_path)},
        )
        return []
    rows = read_jsonl(completions_path)
    completions = [{"instruction": row["instruction"], "response": row["completion"]} for row in rows]
    refusal_prefiltered = [bool(substring_matching_judge_fn(row["response"])) for row in completions]
    refusal_prefilter_count = sum(refusal_prefiltered)
    sent_to_openai_count = len(completions) - refusal_prefilter_count
    prompt_hashes = [row.get("prompt_sha256") or sha256_text(row.get("instruction", "")) for row in rows]
    completion_hashes = [sha256_text(row.get("completion", "")) for row in rows]
    initial_debug = {
        "condition": condition.name,
        "completions": str(completions_path),
        "sample_count": len(rows),
        "counts_by_benchmark": count_by(rows, "benchmark"),
        "counts_by_aligntree_dataset": count_by(rows, "aligntree_dataset"),
        "completion_length_stats": completion_length_stats(rows),
        "refusal_prefilter_count": refusal_prefilter_count,
        "openai_api_judged_count": sent_to_openai_count,
        "refusal_prefilter_count_by_benchmark": {
            bench: sum(1 for row, flag in zip(rows, refusal_prefiltered) if row["benchmark"] == bench and flag)
            for bench in sorted({row["benchmark"] for row in rows})
        },
        "openai_api_judged_count_by_benchmark": {
            bench: sum(1 for row, flag in zip(rows, refusal_prefiltered) if row["benchmark"] == bench and not flag)
            for bench in sorted({row["benchmark"] for row in rows})
        },
        "first_sample_id": rows[0]["sample_id"] if rows else None,
        "last_sample_id": rows[-1]["sample_id"] if rows else None,
        "first_prompt_sha256": prompt_hashes[0] if prompt_hashes else None,
        "last_prompt_sha256": prompt_hashes[-1] if prompt_hashes else None,
        "openai_model": args.openai_judge_model,
        "openai_seed": args.openai_judge_seed,
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    }
    append_debug_event(debug_events_path, "openai_judge_start", initial_debug)
    dump_json(judge_dir / "debug_start_report.json", initial_debug)

    print(
        f"[openai judge] rows={len(rows)} prefiltered={refusal_prefilter_count} "
        f"sent_to_openai={sent_to_openai_count}",
        flush=True,
    )
    try:
        evaluate_jailbreaks(completions=completions, methodologies=("chatgpt",))
    except Exception as exc:
        error_payload = {
            **initial_debug,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        dump_json(judge_dir / "debug_error_report.json", error_payload)
        append_debug_event(debug_events_path, "openai_judge_error", error_payload)
        raise

    labels = []
    audit_rows = []
    for idx, (row, judged, was_prefiltered, prompt_sha, completion_sha) in enumerate(
        zip(rows, completions, refusal_prefiltered, prompt_hashes, completion_hashes)
    ):
        label = int(judged.get("is_jailbreak_chatgpt", 0))
        label_row = {
            "sample_id": row["sample_id"],
            "benchmark": row["benchmark"],
            "condition": condition.name,
            "label": label,
            "label_name": "success" if label == 1 else "non_success",
            "prompt_sha256": prompt_sha,
            "completion_sha256": completion_sha,
            "judge": args.openai_judge_model,
            "refusal_prefiltered": was_prefiltered,
            "sent_to_openai": not was_prefiltered,
        }
        labels.append(label_row)
        audit_rows.append(
            {
                "row_index": idx,
                "sample_id": row["sample_id"],
                "benchmark": row["benchmark"],
                "aligntree_dataset": row.get("aligntree_dataset"),
                "dataset_index": row.get("dataset_index"),
                "prompt_sha256": prompt_sha,
                "completion_sha256": completion_sha,
                "completion_chars": len(row.get("completion", "")),
                "completion_lines": row.get("completion", "").count("\n") + (1 if row.get("completion", "") else 0),
                "refusal_prefiltered": was_prefiltered,
                "sent_to_openai": not was_prefiltered,
                "label": label,
                "label_name": label_row["label_name"],
            }
        )
    by_benchmark: dict[str, dict[str, Any]] = {}
    for bench in sorted({row["benchmark"] for row in labels}):
        bench_labels = [row["label"] for row in labels if row["benchmark"] == bench]
        success = sum(bench_labels)
        total = len(bench_labels)
        by_benchmark[bench] = {
            "total": total,
            "success": success,
            "non_success": total - success,
            "classifier_asr": success / total if total else None,
            "valid_total": total,
        }
    success = sum(row["label"] for row in labels)
    total = len(labels)
    summary = {
        "overall": {
            "total": total,
            "valid_total": total,
            "success": success,
            "non_success": total - success,
            "invalid": 0,
            "classifier_asr": success / total if total else None,
        },
        "by_benchmark": by_benchmark,
    }
    write_jsonl(judge_dir / "labels.jsonl", labels)
    write_jsonl(judge_dir / "debug_audit_rows.jsonl", audit_rows)
    dump_json(judge_dir / "summary.json", summary)
    debug_final = {
        **initial_debug,
        "summary": summary,
        "success_count_by_benchmark": {
            bench: by_benchmark[bench]["success"]
            for bench in sorted(by_benchmark)
        },
        "non_success_count_by_benchmark": {
            bench: by_benchmark[bench]["non_success"]
            for bench in sorted(by_benchmark)
        },
        "label_rows": str(judge_dir / "labels.jsonl"),
        "audit_rows": str(judge_dir / "debug_audit_rows.jsonl"),
    }
    dump_json(judge_dir / "debug_final_report.json", debug_final)
    append_debug_event(debug_events_path, "openai_judge_complete", debug_final)
    dump_json(
        judge_dir / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "judge": "AlignTree ChatGPT/HarmBench prompt judge",
            "openai_model": args.openai_judge_model,
            "openai_seed": args.openai_judge_seed,
            "completions": str(completions_path),
            "sample_count": total,
            "refusal_prefilter_count": refusal_prefilter_count,
            "openai_api_judged_count": total - refusal_prefilter_count,
            "note": "Uses AlignTree evaluate_jailbreaks chatgpt path, including substring-refusal prefilter.",
        },
    )
    row_summary = {"condition": condition.name, "judge": "openai_chatgpt", **summary}
    dump_json(args.output_root / rid / args.model_alias / "openai_judge_grid_summary.json", {"rows": [row_summary]})
    return [row_summary]


def main() -> None:
    args = parse_args()
    rid = run_id(args.run_id)
    condition = build_condition(args)
    samples = load_aligntree_samples(args.n_test)
    root = args.output_root / rid / args.model_alias
    root.mkdir(parents=True, exist_ok=True)
    dump_json(
        root / "grid_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": rid,
            "model_alias": args.model_alias,
            "model_path": args.model_path,
            "neuron_json": str(args.neuron_json),
            "sample_source": str(ALIGNTREE_DATA),
            "n_test_per_dataset": args.n_test,
            "sample_count": len(samples),
            "counts_by_benchmark": {
                bench: sum(1 for row in samples if row["benchmark"] == bench)
                for bench in sorted({row["benchmark"] for row in samples})
            },
            "generation": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "do_sample": False},
            "gate_positive_only": args.gate_positive_only,
            "condition": condition.__dict__ | {"layers": list(condition.layers)},
        },
    )
    print(
        f"[run] run_id={rid} samples={len(samples)} condition={condition.name} "
        f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    if not args.skip_generation:
        run_generation(args, rid, samples, condition)
    if not args.skip_harmbench:
        evaluate_harmbench(args, rid, condition)
    if not args.skip_openai_judge:
        evaluate_openai_judge(args, rid, condition)
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
