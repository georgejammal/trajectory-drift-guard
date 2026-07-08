#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from evaluate_gemma3_benign_utility import build_samples, extract_prediction, is_refusal, summarize
from llama_safety_common import (
    LLAMA32_3B_INSTRUCT_PATH,
    MLPActivationIntervention,
    OUT_ROOT,
    build_chat_prompt,
    load_model,
    load_neurons,
    load_tokenizer,
    write_jsonl,
)
from safety_benchmark_common import dump_json


REPO = Path("/home/georgejammal/projects/semantic-to-symbolic")
SAMPLE_ROOT = REPO / "safeguard_eval" / "sample_sets"
REFUSAL_SET = (
    REPO
    / "safeguard_logit_lens/neuron_sets/llama3p2_3b_instruct_refusal_minus_answerability_top1000_layers14_27.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Llama baseline vs safety intervention on Katz-style benign QA.")
    parser.add_argument("--model-path", default=LLAMA32_3B_INSTRUCT_PATH)
    parser.add_argument("--model-alias", default="llama3p2_3b_instruct")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--only-condition", choices=["baseline", "intervention"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_id(raw: str | None) -> str:
    return raw or datetime.now(timezone.utc).strftime("llama_benign_utility_%Y%m%d_%H%M%S")


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate_condition(
    *,
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    condition: str,
    intervention: MLPActivationIntervention,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    rows: list[dict[str, Any]] = []
    with intervention:
        for batch_idx, batch in enumerate(batched(samples, batch_size), start=1):
            prompts = [build_chat_prompt(tokenizer, sample["prompt"]) for sample in batch]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_len = int(inputs["input_ids"].shape[1])
            decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for sample, completion in zip(batch, decoded):
                pred = extract_prediction(completion, sample["choices"])
                rows.append(
                    {
                        **sample,
                        "condition": condition,
                        "completion": completion.strip(),
                        "prediction": pred,
                        "correct": pred == sample["gold"],
                        "refusal": is_refusal(completion),
                    }
                )
            write_jsonl(output_path, rows)
            print(f"[generate {condition}] batch={batch_idx} seen={len(rows)}/{len(samples)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    rid = run_id(args.run_id)
    output_dir = args.output_root / rid / args.model_alias
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLE_ROOT / f"benign_utility_katz_style_seed{args.seed}_n{args.samples_per_benchmark}.jsonl"
    if sample_path.exists():
        samples = read_jsonl(sample_path)
    else:
        samples = build_samples(args.seed, args.samples_per_benchmark)
        write_jsonl(sample_path, samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": args.model_alias,
        "model_path": args.model_path,
        "sample_set": str(sample_path),
        "sample_count": len(samples),
        "seed": args.seed,
        "samples_per_benchmark": args.samples_per_benchmark,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "neuron_set": str(REFUSAL_SET),
        "protocol": "Katz-style benign QA check: PIQA, OpenBookQA, Social IQA, ARC; accuracy and refusal rate.",
    }
    dump_json(output_dir / "manifest.json", manifest)
    print(f"[load] tokenizer={args.model_path}", flush=True)
    tokenizer = load_tokenizer(args.model_path)
    print(f"[load] model={args.model_path}", flush=True)
    model = load_model(args.model_path, args.dtype)
    summaries: dict[str, Any] = {}
    conditions: list[tuple[str, MLPActivationIntervention]] = []
    if args.only_condition in (None, "baseline"):
        conditions.append(("baseline", MLPActivationIntervention(model, {}, "none", 1.0, False)))
    if args.only_condition in (None, "intervention"):
        neurons_by_layer, neuron_metadata = load_neurons(REFUSAL_SET, 1000, "cosine_similarity", "none")
        manifest["intervention_neuron_metadata"] = neuron_metadata
        dump_json(output_dir / "manifest.json", manifest)
        conditions.append(("refusal_minus_answerability_top1000_abs", MLPActivationIntervention(model, neurons_by_layer, "abs", 1.0, False)))
    for condition, intervention in conditions:
        condition_dir = output_dir / condition
        rows = generate_condition(
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            output_path=condition_dir / "completions.jsonl",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            condition=condition,
            intervention=intervention,
        )
        condition_summary = summarize(rows)
        dump_json(condition_dir / "summary.json", condition_summary)
        summaries[condition] = condition_summary
        print(json.dumps({condition: condition_summary}, indent=2), flush=True)
    if "baseline" in summaries and "refusal_minus_answerability_top1000_abs" in summaries:
        base = summaries["baseline"]
        intr = summaries["refusal_minus_answerability_top1000_abs"]
        comparison = {
            "baseline_accuracy": base["accuracy"],
            "intervention_accuracy": intr["accuracy"],
            "accuracy_delta": intr["accuracy"] - base["accuracy"],
            "baseline_refusal_rate": base["refusal_rate"],
            "intervention_refusal_rate": intr["refusal_rate"],
            "refusal_rate_delta": intr["refusal_rate"] - base["refusal_rate"],
            "by_benchmark": {},
        }
        for benchmark in sorted(base["by_benchmark"]):
            b = base["by_benchmark"][benchmark]
            i = intr["by_benchmark"][benchmark]
            comparison["by_benchmark"][benchmark] = {
                "baseline_accuracy": b["accuracy"],
                "intervention_accuracy": i["accuracy"],
                "accuracy_delta": i["accuracy"] - b["accuracy"],
                "baseline_refusal_rate": b["refusal_rate"],
                "intervention_refusal_rate": i["refusal_rate"],
                "refusal_rate_delta": i["refusal_rate"] - b["refusal_rate"],
            }
        dump_json(output_dir / "comparison_summary.json", comparison)
        print(json.dumps({"comparison": comparison}, indent=2), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
