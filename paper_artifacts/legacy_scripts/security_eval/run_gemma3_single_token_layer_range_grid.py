#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from evaluate_harmbench_classifier_transformers import classify_rows, summarize as summarize_classifier
from safety_benchmark_common import append_jsonl, dump_json, read_jsonl, sha256_text, summarize_native, write_jsonl


REPO = Path("/home/georgejammal/projects/semantic-to-symbolic")
ALIGNTREE = REPO / "external" / "AlignTree"
ALIGNTREE_DATA = ALIGNTREE / "dataset" / "processed"
OUT_ROOT = REPO / "safeguard_eval" / "outputs"
GEMMA3_PATH = (
    "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/"
    "snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
)
SINGLE_TOKEN_NEURON_JSON = (
    REPO
    / "safeguard_logit_lens"
    / "neuron_sets"
    / "gemma3_4b_it_refusal_minus_answerability_single_token_top1000_layers17_33.json"
)
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
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
        description="Run Gemma3 single-token refusal/answerability layer-range intervention grid on AlignTree malicious benchmarks."
    )
    parser.add_argument("--model-path", default=GEMMA3_PATH)
    parser.add_argument("--neuron-json", type=Path, default=SINGLE_TOKEN_NEURON_JSON)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--classifier-path", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--classifier-batch-size", type=int, default=16)
    parser.add_argument("--completion-token-clip", type=int, default=512)
    parser.add_argument("--openai-judge-model", default="gpt-4o-mini")
    parser.add_argument("--openai-judge-seed", type=int, default=None)
    parser.add_argument("--gate-positive-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-llama-judge", action="store_true")
    parser.add_argument("--skip-openai-judge", action="store_true")
    parser.add_argument("--only-conditions", default=None, help="Comma-separated substrings to run/evaluate.")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def run_id(raw: str | None) -> str:
    return raw or datetime.now(timezone.utc).strftime("gemma3_single_token_layer_ranges_max512_%Y%m%d_%H%M%S")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def scale_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_conditions() -> list[Condition]:
    ranges = [
        ("layers17_18", tuple(range(17, 19))),
        ("layers17_21", tuple(range(17, 22))),
        ("layers20_22", tuple(range(20, 23))),
        ("layers17_33", tuple(range(17, 34))),
        ("layers17_30", tuple(range(17, 31))),
    ]
    conditions: list[Condition] = []
    for layer_label, layers in ranges:
        for scale in (-1.0, 0.0, 1.2, 1.5, 2.0):
            conditions.append(
                Condition(
                    name=f"single_token_refusal_minus_answerability_{layer_label}_scale{scale_label(scale)}",
                    layer_label=layer_label,
                    layers=layers,
                    mode="scale",
                    scale_factor=scale,
                )
            )
        conditions.append(
            Condition(
                name=f"single_token_refusal_minus_answerability_{layer_label}_abs",
                layer_label=layer_label,
                layers=layers,
                mode="abs",
                scale_factor=1.0,
            )
        )
    return conditions


def filter_conditions(conditions: list[Condition], raw: str | None) -> list[Condition]:
    if raw is None:
        return conditions
    needles = [part.strip() for part in raw.split(",") if part.strip()]
    return [condition for condition in conditions if any(needle in condition.name for needle in needles)]


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


def format_gemma_prompt(instruction: str) -> str:
    return GEMMA_CHAT_TEMPLATE.format(instruction=instruction)


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


class MLPActivationIntervention:
    def __init__(
        self,
        model: Any,
        neurons_by_layer: dict[int, list[dict[str, Any]]],
        mode: str,
        scale_factor: float,
        gate_positive_only: bool,
    ):
        self.model = model
        self.neurons_by_layer = neurons_by_layer
        self.mode = mode
        self.scale_factor = scale_factor
        self.gate_positive_only = gate_positive_only
        self.original_forwards: dict[int, Any] = {}

    def _layers(self) -> Any:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            language_model = self.model.model.language_model
            if hasattr(language_model, "layers"):
                return language_model.layers
            if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
                return language_model.model.layers
        if hasattr(self.model, "language_model"):
            language_model = self.model.language_model
            if hasattr(language_model, "layers"):
                return language_model.layers
            if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
                return language_model.model.layers
        raise AttributeError("Could not find Gemma decoder layers.")

    def __enter__(self):
        layers = self._layers()
        for layer_idx, rows in self.neurons_by_layer.items():
            mlp = layers[layer_idx].mlp
            self.original_forwards[layer_idx] = mlp.forward
            indices = torch.tensor([row["neuron"] for row in rows], dtype=torch.long)
            mode = self.mode
            factor = float(self.scale_factor)
            gate_only = bool(self.gate_positive_only)

            def patched_forward(module, x, patch_indices=indices, patch_mode=mode, patch_factor=factor, patch_gate_only=gate_only):
                gate_act = module.act_fn(module.gate_proj(x))
                gated = gate_act * module.up_proj(x)
                idx = patch_indices.to(gated.device)
                target = gated[..., idx]
                if patch_mode == "scale":
                    replacement = target * patch_factor
                elif patch_mode == "abs":
                    replacement = target.abs()
                else:
                    replacement = target
                if patch_gate_only:
                    mask = gate_act[..., idx] > 0
                    replacement = torch.where(mask, replacement, target)
                gated[..., idx] = replacement
                return module.down_proj(gated)

            mlp.forward = MethodType(patched_forward, mlp)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        layers = self._layers()
        for layer_idx, original_forward in self.original_forwards.items():
            layers[layer_idx].mlp.forward = original_forward
        return False


def condition_dir(output_root: Path, rid: str, condition: str) -> Path:
    return output_root / rid / "gemma3_4b_it" / condition


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
    with MLPActivationIntervention(model, neurons_by_layer, condition.mode, condition.scale_factor, gate_positive_only):
        for batch_idx, batch in enumerate(batched(rows, batch_size), start=1):
            prompts = [format_gemma_prompt(row["instruction"]) for row in batch]
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
                out = {**sample, "condition": condition.name, "completion": completion.strip(), "response": completion.strip()}
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
        "model_alias": "gemma3_4b_it",
        "model_path": args.model_path,
        "sample_source": "AlignTree dataset/processed malicious datasets, first n_test rows",
        "sample_count": sample_count,
        "prompting": "AlignTree Gemma chat template, include trailing model turn newline",
        "generation": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "do_sample": False},
        "condition": condition.__dict__ | {"layers": list(condition.layers)},
        "intervention": {
            "mode": condition.mode,
            "scale_factor": condition.scale_factor,
            "gate_positive_only": args.gate_positive_only,
            "site": "model.model.language_model.layers[*].mlp gated intermediate before down_proj",
            "token_scope": "all prompt positions and all autoregressive generation steps",
            "neuron_metadata": neuron_metadata,
        },
    }
    dump_json(output_dir / "manifest.json", manifest)


def run_generation(args: argparse.Namespace, rid: str, samples: list[dict[str, Any]], conditions: list[Condition]) -> None:
    print(f"[load] tokenizer={args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] model={args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    model.requires_grad_(False)

    for idx, condition in enumerate(conditions, start=1):
        out_dir = condition_dir(args.output_root, rid, condition.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[condition {idx}/{len(conditions)}] {condition.name}", flush=True)
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_llama_judge(args: argparse.Namespace, rid: str, conditions: list[Condition]) -> list[dict[str, Any]]:
    print(f"[load] llama judge tokenizer={args.classifier_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[load] llama judge model={args.classifier_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.classifier_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    rows_summary: list[dict[str, Any]] = []
    for idx, condition in enumerate(conditions, start=1):
        out_dir = condition_dir(args.output_root, rid, condition.name)
        completions = out_dir / "completions.jsonl"
        if not completions.exists():
            print(f"[llama judge skip missing {idx}/{len(conditions)}] {condition.name}", flush=True)
            continue
        rows = read_jsonl(completions)
        judge_dir = out_dir / "llama_harmbench_judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        print(f"[llama judge {idx}/{len(conditions)}] {condition.name} rows={len(rows)}", flush=True)
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
        rows_summary.append(
            {
                "condition": condition.name,
                "judge": "llama_harmbench",
                "overall": summary["overall"],
                "by_benchmark": summary["by_benchmark"],
            }
        )
        dump_json(args.output_root / rid / "gemma3_4b_it" / "llama_judge_grid_summary.json", {"rows": rows_summary})

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows_summary


def evaluate_openai_judge(args: argparse.Namespace, rid: str, conditions: list[Condition]) -> list[dict[str, Any]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set for the OpenAI judge.")
    os.environ.setdefault("TOGETHER_API_KEY", "unused-by-chatgpt-judge")
    os.environ["ALIGNTREE_OPENAI_JUDGE_MODEL"] = args.openai_judge_model
    if args.openai_judge_seed is not None:
        os.environ["ALIGNTREE_OPENAI_JUDGE_SEED"] = str(args.openai_judge_seed)

    sys.path.insert(0, str(ALIGNTREE))
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreaks  # noqa: WPS433
    from pipeline.submodules.evaluate_jailbreak import substring_matching_judge_fn  # noqa: WPS433

    rows_summary: list[dict[str, Any]] = []
    for idx, condition in enumerate(conditions, start=1):
        out_dir = condition_dir(args.output_root, rid, condition.name)
        completions_path = out_dir / "completions.jsonl"
        if not completions_path.exists():
            print(f"[openai judge skip missing {idx}/{len(conditions)}] {condition.name}", flush=True)
            continue
        rows = read_jsonl(completions_path)
        completions = [{"instruction": row["instruction"], "response": row["completion"]} for row in rows]
        print(f"[openai judge {idx}/{len(conditions)}] {condition.name} rows={len(rows)}", flush=True)
        refusal_prefilter_count = sum(1 for row in completions if substring_matching_judge_fn(row["response"]))
        evaluation = evaluate_jailbreaks(completions=completions, methodologies=("chatgpt",))
        labels = []
        # AlignTree returns refusal-prefiltered rows before OpenAI-judged rows.
        # The original list objects are mutated in place, so use that order for metadata.
        for row, judged in zip(rows, completions):
            label = int(judged.get("is_jailbreak_chatgpt", 0))
            labels.append(
                {
                    "sample_id": row["sample_id"],
                    "benchmark": row["benchmark"],
                    "condition": condition.name,
                    "label": label,
                    "label_name": "success" if label == 1 else "non_success",
                    "prompt_sha256": row.get("prompt_sha256"),
                    "completion_sha256": sha256_text(row.get("completion", "")),
                    "judge": args.openai_judge_model,
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
        judge_dir = out_dir / "openai_chatgpt_judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(judge_dir / "labels.jsonl", labels)
        dump_json(judge_dir / "summary.json", summary)
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
        rows_summary.append({"condition": condition.name, "judge": "openai_chatgpt", **summary})
        dump_json(args.output_root / rid / "gemma3_4b_it" / "openai_judge_grid_summary.json", {"rows": rows_summary})
    return rows_summary


def main() -> None:
    args = parse_args()
    rid = run_id(args.run_id)
    conditions = filter_conditions(build_conditions(), args.only_conditions)
    samples = load_aligntree_samples(args.n_test)
    root = args.output_root / rid / "gemma3_4b_it"
    root.mkdir(parents=True, exist_ok=True)
    dump_json(
        root / "grid_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": rid,
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
            "conditions": [condition.__dict__ | {"layers": list(condition.layers)} for condition in conditions],
        },
    )
    print(
        f"[grid] run_id={rid} samples={len(samples)} configs={len(conditions)} "
        f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    if not args.skip_generation:
        run_generation(args, rid, samples, conditions)
    if not args.skip_llama_judge:
        evaluate_llama_judge(args, rid, conditions)
    if not args.skip_openai_judge:
        evaluate_openai_judge(args, rid, conditions)
    print(f"[done] {root}", flush=True)


if __name__ == "__main__":
    main()
