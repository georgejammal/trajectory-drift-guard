#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from security_drift_common import (
    MODEL_DEFAULTS,
    OUTPUT_ROOT,
    append_jsonl,
    batched,
    count_by,
    dtype_from_name,
    format_prompt,
    get_layers,
    load_neurons,
    model_defaults,
    parse_layers,
    read_jsonl,
    sha256_text,
    write_json,
)


@dataclass(frozen=True)
class Condition:
    name: str
    mode: str
    layers: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline, original abs(a), or negative abs(a) completions for a manifest under try_and_change."
    )
    parser.add_argument("--model-alias", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--neuron-json", type=Path, default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conditions", nargs="+", choices=["baseline", "abs", "neg_abs"], default=["baseline"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows
    if args.benchmarks:
        keep = set(args.benchmarks)
        selected = [row for row in selected if row.get("benchmark") in keep]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


class AbsActivationIntervention:
    def __init__(self, model: Any, neurons_by_layer: dict[int, list[dict[str, Any]]], mode: str) -> None:
        self.model = model
        self.neurons_by_layer = neurons_by_layer
        self.mode = mode
        self.original_forwards: dict[int, Any] = {}
        self.cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "AbsActivationIntervention":
        if self.mode == "baseline":
            return self
        if self.mode not in {"abs", "neg_abs"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        layers = get_layers(self.model)
        for layer_idx, rows in self.neurons_by_layer.items():
            mlp = layers[layer_idx].mlp
            self.original_forwards[layer_idx] = mlp.forward
            indices = [int(row["neuron"]) for row in rows]

            def patched_forward(module, x, patch_layer=layer_idx, patch_indices=indices):
                gate_act = module.act_fn(module.gate_proj(x))
                gated = gate_act * module.up_proj(x)
                idx = self._indices(patch_layer, patch_indices, gated.device)
                target = gated[..., idx]
                patched = gated.clone()
                if self.mode == "abs":
                    patched[..., idx] = target.abs()
                elif self.mode == "neg_abs":
                    patched[..., idx] = -target.abs()
                return module.down_proj(patched)

            mlp.forward = MethodType(patched_forward, mlp)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.mode != "baseline":
            layers = get_layers(self.model)
            for layer_idx, original_forward in self.original_forwards.items():
                layers[layer_idx].mlp.forward = original_forward
        return False

    def _indices(self, layer_idx: int, indices: list[int], device: torch.device) -> torch.Tensor:
        key = (layer_idx, device)
        tensor = self.cache.get(key)
        if tensor is None:
            tensor = torch.tensor(indices, dtype=torch.long, device=device)
            self.cache[key] = tensor
        return tensor


def build_conditions(raw_conditions: list[str], layers: tuple[int, ...]) -> list[Condition]:
    return [Condition(name=name, mode=name, layers=layers) for name in raw_conditions]


def condition_dir(output_root: Path, run_id: str, model_alias: str, condition: str) -> Path:
    return output_root / run_id / model_alias / condition


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def generate_condition(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    model_alias: str,
    condition: Condition,
    neurons_by_layer: dict[int, list[dict[str, Any]]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    resume: bool,
) -> list[dict[str, Any]]:
    done = load_done(output_path) if resume else {}
    if output_path.exists() and not resume:
        output_path.unlink()
    outputs = list(done.values())
    pending = [row for row in rows if str(row["sample_id"]) not in done]
    start = time.time()
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    with AbsActivationIntervention(model, neurons_by_layer, condition.mode):
        for batch_idx, batch in enumerate(batched(pending, batch_size), start=1):
            prompts = [format_prompt(model_alias, str(row["instruction"])) for row in batch]
            inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                    generation_config=generation_config,
                )
            input_len = int(inputs.input_ids.shape[1])
            decoded = tokenizer.batch_decode(
                generated[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample, completion in zip(batch, decoded):
                text = completion.strip()
                out = {
                    **sample,
                    "model_alias": model_alias,
                    "condition": condition.name,
                    "completion": text,
                    "response": text,
                    "completion_sha256": sha256_text(text),
                    "max_new_tokens": max_new_tokens,
                }
                outputs.append(out)
                append_jsonl(output_path, out)
            print(
                f"[generate] model={model_alias} condition={condition.name} "
                f"batch={batch_idx} seen={len(outputs)}/{len(rows)} elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return outputs


def main() -> None:
    args = parse_args()
    defaults = model_defaults(args.model_alias)
    model_path = args.model_path or str(defaults["model_path"])
    neuron_json = args.neuron_json or Path(defaults["neuron_json"])
    layers = parse_layers(args.layers or str(defaults["layers"]))
    rows = filter_rows(read_jsonl(args.manifest_jsonl), args)
    if not rows:
        raise RuntimeError("No manifest rows selected.")

    run_root = args.output_root / args.run_id / args.model_alias
    run_root.mkdir(parents=True, exist_ok=True)
    neurons_by_layer, neuron_metadata = load_neurons(neuron_json, layers)
    conditions = build_conditions(args.conditions, layers)
    write_json(
        run_root / "generation_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "model_alias": args.model_alias,
            "model_path": model_path,
            "manifest_jsonl": str(args.manifest_jsonl),
            "selected_samples": len(rows),
            "counts_by_benchmark": count_by(rows, "benchmark"),
            "conditions": [condition.__dict__ | {"layers": list(condition.layers)} for condition in conditions],
            "generation": {
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "attn_implementation": args.attn_implementation,
            },
            "intervention": {
                "abs_definition": "replace selected gated MLP coordinates a with abs(a) before down_proj; no ReLU and no gate-positive mask",
                "neg_abs_definition": "replace selected gated MLP coordinates a with -abs(a) before down_proj; no ReLU and no gate-positive mask",
                "site": "mlp gated intermediate act(gate_proj(x)) * up_proj(x), immediately before down_proj",
                "neuron_metadata": neuron_metadata,
            },
        },
    )

    print(f"[load tokenizer] {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
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

    for condition in conditions:
        out_dir = condition_dir(args.output_root, args.run_id, args.model_alias, condition.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / "manifest.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": args.run_id,
                "model_alias": args.model_alias,
                "condition": condition.__dict__ | {"layers": list(condition.layers)},
                "sample_count": len(rows),
                "manifest_jsonl": str(args.manifest_jsonl),
            },
        )
        outputs = generate_condition(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            model_alias=args.model_alias,
            condition=condition,
            neurons_by_layer=neurons_by_layer,
            output_path=out_dir / "completions.jsonl",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )
        write_json(
            out_dir / "native_summary.json",
            {
                "total": len(outputs),
                "empty_completions": sum(1 for row in outputs if not row.get("completion")),
                "counts_by_benchmark": count_by(outputs, "benchmark"),
                "mean_completion_chars": (
                    sum(len(str(row.get("completion", ""))) for row in outputs) / len(outputs) if outputs else None
                ),
            },
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[done] {run_root}", flush=True)


if __name__ == "__main__":
    main()
