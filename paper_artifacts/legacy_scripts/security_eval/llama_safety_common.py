#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safety_benchmark_common import append_jsonl, dump_json


REPO = Path("/home/georgejammal/projects/semantic-to-symbolic")
LLAMA32_3B_INSTRUCT_PATH = (
    "/home/georgejammal/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/"
    "snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
)
OUT_ROOT = REPO / "safeguard_eval" / "outputs"
NEURON_ROOT = REPO / "safeguard_logit_lens" / "neuron_sets"


REFUSAL_TARGET_TEXTS = [
    "I",
    " I",
    "Sorry",
    " sorry",
    "Unfortunately",
    " unfortunately",
    "cannot",
    " cannot",
    "can't",
    " can't",
    "unable",
    " unable",
    "refuse",
    " refuse",
    "safe",
    " safety",
]

ANSWERABILITY_TARGET_TEXTS = [
    "Sure",
    " sure",
    "Yes",
    " yes",
    "Certainly",
    " certainly",
    "Here",
    " here",
    "can",
    " can",
    "provide",
    " provide",
    "steps",
    " steps",
    "code",
    " code",
    "tutorial",
    " tutorial",
]


def run_id(raw: str | None) -> str:
    return raw or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: str, dtype: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(dtype),
        device_map="auto",
    ).eval()


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_language_model(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("Could not find language model.")


def get_layers(model: Any) -> Any:
    language_model = get_language_model(model)
    if hasattr(language_model, "layers"):
        return language_model.layers
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        return language_model.model.layers
    raise AttributeError("Could not find decoder layers.")


def get_final_norm(model: Any) -> Any:
    language_model = get_language_model(model)
    if hasattr(language_model, "norm"):
        return language_model.norm
    if hasattr(language_model, "model") and hasattr(language_model.model, "norm"):
        return language_model.model.norm
    raise AttributeError("Could not find final norm.")


def upper_half_layers(model: Any, exclude_last: int = 0) -> list[int]:
    layers = get_layers(model)
    end = len(layers) - int(exclude_last)
    start = len(layers) // 2
    return list(range(start, end))


def family_vector(tokenizer: Any, lm_head_weight: torch.Tensor, texts: list[str]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    vectors = []
    specs: list[dict[str, Any]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        vec = lm_head_weight[torch.tensor(ids, device=lm_head_weight.device)].mean(dim=0)
        vectors.append(vec)
        specs.append(
            {
                "text": text,
                "token_ids": [int(token_id) for token_id in ids],
                "vector_norm": float(vec.float().norm().item()),
            }
        )
    if not vectors:
        raise RuntimeError("No target texts tokenized.")
    return torch.stack(vectors, dim=0).mean(dim=0), specs


def load_neurons(path: Path, top_k: int | None, score_key: str, weighting: str) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload["top_neurons"])
    if top_k is not None:
        rows = rows[:top_k]
    max_abs = max((abs(float(row.get(score_key, 0.0))) for row in rows), default=1.0) or 1.0
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    n = len(rows)
    for rank, row in enumerate(rows, start=1):
        if weighting == "rank":
            weight = 1.0 - ((rank - 1) / max(n - 1, 1))
        elif weighting == "score":
            weight = abs(float(row.get(score_key, 0.0))) / max_abs
        else:
            weight = 1.0
        by_layer[int(row["layer"])].append(
            {
                "neuron": int(row["neuron"]),
                "weight": float(weight),
                "rank": rank,
                "score": float(row.get(score_key, 0.0)),
            }
        )
    metadata = {
        "path": str(path),
        "top_k": top_k,
        "loaded": sum(len(v) for v in by_layer.values()),
        "score_key": score_key,
        "score_weighting": weighting,
        "source_direction": payload.get("direction_name"),
        "source_score_definition": payload.get("score_definition"),
        "layer_counts": {str(layer): len(items) for layer, items in sorted(by_layer.items())},
    }
    return dict(sorted(by_layer.items())), metadata


class MLPActivationIntervention:
    def __init__(
        self,
        model: AutoModelForCausalLM,
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

    def __enter__(self):
        if self.mode == "none":
            return self
        layers = get_layers(self.model)
        for layer_idx, rows in self.neurons_by_layer.items():
            mlp = layers[layer_idx].mlp
            self.original_forwards[layer_idx] = mlp.forward
            indices = torch.tensor([row["neuron"] for row in rows], dtype=torch.long)
            weights = torch.tensor([row["weight"] for row in rows], dtype=torch.float32)
            mode = self.mode
            factor = float(self.scale_factor)
            gate_only = bool(self.gate_positive_only)

            def patched_forward(module, x, patch_indices=indices, patch_weights=weights, patch_mode=mode, patch_factor=factor, patch_gate_only=gate_only):
                gate_act = module.act_fn(module.gate_proj(x))
                gated = gate_act * module.up_proj(x)
                idx = patch_indices.to(gated.device)
                w = patch_weights.to(device=gated.device, dtype=gated.dtype).view(*([1] * (gated.ndim - 1)), -1)
                target = gated[..., idx]
                if patch_mode == "scale":
                    replacement = target * patch_factor
                elif patch_mode == "abs":
                    replacement = target.abs()
                elif patch_mode == "weighted_scale":
                    replacement = target * (1.0 + (patch_factor - 1.0) * w)
                elif patch_mode == "weighted_abs":
                    replacement = target + w * (target.abs() - target)
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
        layers = get_layers(self.model)
        for layer_idx, original_forward in self.original_forwards.items():
            layers[layer_idx].mlp.forward = original_forward
        return False


def generate_completions(
    *,
    model: AutoModelForCausalLM,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    intervention: MLPActivationIntervention,
    condition: str,
) -> list[dict[str, Any]]:
    if output_path.exists():
        output_path.unlink()
    rows_out: list[dict[str, Any]] = []
    start = time.time()
    with intervention:
        for batch_idx, batch in enumerate(batched(samples, batch_size), start=1):
            prompts = [build_chat_prompt(tokenizer, row["prompt"]) for row in batch]
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
            decoded = tokenizer.batch_decode(
                generated[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample, completion in zip(batch, decoded):
                row = {
                    **sample,
                    "condition": condition,
                    "completion": completion.strip(),
                }
                rows_out.append(row)
                append_jsonl(output_path, row)
            print(
                f"[generate {condition}] batch={batch_idx} seen={len(rows_out)}/{len(samples)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows_out


def write_generation_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    intervention = manifest["intervention"]
    neuron_metadata = intervention.get("neuron_metadata") or {}
    readme = f"""# Llama Safety Generation Output

## Generation Setup
- Model alias: `{manifest["model_alias"]}`
- Model path: `{manifest["model_path"]}`
- Sample set: `{manifest["sample_set"]}`
- Sample count: `{manifest["sample_count"]}`
- Prompting: {manifest["prompting"]}
- Batch size: `{manifest["generation"]["batch_size"]}`
- Max new tokens: `{manifest["generation"]["max_new_tokens"]}`
- Decoding: greedy (`do_sample=False`)

## Intervention Setup
- Condition: `{manifest["condition"]}`
- Mode: `{intervention["mode"]}`
- Scale factor: `{intervention["scale_factor"]}`
- Gate-positive only: `{intervention["gate_positive_only"]}`
- Site: {intervention["site"]}
- Token scope: {intervention["token_scope"]}
- Neuron set: `{neuron_metadata.get("path")}`
- Top-k: `{neuron_metadata.get("top_k")}`
- Source direction: `{neuron_metadata.get("source_direction")}`
- Score key: `{neuron_metadata.get("score_key")}`
- Score weighting: `{neuron_metadata.get("score_weighting")}`
"""
    output_dir.joinpath("README.md").write_text(readme, encoding="utf-8")
