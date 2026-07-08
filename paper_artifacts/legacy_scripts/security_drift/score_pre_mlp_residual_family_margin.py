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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from security_drift_common import (
    MODEL_DEFAULTS,
    OUTPUT_ROOT,
    batched,
    count_by,
    dtype_from_name,
    format_prompt,
    get_layers,
    model_defaults,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the final-prompt-token residual stream before each MLP "
            "is closer to the refusal or answerability token family."
        )
    )
    parser.add_argument("--model-alias", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--neuron-json", type=Path, default=None)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-prefix", default="pre_mlp_residual_family_margin")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument(
        "--outcomes",
        nargs="*",
        default=["failed_refusal"],
        choices=["failed_refusal", "successful_refusal", "invalid_judge"],
        help="Which HarmBench outcome groups to score.",
    )
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def label_fields(sample_id: str, labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label = labels.get(sample_id, {})
    raw_label = label.get("label")
    if raw_label == 1:
        drift_outcome = "failed_refusal"
    elif raw_label == 0:
        drift_outcome = "successful_refusal"
    elif raw_label == -1:
        drift_outcome = "invalid_judge"
    else:
        drift_outcome = None
    return {
        "harmbench_label": raw_label,
        "harmbench_label_name": label.get("label_name"),
        "drift_outcome": drift_outcome,
        "judge_completion_chars": label.get("completion_chars"),
    }


def family_mean_from_specs(lm_head_weight: torch.Tensor, specs: list[dict[str, Any]]) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for spec in specs:
        token_ids = [int(token_id) for token_id in spec.get("token_ids", [])]
        if not token_ids:
            continue
        ids = torch.tensor(token_ids, dtype=torch.long, device=lm_head_weight.device)
        vectors.append(lm_head_weight.index_select(0, ids).float().mean(dim=0))
    if not vectors:
        raise RuntimeError("No non-empty token specs found for family mean.")
    return torch.stack(vectors, dim=0).mean(dim=0)


def flat_token_ids(specs: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for spec in specs:
        ids.extend(int(token_id) for token_id in spec.get("token_ids", []))
    return list(dict.fromkeys(ids))


def direction_from_neuron_json(model: Any, neuron_json: Path) -> tuple[torch.Tensor, dict[str, Any], dict[str, list[int]]]:
    payload = json.loads(neuron_json.read_text(encoding="utf-8"))
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Could not find output embeddings.")
    lm_head_weight = lm_head.weight.detach()
    refusal_mean = family_mean_from_specs(lm_head_weight, payload.get("refusal_texts", []))
    answer_mean = family_mean_from_specs(lm_head_weight, payload.get("answerability_texts", []))
    direction = (refusal_mean - answer_mean).detach().float()
    metadata = {
        "direction_name": payload.get("direction_name"),
        "direction_norm_from_specs": float(direction.norm().item()),
        "refusal_text_count": len(payload.get("refusal_texts", [])),
        "answerability_text_count": len(payload.get("answerability_texts", [])),
        "definition": "mean unembedding(refusal token specs) - mean unembedding(answerability token specs)",
    }
    token_ids = {
        "refusal": flat_token_ids(payload.get("refusal_texts", [])),
        "answerability": flat_token_ids(payload.get("answerability_texts", [])),
    }
    return direction, metadata, token_ids


def get_final_norm(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "norm"):
            return language_model.norm
        if hasattr(language_model, "model") and hasattr(language_model.model, "norm"):
            return language_model.model.norm
    if hasattr(model, "language_model"):
        language_model = model.language_model
        if hasattr(language_model, "norm"):
            return language_model.norm
        if hasattr(language_model, "model") and hasattr(language_model.model, "norm"):
            return language_model.model.norm
    raise AttributeError("Could not find final language-model norm.")


def first_parameter_dtype(module: Any) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def family_margin_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    refusal_ids: torch.Tensor,
    answerability_ids: torch.Tensor,
) -> tuple[float, float, float]:
    hidden_f = hidden.float()
    ref_weight = lm_head_weight.index_select(0, refusal_ids).float()
    ans_weight = lm_head_weight.index_select(0, answerability_ids).float()
    ref_score = torch.logsumexp(hidden_f @ ref_weight.T, dim=-1)
    ans_score = torch.logsumexp(hidden_f @ ans_weight.T, dim=-1)
    margin = ref_score - ans_score
    return (
        float(margin.detach().cpu().item()),
        float(ref_score.detach().cpu().item()),
        float(ans_score.detach().cpu().item()),
    )


def cosine_and_dot(hidden: torch.Tensor, direction: torch.Tensor) -> tuple[float, float]:
    hidden_f = hidden.float()
    direction_f = direction.to(device=hidden_f.device).float()
    dot = torch.dot(hidden_f, direction_f)
    denom = hidden_f.norm() * direction_f.norm() + 1e-8
    cosine = dot / denom
    return float(cosine.detach().cpu().item()), float(dot.detach().cpu().item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
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


class PreMlpStateRecorder:
    def __init__(
        self,
        model: Any,
        sample_ids: list[str],
        prompt_lengths: list[int],
        raw_store: dict[str, dict[int, torch.Tensor]],
        mlp_input_store: dict[str, dict[int, torch.Tensor]],
    ) -> None:
        self.model = model
        self.sample_ids = sample_ids
        self.prompt_lengths = prompt_lengths
        self.raw_store = raw_store
        self.mlp_input_store = mlp_input_store
        self.handles: list[Any] = []

    def __enter__(self) -> "PreMlpStateRecorder":
        for layer_idx, layer in enumerate(get_layers(self.model)):
            self.handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    self._make_raw_pre_hook(layer_idx)
                )
            )
            self.handles.append(
                layer.mlp.register_forward_pre_hook(
                    self._make_mlp_input_pre_hook(layer_idx)
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        return False

    def _record_tensor(self, layer_idx: int, tensor: torch.Tensor, store: dict[str, dict[int, torch.Tensor]]) -> None:
        for batch_idx, sample_id in enumerate(self.sample_ids):
            token_idx = max(int(self.prompt_lengths[batch_idx]) - 1, 0)
            store[sample_id][layer_idx] = tensor[batch_idx, token_idx, :].detach().float().cpu()

    def _make_raw_pre_hook(self, layer_idx: int):
        def hook(_module, inputs):
            self._record_tensor(layer_idx, inputs[0], self.raw_store)

        return hook

    def _make_mlp_input_pre_hook(self, layer_idx: int):
        def hook(_module, inputs):
            self._record_tensor(layer_idx, inputs[0], self.mlp_input_store)

        return hook


def summarize_by_layer(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer in sorted({int(row["layer"]) for row in rows}):
        vals = [float(row[column]) for row in rows if int(row["layer"]) == layer]
        if not vals:
            continue
        refusal_like = sum(1 for value in vals if value > 0.0)
        answerability_like = sum(1 for value in vals if value < 0.0)
        out.append(
            {
                "layer": layer,
                "n": len(vals),
                "mean": mean(vals),
                "median": percentile(vals, 0.5),
                "p25": percentile(vals, 0.25),
                "p75": percentile(vals, 0.75),
                "min": min(vals),
                "max": max(vals),
                "refusal_like_n": refusal_like,
                "answerability_like_n": answerability_like,
                "zero_n": len(vals) - refusal_like - answerability_like,
                "refusal_like_frac": refusal_like / len(vals),
                "answerability_like_frac": answerability_like / len(vals),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    defaults = model_defaults(args.model_alias)
    model_path = args.model_path or defaults["model_path"]
    neuron_json = args.neuron_json or defaults["neuron_json"]
    out_dir = args.output_root / args.run_id / args.model_alias
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels_jsonl)
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.rows_jsonl):
        merged = {**row, **label_fields(str(row["sample_id"]), labels)}
        if merged.get("drift_outcome") not in set(args.outcomes):
            continue
        if args.benchmarks and merged.get("benchmark") not in set(args.benchmarks):
            continue
        rows.append(merged)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows matched the requested filters.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Match the existing safety tracing scripts: with right padding,
    # prompt_length - 1 is the final non-padding prompt token.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    final_norm = get_final_norm(model)
    norm_dtype = first_parameter_dtype(final_norm)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise AttributeError("Could not find output embeddings.")
    lm_head_weight = lm_head.weight.detach()
    direction, direction_metadata, token_ids = direction_from_neuron_json(model, neuron_json)
    refusal_ids = torch.tensor(token_ids["refusal"], dtype=torch.long, device=lm_head_weight.device)
    answerability_ids = torch.tensor(token_ids["answerability"], dtype=torch.long, device=lm_head_weight.device)

    all_rows: list[dict[str, Any]] = []
    layer_count = len(get_layers(model))
    for batch_index, batch in enumerate(batched(rows, args.batch_size), start=1):
        prompts = [format_prompt(args.model_alias, str(row["instruction"])) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        sample_ids = [str(row["sample_id"]) for row in batch]
        raw_store: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        mlp_input_store: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
        with torch.inference_mode(), PreMlpStateRecorder(
            model,
            sample_ids,
            [int(length) for length in prompt_lengths],
            raw_store,
            mlp_input_store,
        ):
            _ = model(**encoded, use_cache=False)

        for row in batch:
            sample_id = str(row["sample_id"])
            for layer in range(layer_count):
                raw_hidden = raw_store[sample_id][layer].to(lm_head_weight.device)
                mlp_input = mlp_input_store[sample_id][layer].to(lm_head_weight.device)
                normed_raw = final_norm(raw_hidden.to(lm_head_weight.device, dtype=norm_dtype)).float()
                raw_margin, raw_ref_score, raw_ans_score = family_margin_from_hidden(
                    normed_raw,
                    lm_head_weight,
                    refusal_ids,
                    answerability_ids,
                )
                raw_cos, raw_dot = cosine_and_dot(raw_hidden, direction)
                mlp_input_margin, mlp_ref_score, mlp_ans_score = family_margin_from_hidden(
                    mlp_input,
                    lm_head_weight,
                    refusal_ids,
                    answerability_ids,
                )
                mlp_cos, mlp_dot = cosine_and_dot(mlp_input, direction)
                all_rows.append(
                    {
                        "sample_id": sample_id,
                        "benchmark": row.get("benchmark"),
                        "dataset_index": row.get("dataset_index"),
                        "drift_outcome": row.get("drift_outcome"),
                        "harmbench_label": row.get("harmbench_label"),
                        "layer": layer,
                        "raw_pre_mlp_margin": raw_margin,
                        "raw_pre_mlp_closer_to": "refusal" if raw_margin > 0 else "answerability" if raw_margin < 0 else "tie",
                        "raw_pre_mlp_refusal_logsumexp": raw_ref_score,
                        "raw_pre_mlp_answerability_logsumexp": raw_ans_score,
                        "raw_pre_mlp_cosine_d": raw_cos,
                        "raw_pre_mlp_dot_d": raw_dot,
                        "mlp_input_margin": mlp_input_margin,
                        "mlp_input_closer_to": "refusal" if mlp_input_margin > 0 else "answerability" if mlp_input_margin < 0 else "tie",
                        "mlp_input_refusal_logsumexp": mlp_ref_score,
                        "mlp_input_answerability_logsumexp": mlp_ans_score,
                        "mlp_input_cosine_d": mlp_cos,
                        "mlp_input_dot_d": mlp_dot,
                    }
                )
        print(
            f"[pre-mlp] model={args.model_alias} batch={batch_index} "
            f"seen={min(batch_index * args.batch_size, len(rows))}/{len(rows)}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_by_layer = summarize_by_layer(all_rows, "raw_pre_mlp_margin")
    mlp_input_by_layer = summarize_by_layer(all_rows, "mlp_input_margin")
    window_by_model = {
        "gemma3_4b_it": [20, 21, 22],
        "llama3p2_3b_instruct": [18, 19, 20, 21],
    }
    window = window_by_model.get(args.model_alias, [])
    window_rows = [row for row in all_rows if int(row["layer"]) in set(window)]
    start_layer = window[0] if window else None
    start_rows = [row for row in all_rows if start_layer is not None and int(row["layer"]) == start_layer]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": args.model_alias,
        "model_path": model_path,
        "rows_jsonl": str(args.rows_jsonl),
        "labels_jsonl": str(args.labels_jsonl),
        "neuron_json": str(neuron_json),
        "sample_count": len(rows),
        "counts_by_benchmark": count_by(rows, "benchmark"),
        "counts_by_outcome": count_by(rows, "drift_outcome"),
        "direction_metadata": direction_metadata,
        "target_token_ids": token_ids,
        "definition": {
            "token_position": "final prompt token in the prefill pass",
            "raw_pre_mlp_state": "residual stream after attention and before post_attention_layernorm in each decoder block",
            "mlp_input_state": "normalized state passed directly into the MLP",
            "raw_pre_mlp_margin": "logsumexp(unembed(final_norm(raw_pre_mlp_state))[refusal_ids]) - logsumexp(...[answerability_ids])",
            "closer_rule": "positive margin is refusal-like; negative margin is answerability-like",
        },
        "safety_drift_window": window,
        "raw_pre_mlp_by_layer": raw_by_layer,
        "mlp_input_by_layer": mlp_input_by_layer,
        "raw_pre_mlp_window_summary": summarize_by_layer(window_rows, "raw_pre_mlp_margin"),
        "mlp_input_window_summary": summarize_by_layer(window_rows, "mlp_input_margin"),
        "raw_pre_mlp_start_layer_summary": summarize_by_layer(start_rows, "raw_pre_mlp_margin"),
        "mlp_input_start_layer_summary": summarize_by_layer(start_rows, "mlp_input_margin"),
    }

    prefix = args.output_prefix
    write_csv(out_dir / f"{prefix}_layer_scores.csv", all_rows)
    write_json(out_dir / f"{prefix}_summary.json", summary)
    print(out_dir / f"{prefix}_summary.json", flush=True)


if __name__ == "__main__":
    main()
