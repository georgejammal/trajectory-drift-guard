#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("qwen-vl-utils is required for Qwen2.5-VL scoring.") from exc


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get(
    "QWEN25VL_3B_INSTRUCT_PATH",
    "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/"
    "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
)
DATASETS = {
    "countbenchqa": str(ARTIFACT_ROOT / "data" / "countbenchqa"),
    "how_many": str(ARTIFACT_ROOT / "data" / "how_many"),
    "tallyqa_balanced": str(ARTIFACT_ROOT / "data" / "tallyqa_balanced"),
}
DEFAULT_PREDICTIONS_ROOT = (
    ARTIFACT_ROOT
    / "outputs/experiment_runs/counting/it_digits1_9_union_top10_scaling_abs_suffix_one_number/qwen2_5_vl_3b_instruct"
)
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "outputs" / "experiment_runs" / "counting"
DEFAULT_SUFFIX = " Answer with one number do not add any further explanations."

NUMBER_WORDS_EN = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
NUMBER_WORDS_ZH = {
    1: "\u4e00",
    2: "\u4e8c",
    3: "\u4e09",
    4: "\u56db",
    5: "\u4e94",
    6: "\u516d",
    7: "\u4e03",
    8: "\u516b",
    9: "\u4e5d",
}

FIXED_SET_PATHS = {
    "fixed_en_dmw_top500": (
        ARTIFACT_ROOT
        / "resources/neuron_sets/counting/qwen2p5_mean_digit_minus_word_top100_500_no_gate/"
        / "qwen2p5_vl_3b_it_mean_digit_minus_word_top500_cosine_upper_half_to_end.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL counting drift sweep over token positions, windows, directions, and neuron sets."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--prediction-subdir", default="gold_1_9_union_top10")
    parser.add_argument("--prediction-condition", default="factor_1p0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--datasets", default="countbenchqa,how_many,tallyqa_balanced")
    parser.add_argument("--split", default="test")
    parser.add_argument("--digits", default="2,3,4,5,6,7,8,9")
    parser.add_argument("--samples-per-digit-per-dataset", type=int, default=30)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=35)
    parser.add_argument("--min-window-width", type=int, default=1)
    parser.add_argument("--max-window-width", default="all")
    parser.add_argument(
        "--position-modes",
        default=(
            "final_m0,final_m1,final_m2,final_m3,final_m4,final_m5,final_m6,"
            "final_m7,final_m8,final_m9,final_m10,question_mark,suffix_period"
        ),
    )
    parser.add_argument(
        "--direction-families",
        default=(
            "per_en_to_digit,per_zh_to_digit,per_pooled_to_digit,"
            "mean_en_to_digit,mean_zh_to_digit,mean_pooled_to_digit"
        ),
    )
    parser.add_argument("--norms", default="raw,unit")
    parser.add_argument(
        "--selection-modes",
        default=(
            "left2sd,left3sd,right2sd,right3sd,"
            "fixed_en_dmw_top500"
        ),
    )
    parser.add_argument("--save-score-tensor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_id(raw: str | None) -> str:
    return raw or "qwen_counting_digit_drift_position_sweep_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_dataset(loaded: Any, split: str):
    return loaded[split] if isinstance(loaded, DatasetDict) else loaded


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def parse_number_text(text: str) -> tuple[int | None, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    digit_match = re.search(r"\b([0-9]+)\b", lowered)
    if digit_match:
        return int(digit_match.group(1)), "arabic_digit"
    for word, value in {
        **{v: k for k, v in NUMBER_WORDS_EN.items()},
        "zero": 0,
        "ten": 10,
    }.items():
        if re.search(rf"\b{word}\b", lowered):
            return value, "english_word"
    for word, value in {
        "\u96f6": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e24": 2,
        "\u4fe9": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
        "\u5341": 10,
    }.items():
        if word in stripped:
            return value, "chinese_word"
    return None, "unparsed"


def row_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text", "prompt"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find question field in row keys: {sorted(row)}")


def image_from_row(row: dict[str, Any]) -> Image.Image:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image is not None and hasattr(image, "convert"):
        return image.convert("RGB")
    for key in ("image_path", "path", "filename", "file_name"):
        value = row.get(key)
        if value:
            return Image.open(value).convert("RGB")
    raise KeyError(f"Could not find image field in row keys: {sorted(row)}")


def gold_from_row(row: dict[str, Any]) -> int:
    for key in ("number", "gold", "answer", "label", "count"):
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                parsed, _ = parse_number_text(value)
                if parsed is not None:
                    return parsed
            return int(value)
    raise KeyError(f"Could not find gold field in row keys: {sorted(row)}")


def build_messages(image: Image.Image, question: str, suffix: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{question}{suffix}"},
            ],
        }
    ]


def prepare_inputs(processor: Any, rows: list[dict[str, Any]], suffix: str) -> Any:
    messages = [build_messages(image_from_row(row), row_question(row), suffix) for row in rows]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def choose_samples(
    *,
    predictions_root: Path,
    prediction_subdir: str,
    prediction_condition: str,
    dataset_names: list[str],
    split: str,
    digits: list[int],
    per_digit_per_dataset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, int]]]:
    samples: list[dict[str, Any]] = []
    dataset_cache: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    digit_set = set(digits)

    for dataset_name in dataset_names:
        dataset = as_dataset(load_from_disk(DATASETS[dataset_name]), split)
        dataset_cache[dataset_name] = dataset
        predictions_path = predictions_root / dataset_name / prediction_subdir / prediction_condition / "predictions.jsonl"
        predictions = read_jsonl(predictions_path)
        prediction_golds = sorted({int(pred["gold"]) for pred in predictions})
        filtered_source_indices = [
            idx for idx, row in enumerate(dataset) if int(gold_from_row(row)) in set(prediction_golds)
        ]
        chosen: dict[int, list[dict[str, Any]]] = {digit: [] for digit in digits}

        for pred in predictions:
            gold = int(pred["gold"])
            if gold not in digit_set or len(chosen[gold]) >= per_digit_per_dataset:
                continue
            if "source_index" in pred:
                source_index = int(pred["source_index"])
            else:
                filtered_index = int(pred["dataset_index"])
                source_index = filtered_index
                if source_index >= len(dataset) or int(gold_from_row(dataset[source_index])) != gold:
                    if filtered_index >= len(filtered_source_indices):
                        raise ValueError(
                            f"Filtered dataset index {filtered_index} out of range for {dataset_name} "
                            f"with {len(filtered_source_indices)} filtered rows"
                        )
                    source_index = int(filtered_source_indices[filtered_index])
            row = dataset[source_index]
            if int(gold_from_row(row)) != gold:
                raise ValueError(f"Gold mismatch in {dataset_name}:{source_index}")
            chosen[gold].append(
                {
                    "dataset": dataset_name,
                    "source_index": source_index,
                    "question": pred["question"],
                    "gold": gold,
                    "baseline_prediction_text": pred["prediction_text"],
                    "baseline_prediction_number": pred.get("prediction_number"),
                    "baseline_correct": bool(pred["correct"]),
                }
            )

        counts[dataset_name] = {str(digit): len(chosen[digit]) for digit in digits}
        for digit in digits:
            if len(chosen[digit]) != per_digit_per_dataset:
                raise ValueError(
                    f"Expected {per_digit_per_dataset} samples for {dataset_name} digit={digit}; got {len(chosen[digit])}"
                )
            samples.extend(chosen[digit])
    return samples, dataset_cache, counts


def get_lm_head_weight(model: Any) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach().float()
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is not None and hasattr(language_model, "lm_head"):
        return language_model.lm_head.weight.detach().float()
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head.weight.detach().float()
    raise AttributeError("Could not find lm_head.weight on Qwen model.")


def token_mean(tokenizer: Any, unembed: torch.Tensor, text: str) -> tuple[torch.Tensor, list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"No token ids for {text!r}")
    token_ids = torch.tensor(ids, dtype=torch.long, device=unembed.device)
    return unembed[token_ids].float().mean(dim=0), [int(x) for x in ids]


def build_direction_bank(
    tokenizer: Any,
    unembed: torch.Tensor,
    digits: list[int],
    families: list[str],
    norms: list[str],
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    per_digit_base: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    metadata: dict[str, Any] = {"digits": digits, "token_ids": {}, "direction_norms": {}}

    for digit in digits:
        digit_vec, digit_ids = token_mean(tokenizer, unembed, str(digit))
        en_vec, en_ids = token_mean(tokenizer, unembed, NUMBER_WORDS_EN[digit])
        zh_vec, zh_ids = token_mean(tokenizer, unembed, NUMBER_WORDS_ZH[digit])
        pooled_word = 0.5 * (en_vec + zh_vec)
        metadata["token_ids"][str(digit)] = {
            "digit": digit_ids,
            "en_word": en_ids,
            "zh_word": zh_ids,
            "en_text": NUMBER_WORDS_EN[digit],
            "zh_text": NUMBER_WORDS_ZH[digit],
        }
        per_digit_base["per_en_to_digit"][digit] = digit_vec - en_vec
        per_digit_base["per_zh_to_digit"][digit] = digit_vec - zh_vec
        per_digit_base["per_pooled_to_digit"][digit] = digit_vec - pooled_word

    mean_sources = {
        "mean_en_to_digit": "per_en_to_digit",
        "mean_zh_to_digit": "per_zh_to_digit",
        "mean_pooled_to_digit": "per_pooled_to_digit",
        "mean_unitcomp_en_to_digit": "per_en_to_digit",
        "mean_unitcomp_zh_to_digit": "per_zh_to_digit",
        "mean_unitcomp_pooled_to_digit": "per_pooled_to_digit",
    }
    for family, source in mean_sources.items():
        components = [per_digit_base[source][digit] for digit in digits]
        if family.startswith("mean_unitcomp"):
            components = [F.normalize(component, dim=0) for component in components]
        mean_vec = torch.stack(components, dim=0).mean(dim=0)
        for digit in digits:
            per_digit_base[family][digit] = mean_vec

    direction_bank: dict[str, dict[int, torch.Tensor]] = {}
    for family in families:
        if family not in per_digit_base:
            raise ValueError(f"Unknown direction family {family}; available={sorted(per_digit_base)}")
        for norm in norms:
            if norm not in {"raw", "unit"}:
                raise ValueError(f"Unknown norm {norm}; use raw or unit")
            key = f"{family}:{norm}"
            direction_bank[key] = {}
            for digit in digits:
                raw = per_digit_base[family][digit].detach().float()
                direction_bank[key][digit] = raw if norm == "raw" else F.normalize(raw, dim=0)
            metadata["direction_norms"][key] = {
                str(digit): float(per_digit_base[family][digit].norm().item())
                for digit in digits
            }
    return direction_bank, metadata


def infer_layer_base(payload: dict[str, Any]) -> str:
    if "global_top_neurons" in payload and "top_neurons" not in payload:
        return "one"
    return "zero"


def load_fixed_sets(selection_modes: list[str]) -> tuple[dict[str, dict[int, set[int]]], dict[str, Any]]:
    fixed: dict[str, dict[int, set[int]]] = {}
    metadata: dict[str, Any] = {}
    for mode in selection_modes:
        if not mode.startswith("fixed_"):
            continue
        path = FIXED_SET_PATHS.get(mode)
        if path is None:
            raise ValueError(f"No fixed set path registered for {mode}")
        payload = read_json(path)
        rows = payload.get("top_neurons") or payload.get("global_top_neurons")
        if rows is None:
            raise ValueError(f"Expected top_neurons or global_top_neurons in {path}")
        base = infer_layer_base(payload)
        by_layer: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            layer = int(row.get("layer_index", row.get("layer")))
            if base == "one":
                layer -= 1
            neuron = int(row.get("neuron_index", row.get("neuron")))
            by_layer[layer].add(neuron)
        fixed[mode] = dict(by_layer)
        metadata[mode] = {
            "path": str(path),
            "layer_index_base": base,
            "n_neurons": sum(len(v) for v in by_layer.values()),
            "layer_counts": {str(layer): len(vals) for layer, vals in sorted(by_layer.items())},
            "source_summary": {
                key: payload.get(key)
                for key in (
                    "score_method",
                    "score_definition",
                    "language",
                    "digits",
                    "top_k",
                    "restrict_layers",
                    "direction_name",
                )
                if key in payload
            },
        }
    return fixed, metadata


def build_q_matrices(
    *,
    model: Any,
    direction_bank: dict[str, dict[int, torch.Tensor]],
    digits: list[int],
    layers_to_use: list[int],
    selection_modes: list[str],
    fixed_sets: dict[str, dict[int, set[int]]],
) -> tuple[dict[int, torch.Tensor], list[str], dict[str, Any]]:
    d_model = int(next(iter(next(iter(direction_bank.values())).values())).numel())
    thresholds = {
        "right1sd": 1.0 / math.sqrt(d_model),
        "right2sd": 2.0 / math.sqrt(d_model),
        "right3sd": 3.0 / math.sqrt(d_model),
        "left1sd": 1.0 / math.sqrt(d_model),
        "left2sd": 2.0 / math.sqrt(d_model),
        "left3sd": 3.0 / math.sqrt(d_model),
        "abs1sd": 1.0 / math.sqrt(d_model),
        "abs2sd": 2.0 / math.sqrt(d_model),
        "abs3sd": 3.0 / math.sqrt(d_model),
    }
    dynamic_modes = set(thresholds)
    fixed_modes = set(fixed_sets)
    unknown = sorted(set(selection_modes) - dynamic_modes - fixed_modes)
    if unknown:
        raise ValueError(f"Unknown selection modes {unknown}")

    condition_names = [
        f"{direction_key}|{selection_mode}"
        for direction_key in sorted(direction_bank)
        for selection_mode in selection_modes
    ]
    layers = model.model.language_model.layers
    device = next(model.parameters()).device
    q_by_layer: dict[int, torch.Tensor] = {}
    meta = {
        "d_model": d_model,
        "thresholds": thresholds,
        "conditions": {
            name: {"total_by_digit": {str(d): 0 for d in digits}, "layer_counts_by_digit": {str(d): {} for d in digits}}
            for name in condition_names
        },
    }

    for layer_idx in layers_to_use:
        weight = layers[layer_idx].mlp.down_proj.weight.detach().float()
        col_norms = torch.linalg.norm(weight, dim=0).clamp_min(1e-12)
        rows = []
        for condition in condition_names:
            direction_key, selection_mode = condition.split("|", 1)
            for digit in digits:
                scoring_direction = direction_bank[direction_key][digit].to(device).float()
                selection_direction = F.normalize(scoring_direction, dim=0)
                q = (weight.T @ scoring_direction).detach().float()
                if selection_mode in dynamic_modes:
                    cos = (weight.T @ selection_direction).detach().float() / col_norms
                    threshold = thresholds[selection_mode]
                    if selection_mode.startswith("right"):
                        mask = cos > threshold
                    elif selection_mode.startswith("left"):
                        mask = cos < -threshold
                    else:
                        mask = cos.abs() > threshold
                else:
                    mask = torch.zeros_like(q, dtype=torch.bool)
                    selected = sorted(fixed_sets[selection_mode].get(layer_idx, set()))
                    if selected:
                        mask[torch.tensor(selected, device=mask.device, dtype=torch.long)] = True
                rows.append(torch.where(mask, q, torch.zeros_like(q)).detach().cpu().float())
                count = int(mask.sum().item())
                condition_digit_key = f"{condition}|d{digit}"
                meta["conditions"].setdefault(condition_digit_key, {"total": 0, "layer_counts": {}})
                meta["conditions"][condition_digit_key]["total"] += count
                meta["conditions"][condition_digit_key]["layer_counts"][str(layer_idx)] = count
        q_by_layer[layer_idx] = torch.stack(rows, dim=0)
    expanded_names = [
        f"{condition}|d{digit}"
        for condition in condition_names
        for digit in digits
    ]
    return q_by_layer, expanded_names, condition_names, meta


def resolve_layers(model: Any, start: int, end: int) -> list[int]:
    n_layers = len(model.model.language_model.layers)
    if start < 0 or end >= n_layers or start > end:
        raise ValueError(f"Invalid layer range {start}-{end}; Qwen has layers 0-{n_layers - 1}")
    return list(range(start, end + 1))


def resolve_max_window_width(raw: str, layers_to_use: list[int]) -> int:
    if str(raw).lower() == "all":
        return len(layers_to_use)
    value = int(raw)
    if value < 1:
        raise ValueError("--max-window-width must be positive or 'all'")
    return min(value, len(layers_to_use))


def positions_for_modes(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    modes: list[str],
) -> dict[str, torch.Tensor]:
    final_pos = attention_mask.shape[1] - 1 - torch.flip(attention_mask.long(), dims=[1]).argmax(dim=1)
    positions: dict[str, torch.Tensor] = {}
    for mode in modes:
        if mode.startswith("final_m"):
            offset = int(mode.removeprefix("final_m"))
            positions[mode] = final_pos - offset

    def find_last_token(token_text: str, cutoff_offset: int) -> torch.Tensor:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"No token ids for position token {token_text!r}")
        token_id = int(ids[-1])
        found = []
        for row_idx in range(input_ids.shape[0]):
            cutoff = max(int(final_pos[row_idx].item()) - cutoff_offset, 0)
            row = input_ids[row_idx, :cutoff]
            mask = attention_mask[row_idx, :cutoff].bool()
            matches = torch.nonzero((row == token_id) & mask, as_tuple=False).flatten()
            found.append(int(matches[-1].item()) if len(matches) else max(cutoff - 1, 0))
        return torch.tensor(found, device=input_ids.device, dtype=torch.long)

    if "question_mark" in modes:
        positions["question_mark"] = find_last_token("?", cutoff_offset=0)
    if "suffix_period" in modes:
        positions["suffix_period"] = find_last_token(".", cutoff_offset=0)
    for name, pos in positions.items():
        if torch.any(pos < 0):
            raise ValueError(f"Negative token position for {name}: {pos}")
    return positions


class MultiPositionWorkTracer:
    def __init__(
        self,
        model: Qwen2_5_VLForConditionalGeneration,
        q_by_layer: dict[int, torch.Tensor],
        position_modes: list[str],
    ) -> None:
        self.model = model
        self.q_by_layer = q_by_layer
        self.layer_indices = sorted(q_by_layer)
        self.position_modes = position_modes
        self.original_forwards: dict[int, Any] = {}
        self.positions: dict[str, torch.Tensor] | None = None
        self.cache: dict[str, dict[int, torch.Tensor]] = {}

    def set_positions(self, positions: dict[str, torch.Tensor]) -> None:
        self.positions = {name: pos.detach().long() for name, pos in positions.items()}
        self.cache = {name: {} for name in self.position_modes}

    def __enter__(self) -> "MultiPositionWorkTracer":
        layers = self.model.model.language_model.layers
        for layer_idx in self.layer_indices:
            mlp = layers[layer_idx].mlp
            self.original_forwards[layer_idx] = mlp.forward

            def patched_forward(module, x, layer=layer_idx):
                gate_act = module.act_fn(module.gate_proj(x))
                gated = gate_act * module.up_proj(x)
                if self.positions is None:
                    raise RuntimeError("positions were not set")
                batch_ids = torch.arange(gated.shape[0], device=gated.device)
                q_matrix = self.q_by_layer[layer].to(gated.device).float()
                for mode in self.position_modes:
                    pos = self.positions[mode].to(gated.device)
                    selected = gated[batch_ids, pos].float()
                    self.cache[mode][layer] = (selected @ q_matrix.T).detach().cpu().float()
                return module.down_proj(gated)

            mlp.forward = MethodType(patched_forward, mlp)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        layers = self.model.model.language_model.layers
        for layer_idx, original_forward in self.original_forwards.items():
            layers[layer_idx].mlp.forward = original_forward
        return False


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(scores, method="average")
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "q25": float("nan"), "median": float("nan"), "q75": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
    }


def threshold_stats(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool)
    order = np.argsort(scores)
    candidates = np.unique(scores[order])
    best = {
        "threshold": float(candidates[0]) if len(candidates) else 0.0,
        "balanced_accuracy": -1.0,
        "accuracy": 0.0,
        "tpr_wrong": 0.0,
        "tnr_correct": 0.0,
    }
    for threshold in candidates:
        pred = scores >= threshold
        tp = int(np.logical_and(pred, labels).sum())
        tn = int(np.logical_and(~pred, ~labels).sum())
        fp = int(np.logical_and(pred, ~labels).sum())
        fn = int(np.logical_and(~pred, labels).sum())
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (tpr + tnr)
        if bal > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "balanced_accuracy": float(bal),
                "accuracy": float((tp + tn) / len(labels)),
                "tpr_wrong": float(tpr),
                "tnr_correct": float(tnr),
            }
    return best


def summarize_scores(
    *,
    score_tensor: np.ndarray,
    samples: list[dict[str, Any]],
    position_modes: list[str],
    condition_names: list[str],
    layers_to_use: list[int],
    min_width: int,
    max_width: int,
) -> dict[str, Any]:
    labels = np.asarray([int(not item["baseline_correct"]) for item in samples], dtype=np.int32)
    layer_offsets = {layer: idx for idx, layer in enumerate(layers_to_use)}
    windows = [
        (start, start + width - 1)
        for start in layers_to_use
        for width in range(min_width, max_width + 1)
        if start + width - 1 in layer_offsets
    ]
    global_top: list[dict[str, Any]] = []

    for mode_idx, mode in enumerate(position_modes):
        for cond_idx, condition in enumerate(condition_names):
            layer_scores = score_tensor[:, mode_idx, cond_idx, :]
            cumsum = np.concatenate(
                [np.zeros((layer_scores.shape[0], 1), dtype=np.float32), np.cumsum(layer_scores, axis=1)],
                axis=1,
            )
            candidates: list[tuple[float, int, int]] = []
            for start, end in windows:
                s = layer_offsets[start]
                e = layer_offsets[end]
                scores = cumsum[:, e + 1] - cumsum[:, s]
                auc = rank_auc(labels, scores)
                if auc is None:
                    continue
                candidates.append((float(auc), int(start), int(end)))
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            best_rows: list[dict[str, Any]] = []
            for auc, start, end in candidates[:5]:
                s = layer_offsets[start]
                e = layer_offsets[end]
                scores = cumsum[:, e + 1] - cumsum[:, s]
                correct_scores = scores[labels == 0]
                wrong_scores = scores[labels == 1]
                best_rows.append(
                    {
                        "position_mode": mode,
                        "condition": condition,
                        "window": f"{start}-{end}",
                        "window_start": int(start),
                        "window_end": int(end),
                        "width": int(end - start + 1),
                        "auc_wrong_high": float(auc),
                        "n": int(len(samples)),
                        "n_correct": int((labels == 0).sum()),
                        "n_wrong": int((labels == 1).sum()),
                        "threshold": threshold_stats(labels, scores),
                        "correct_score_stats": quantiles(correct_scores),
                        "wrong_score_stats": quantiles(wrong_scores),
                    }
                )
            best_rows.sort(key=lambda row: (-row["auc_wrong_high"], row["window_start"], row["window_end"]))
            global_top.extend(best_rows[:5])
    global_top.sort(key=lambda row: -row["auc_wrong_high"])

    per_digit_at_best: list[dict[str, Any]] = []
    per_dataset_at_best: list[dict[str, Any]] = []
    for best in global_top[:20]:
        mode_idx = position_modes.index(best["position_mode"])
        cond_idx = condition_names.index(best["condition"])
        s = layer_offsets[int(best["window_start"])]
        e = layer_offsets[int(best["window_end"])]
        scores = score_tensor[:, mode_idx, cond_idx, s : e + 1].sum(axis=1)
        for digit in sorted({int(item["gold"]) for item in samples}):
            mask = np.asarray([int(item["gold"]) == digit for item in samples], dtype=bool)
            auc = rank_auc(labels[mask], scores[mask])
            if auc is not None:
                per_digit_at_best.append(
                    {
                        "parent_position_mode": best["position_mode"],
                        "parent_condition": best["condition"],
                        "parent_window": best["window"],
                        "scope": f"digit_{digit}",
                        "auc_wrong_high": float(auc),
                        "n": int(mask.sum()),
                        "n_correct": int((labels[mask] == 0).sum()),
                        "n_wrong": int((labels[mask] == 1).sum()),
                        "correct_score_stats": quantiles(scores[mask][labels[mask] == 0]),
                        "wrong_score_stats": quantiles(scores[mask][labels[mask] == 1]),
                    }
                )
        for dataset in sorted({item["dataset"] for item in samples}):
            mask = np.asarray([item["dataset"] == dataset for item in samples], dtype=bool)
            auc = rank_auc(labels[mask], scores[mask])
            if auc is not None:
                per_dataset_at_best.append(
                    {
                        "parent_position_mode": best["position_mode"],
                        "parent_condition": best["condition"],
                        "parent_window": best["window"],
                        "scope": dataset,
                        "auc_wrong_high": float(auc),
                        "n": int(mask.sum()),
                        "n_correct": int((labels[mask] == 0).sum()),
                        "n_wrong": int((labels[mask] == 1).sum()),
                        "correct_score_stats": quantiles(scores[mask][labels[mask] == 0]),
                        "wrong_score_stats": quantiles(scores[mask][labels[mask] == 1]),
                    }
                )
    return {
        "global_top_windows": global_top[:100],
        "per_digit_at_top_windows": per_digit_at_best,
        "per_dataset_at_top_windows": per_dataset_at_best,
    }


def prompt_token_debug(processor: Any, inputs: Any, positions: dict[str, torch.Tensor]) -> dict[str, Any]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    row_idx = 0
    valid = torch.nonzero(attention_mask[row_idx].bool(), as_tuple=False).flatten()
    ids = input_ids[row_idx, valid].detach().cpu().tolist()
    decoded_tail = []
    base = int(valid[0].item())
    for absolute_pos, token_id in zip(valid.detach().cpu().tolist()[-80:], ids[-80:]):
        decoded_tail.append(
            {
                "absolute_position": int(absolute_pos),
                "relative_from_valid_start": int(absolute_pos - base),
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            }
        )
    return {
        "tail_tokens": decoded_tail,
        "positions": {
            mode: {
                "absolute_position": int(pos[row_idx].item()),
                "token_id": int(input_ids[row_idx, int(pos[row_idx].item())].item()),
                "token": tokenizer.decode([int(input_ids[row_idx, int(pos[row_idx].item())].item())], skip_special_tokens=False),
            }
            for mode, pos in positions.items()
        },
    }


def maybe_plot(output_dir: Path, score_tensor: np.ndarray, samples: list[dict[str, Any]], summary: dict[str, Any], position_modes: list[str], condition_names: list[str], layers_to_use: list[int]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
        return []
    labels = np.asarray([int(not item["baseline_correct"]) for item in samples], dtype=np.int32)
    layer_offsets = {layer: idx for idx, layer in enumerate(layers_to_use)}
    paths = []
    for row in summary["global_top_windows"][:6]:
        mode_idx = position_modes.index(row["position_mode"])
        cond_idx = condition_names.index(row["condition"])
        s = layer_offsets[int(row["window_start"])]
        e = layer_offsets[int(row["window_end"])]
        scores = score_tensor[:, mode_idx, cond_idx, s : e + 1].sum(axis=1)
        correct = scores[labels == 0]
        wrong = scores[labels == 1]
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.boxplot([correct, wrong], tick_labels=["Correct", "Wrong"], showfliers=False)
        ax.set_ylabel("Drift score")
        ax.set_title(f"{row['position_mode']} {row['window']} AUC={row['auc_wrong_high']:.3f}")
        fig.tight_layout()
        safe_condition = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["condition"])[:120]
        path = output_dir / f"boxplot_{row['position_mode']}_{safe_condition}_{row['window']}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_readme(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen Counting Digit Drift Position Sweep",
        "",
        f"- Samples: {manifest['n_samples']} total, {manifest['n_correct']} correct, {manifest['n_wrong']} wrong.",
        f"- Digits: {manifest['digits']}; {manifest['samples_per_digit_per_dataset']} samples per digit per dataset.",
        "- Direction orientation: `to_digit`; larger score means stronger work toward the Arabic digit side.",
        "- Vectors are LM-head/unembedding rows, matching the existing Qwen neuron-set builders.",
        f"- Layers: {manifest['layer_range'][0]}--{manifest['layer_range'][-1]}; windows {manifest['min_window_width']}--{manifest['max_window_width']}.",
        "",
        "## Best Windows",
        "",
        "| Rank | Position | Condition | Window | AUC | Bal. acc. | Correct mean | Wrong mean |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["global_top_windows"][:30], start=1):
        lines.append(
            "| {rank} | {position} | `{condition}` | {window} | {auc:.4f} | {bal:.4f} | {cmean:.4f} | {wmean:.4f} |".format(
                rank=rank,
                position=row["position_mode"],
                condition=row["condition"],
                window=row["window"],
                auc=row["auc_wrong_high"],
                bal=row["threshold"]["balanced_accuracy"],
                cmean=row["correct_score_stats"]["mean"],
                wmean=row["wrong_score_stats"]["mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if False:
        os.chdir("/tmp")
    args = parse_args()
    output_dir = args.output_root / run_id(args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = parse_csv(args.datasets)
    unknown_datasets = sorted(set(dataset_names) - set(DATASETS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}; known={sorted(DATASETS)}")
    digits = parse_ints(args.digits)
    position_modes = parse_csv(args.position_modes)
    direction_families = parse_csv(args.direction_families)
    norms = parse_csv(args.norms)
    selection_modes = parse_csv(args.selection_modes)

    samples, dataset_cache, selection_counts = choose_samples(
        predictions_root=args.predictions_root,
        prediction_subdir=args.prediction_subdir,
        prediction_condition=args.prediction_condition,
        dataset_names=dataset_names,
        split=args.split,
        digits=digits,
        per_digit_per_dataset=args.samples_per_digit_per_dataset,
    )

    print(f"[load] processor={args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    print(f"[load] model={args.model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    layers_to_use = resolve_layers(model, args.layer_start, args.layer_end)
    max_window_width = resolve_max_window_width(args.max_window_width, layers_to_use)
    unembed = get_lm_head_weight(model).detach().float()
    direction_bank, direction_metadata = build_direction_bank(
        processor.tokenizer,
        unembed,
        digits,
        direction_families,
        norms,
    )
    fixed_sets, fixed_set_metadata = load_fixed_sets(selection_modes)
    q_by_layer, expanded_condition_names, compact_condition_names, q_metadata = build_q_matrices(
        model=model,
        direction_bank=direction_bank,
        digits=digits,
        layers_to_use=layers_to_use,
        selection_modes=selection_modes,
        fixed_sets=fixed_sets,
    )
    expanded_condition_index = {name: idx for idx, name in enumerate(expanded_condition_names)}
    compact_condition_index = {name: idx for idx, name in enumerate(compact_condition_names)}

    n = len(samples)
    score_tensor = np.zeros((n, len(position_modes), len(compact_condition_names), len(layers_to_use)), dtype=np.float32)
    prompt_debug: dict[str, Any] | None = None
    start_time = time.time()
    layer_to_tensor_offset = {layer: i for i, layer in enumerate(layers_to_use)}

    with MultiPositionWorkTracer(model, q_by_layer, position_modes) as tracer:
        for batch_start, batch in batched(samples, args.batch_size):
            rows = [dataset_cache[item["dataset"]][item["source_index"]] for item in batch]
            inputs = prepare_inputs(processor, rows, args.suffix).to(model.device)
            positions = positions_for_modes(inputs["input_ids"], inputs["attention_mask"], processor.tokenizer, position_modes)
            tracer.set_positions(positions)
            if prompt_debug is None:
                prompt_debug = prompt_token_debug(processor, inputs, positions)
            with torch.inference_mode():
                _ = model(**inputs, use_cache=False, logits_to_keep=1)
            for local_idx, sample in enumerate(batch):
                digit = int(sample["gold"])
                for mode_idx, mode in enumerate(position_modes):
                    for layer in layers_to_use:
                        layer_offset = layer_to_tensor_offset[layer]
                        values = tracer.cache[mode][layer][local_idx]
                        for base_condition in [
                            f"{direction_key}|{selection_mode}"
                            for direction_key in sorted(direction_bank)
                            for selection_mode in selection_modes
                        ]:
                            expanded = f"{base_condition}|d{digit}"
                            score_tensor[batch_start + local_idx, mode_idx, compact_condition_index[base_condition], layer_offset] = float(
                                values[expanded_condition_index[expanded]].item()
                            )
            seen = min(batch_start + len(batch), n)
            print(f"[score] seen={seen}/{n} elapsed={time.time() - start_time:.1f}s", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("[summarize] sweeping windows", flush=True)
    summary = summarize_scores(
        score_tensor=score_tensor,
        samples=samples,
        position_modes=position_modes,
        condition_names=compact_condition_names,
        layers_to_use=layers_to_use,
        min_width=args.min_window_width,
        max_width=max_window_width,
    )
    summary_path = output_dir / "auc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    score_tensor_path = output_dir / "score_tensor.npz"
    if args.save_score_tensor:
        np.savez_compressed(
            score_tensor_path,
            score_tensor=score_tensor,
            labels=np.asarray([int(not item["baseline_correct"]) for item in samples], dtype=np.int32),
            position_modes=np.asarray(position_modes),
            condition_names=np.asarray(compact_condition_names),
            expanded_q_condition_names=np.asarray(expanded_condition_names),
            layers=np.asarray(layers_to_use, dtype=np.int32),
        )

    plot_paths = maybe_plot(output_dir, score_tensor, samples, summary, position_modes, compact_condition_names, layers_to_use) if args.make_plot else []
    manifest = {
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "datasets": dataset_names,
        "dataset_paths": {name: DATASETS[name] for name in dataset_names},
        "predictions_root": str(args.predictions_root),
        "prediction_subdir": args.prediction_subdir,
        "prediction_condition": args.prediction_condition,
        "split": args.split,
        "suffix": args.suffix,
        "digits": digits,
        "samples_per_digit_per_dataset": args.samples_per_digit_per_dataset,
        "selection_counts": selection_counts,
        "n_samples": len(samples),
        "n_correct": sum(1 for sample in samples if sample["baseline_correct"]),
        "n_wrong": sum(1 for sample in samples if not sample["baseline_correct"]),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "layer_range": layers_to_use,
        "min_window_width": args.min_window_width,
        "max_window_width": max_window_width,
        "position_modes": position_modes,
        "direction_families": direction_families,
        "norms": norms,
        "selection_modes": selection_modes,
        "condition_names": compact_condition_names,
        "expanded_q_condition_names": expanded_condition_names,
        "direction_definition": "to_digit direction d_k = U(str(k)) - U(word_family(k)); larger score is digit-side drift.",
        "vector_space": "lm_head/unembedding rows",
        "direction_metadata": direction_metadata,
        "fixed_set_metadata": fixed_set_metadata,
        "q_metadata": q_metadata,
        "prompt_debug_first_sample": prompt_debug,
        "samples_path": str(samples_path),
        "summary_path": str(summary_path),
        "score_tensor_path": str(score_tensor_path) if args.save_score_tensor else None,
        "plot_paths": plot_paths,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", manifest, summary)

    print(f"[done] {output_dir}", flush=True)
    for rank, row in enumerate(summary["global_top_windows"][:12], start=1):
        print(
            "[best {rank}] pos={pos} cond={cond} window={window} auc={auc:.4f} bal_acc={bal:.4f}".format(
                rank=rank,
                pos=row["position_mode"],
                cond=row["condition"],
                window=row["window"],
                auc=row["auc_wrong_high"],
                bal=row["threshold"]["balanced_accuracy"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
