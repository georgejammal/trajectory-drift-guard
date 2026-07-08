#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, features
from scipy.stats import rankdata
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("qwen-vl-utils is required for Qwen2.5-VL.") from exc


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = os.environ.get(
    "QWEN25VL_3B_INSTRUCT_PATH",
    "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/"
    "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
)
FLORES_ARABIC_PAIRS = Path(
    ARTIFACT_ROOT
    / "data"
    / "flores_transfer_pairs"
    / "flores101_en_to_cc_ocr_languages_random500"
    / "pairs_by_language"
    / "en_to_arabic.json"
)
OUTPUT_ROOT = ARTIFACT_ROOT / "outputs" / "experiment_runs" / "multilingual_ocr"
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DEFAULT_PROMPT = "Transcribe exactly the Arabic word in the image. Output only the word."

NEURON_SET_PATHS = {
    "real_gt3sd_layers18_32": Path(
        ARTIFACT_ROOT
        / "resources"
        / "neuron_sets"
        / "multilingual_ocr"
        / "qwen25vl_ccocr_mean_unembed_real3_null1_layers18_32_20260621_candidates"
        / "arabic"
        / "layers18_32"
        / "arabic_mean_unembed_real_gt3sd_layers18_32.csv"
    ),
    "real_gt3sd_null_lte1sd_layers18_32": Path(
        ARTIFACT_ROOT
        / "resources"
        / "neuron_sets"
        / "multilingual_ocr"
        / "qwen25vl_ccocr_mean_unembed_real3_null1_layers18_32_20260621_candidates"
        / "arabic"
        / "layers18_32"
        / "arabic_mean_unembed_real_gt3sd_null_lte1sd_layers18_32.csv"
    ),
    "real_gt3sd_null_lte3sd_layers18_32": Path(
        ARTIFACT_ROOT
        / "resources"
        / "neuron_sets"
        / "multilingual_ocr"
        / "qwen25vl_ccocr_mean_unembed_real3_null1_layers18_32_20260621_candidates"
        / "arabic"
        / "layers18_32"
        / "arabic_mean_unembed_real_gt3sd_null_lte1sd_layers18_32.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-token Arabic OCR drift probe using fixed FLORES Arabic-minus-English direction."
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--flores-pairs-json", type=Path, default=FLORES_ARABIC_PAIRS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-words", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--font-size", type=int, default=142)
    parser.add_argument("--font-path", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--num-direction-pairs", type=int, default=250)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=35)
    parser.add_argument("--min-window-width", type=int, default=1)
    parser.add_argument("--max-window-width", default="all")
    parser.add_argument(
        "--position-modes",
        default=(
            "final_m0,final_m1,final_m2,final_m3,final_m4,final_m5,"
            "final_m6,final_m7,final_m8,final_m9,final_m10,question_mark,suffix_period"
        ),
    )
    parser.add_argument(
        "--neuron-sets",
        default="real_gt3sd_layers18_32,real_gt3sd_null_lte1sd_layers18_32,real_gt3sd_null_lte3sd_layers18_32",
    )
    parser.add_argument("--make-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_id(raw: str | None) -> str:
    return raw or "qwen_arabic_onetoken_ocr_drift_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def arabic_only(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return "".join(re.findall(r"[\u0621-\u064A]+", text))


def arabic_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.findall(r"[\u0621-\u064A]+", text)


def load_flores_pairs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    rows = payload["pairs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No FLORES pairs found in {path}")
    return payload, rows


def choose_one_token_words(
    *,
    tokenizer: Any,
    flores_rows: list[dict[str, Any]],
    num_words: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: Counter[str] = Counter()
    first_seen: dict[str, dict[str, Any]] = {}
    for row in flores_rows:
        for word in arabic_words(str(row["target"])):
            if len(word) < min_chars or len(word) > max_chars:
                continue
            if not re.fullmatch(r"[\u0621-\u064A]+", word):
                continue
            counts[word] += 1
            first_seen.setdefault(word, row)

    chosen = []
    rejected_multi = 0
    for word, count in counts.most_common():
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) != 1:
            rejected_multi += 1
            continue
        row = first_seen[word]
        chosen.append(
            {
                "sample_id": f"arabic_onetoken_{len(chosen):04d}",
                "word": word,
                "token_id": int(token_ids[0]),
                "token_text": tokenizer.decode(token_ids, skip_special_tokens=False),
                "frequency_in_flores_subset": int(count),
                "source_flores_id": row.get("flores_id"),
                "source_english": row.get("english"),
                "source_arabic_sentence": row.get("target"),
            }
        )
        if len(chosen) >= num_words:
            break
    if len(chosen) < 2:
        raise RuntimeError(f"Only found {len(chosen)} one-token Arabic words.")
    return chosen, {
        "candidate_unique_words": len(counts),
        "rejected_multi_token_before_stop": rejected_multi,
        "selected_words": len(chosen),
        "min_chars": min_chars,
        "max_chars": max_chars,
    }


def fit_font(word: str, font_path: Path, width: int, height: int, start_size: int) -> ImageFont.FreeTypeFont:
    scratch = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(scratch)
    for size in range(start_size, 20, -2):
        font = ImageFont.truetype(str(font_path), size=size)
        bbox = draw.textbbox((0, 0), word, font=font, direction="rtl", language="ar")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= width * 0.78 and text_height <= height * 0.70:
            return font
    return ImageFont.truetype(str(font_path), size=24)


def render_word_image(row: dict[str, Any], image_path: Path, font_path: Path, width: int, height: int, font_size: int) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = fit_font(row["word"], font_path, width, height, font_size)
    bbox = draw.textbbox((0, 0), row["word"], font=font, direction="rtl", language="ar")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) / 2 - bbox[0]
    y = (height - text_height) / 2 - bbox[1]
    draw.text((x, y), row["word"], font=font, fill="black", direction="rtl", language="ar")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def render_dataset(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not features.check("raqm"):
        raise RuntimeError("PIL was built without libraqm; Arabic rendering may be incorrect.")
    rendered = []
    for row in rows:
        image_path = output_dir / "images" / f"{row['sample_id']}.png"
        render_word_image(row, image_path, args.font_path, args.width, args.height, args.font_size)
        rendered.append(
            {
                **row,
                "image_path": str(image_path.resolve()),
                "width": args.width,
                "height": args.height,
                "font_path": str(args.font_path),
                "background": "white",
                "foreground": "black",
                "noise": "none",
            }
        )
    return rendered


def build_qwen_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path.resolve()}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(processor: Any, batch: list[dict[str, Any]], prompt: str) -> Any:
    messages = [build_qwen_messages(Path(row["image_path"]), prompt) for row in batch]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def decode_generated(generated: torch.Tensor, inputs: Any, processor: Any) -> list[str]:
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run_qwen_ocr(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    predictions = []
    for _, batch in batched(samples, args.batch_size):
        inputs = prepare_inputs(processor, batch, args.prompt).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        decoded = decode_generated(generated, inputs, processor)
        for sample, prediction in zip(batch, decoded):
            pred_arabic = arabic_only(prediction)
            gold = arabic_only(sample["word"])
            predictions.append(
                {
                    **sample,
                    "gold": gold,
                    "prediction": prediction.strip(),
                    "prediction_arabic_only": pred_arabic,
                    "correct": pred_arabic == gold,
                    "contains_gold": gold in pred_arabic,
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def get_output_embedding_weight(model: Any) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight.detach().float()
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().float()
    raise AttributeError("Could not find output embedding / lm_head weight.")


def mean_unembedding_for_text(tokenizer: Any, unembed: torch.Tensor, text: str) -> torch.Tensor:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if callable(tokenizer) else tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Text tokenized to an empty sequence: {text!r}")
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=unembed.device)
    return unembed[token_tensor].float().mean(dim=0).detach().cpu()


def build_flores_direction(
    *,
    tokenizer: Any,
    unembed: torch.Tensor,
    rows: list[dict[str, Any]],
    num_pairs: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected = rows[:num_pairs]
    components = []
    source_lengths = []
    target_lengths = []
    for row in selected:
        source_ids = tokenizer(str(row["english"]), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(str(row["target"]), add_special_tokens=False)["input_ids"]
        source_lengths.append(len(source_ids))
        target_lengths.append(len(target_ids))
        source_vec = unembed[torch.tensor(source_ids, dtype=torch.long, device=unembed.device)].float().mean(dim=0).detach().cpu()
        target_vec = unembed[torch.tensor(target_ids, dtype=torch.long, device=unembed.device)].float().mean(dim=0).detach().cpu()
        components.append(target_vec - source_vec)
    raw = torch.stack(components, dim=0).mean(dim=0)
    unit = F.normalize(raw, dim=0)
    return {"norm0_raw": raw, "norm1_unit": unit}, {
        "definition": "mean_i(mean_unembed(arabic_sentence_i) - mean_unembed(english_sentence_i))",
        "num_direction_pairs": num_pairs,
        "raw_direction_norm": float(raw.norm().item()),
        "unit_direction_norm": float(unit.norm().item()),
        "source_token_length_mean": float(np.mean(source_lengths)),
        "target_token_length_mean": float(np.mean(target_lengths)),
    }


def load_neuron_sets(names: list[str]) -> tuple[dict[str, dict[int, set[int]]], dict[str, Any]]:
    sets: dict[str, dict[int, set[int]]] = {}
    meta: dict[str, Any] = {}
    for name in names:
        path = NEURON_SET_PATHS.get(name)
        if path is None:
            raise ValueError(f"Unknown neuron set {name}; known={sorted(NEURON_SET_PATHS)}")
        by_layer: dict[int, set[int]] = {}
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                layer = int(row["layer"])
                neuron = int(row["neuron_index"])
                by_layer.setdefault(layer, set()).add(neuron)
                rows.append(row)
        sets[name] = by_layer
        meta[name] = {
            "path": str(path),
            "num_neurons": sum(len(v) for v in by_layer.values()),
            "layer_counts": {str(layer): len(vals) for layer, vals in sorted(by_layer.items())},
            "manifest_path": str(path) + ".manifest.json",
        }
    return sets, meta


def resolve_layers(model: Any, start: int, end: int) -> list[int]:
    n_layers = len(model.model.language_model.layers)
    if start < 0 or end >= n_layers or start > end:
        raise ValueError(f"Invalid layer range {start}-{end}; model has layers 0-{n_layers - 1}")
    return list(range(start, end + 1))


def resolve_max_window_width(raw: str, layers: list[int]) -> int:
    if str(raw).lower() == "all":
        return len(layers)
    value = int(raw)
    if value < 1:
        raise ValueError("--max-window-width must be positive or all")
    return min(value, len(layers))


def build_q_matrices(
    *,
    model: Any,
    directions: dict[str, torch.Tensor],
    neuron_sets: dict[str, dict[int, set[int]]],
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], list[str]]:
    q_by_layer: dict[int, torch.Tensor] = {}
    condition_names = [f"{direction_name}|{set_name}" for direction_name in directions for set_name in neuron_sets]
    decoder_layers = model.model.language_model.layers
    device = next(model.parameters()).device
    for layer_idx in layers:
        weight = decoder_layers[layer_idx].mlp.down_proj.weight.detach().float()
        rows = []
        for direction_name in directions:
            direction = directions[direction_name].to(device).float()
            q = (weight.T @ direction).detach().float()
            for set_name, by_layer in neuron_sets.items():
                mask = torch.zeros_like(q, dtype=torch.bool)
                selected = sorted(by_layer.get(layer_idx, set()))
                if selected:
                    mask[torch.tensor(selected, device=mask.device, dtype=torch.long)] = True
                rows.append(torch.where(mask, q, torch.zeros_like(q)).detach().cpu().float())
        q_by_layer[layer_idx] = torch.stack(rows, dim=0)
    return q_by_layer, condition_names


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
            positions[mode] = final_pos - int(mode.removeprefix("final_m"))

    def find_last(token_text: str) -> torch.Tensor:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"No token ids for token {token_text!r}")
        token_id = int(token_ids[-1])
        out = []
        for row_idx in range(input_ids.shape[0]):
            cutoff = int(final_pos[row_idx].item()) + 1
            row = input_ids[row_idx, :cutoff]
            mask = attention_mask[row_idx, :cutoff].bool()
            matches = torch.nonzero((row == token_id) & mask, as_tuple=False).flatten()
            out.append(int(matches[-1].item()) if len(matches) else max(cutoff - 1, 0))
        return torch.tensor(out, device=input_ids.device, dtype=torch.long)

    if "question_mark" in modes:
        positions["question_mark"] = find_last("?")
    if "suffix_period" in modes:
        positions["suffix_period"] = find_last(".")
    for mode, pos in positions.items():
        if torch.any(pos < 0):
            raise ValueError(f"Negative token position for {mode}: {pos}")
    return positions


class WorkTracer:
    def __init__(self, model: Any, q_by_layer: dict[int, torch.Tensor], position_modes: list[str]) -> None:
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

    def __enter__(self) -> "WorkTracer":
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


def prompt_token_debug(processor: Any, inputs: Any, positions: dict[str, torch.Tensor]) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    input_ids = inputs["input_ids"]
    row_idx = 0
    return {
        mode: {
            "absolute_position": int(pos[row_idx].item()),
            "token_id": int(input_ids[row_idx, int(pos[row_idx].item())].item()),
            "token": tokenizer.decode([int(input_ids[row_idx, int(pos[row_idx].item())].item())], skip_special_tokens=False),
        }
        for mode, pos in positions.items()
    }


def score_prefill(
    *,
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    q_by_layer: dict[int, torch.Tensor],
    condition_names: list[str],
    layers: list[int],
    position_modes: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = np.zeros((len(samples), len(position_modes), len(condition_names), len(layers)), dtype=np.float32)
    layer_offsets = {layer: idx for idx, layer in enumerate(layers)}
    prompt_debug = None
    with WorkTracer(model, q_by_layer, position_modes) as tracer:
        for batch_start, batch in batched(samples, args.batch_size):
            inputs = prepare_inputs(processor, batch, args.prompt).to(model.device)
            positions = positions_for_modes(inputs["input_ids"], inputs["attention_mask"], processor.tokenizer, position_modes)
            tracer.set_positions(positions)
            if prompt_debug is None:
                prompt_debug = prompt_token_debug(processor, inputs, positions)
            with torch.inference_mode():
                _ = model(**inputs, use_cache=False, logits_to_keep=1)
            for local_idx in range(len(batch)):
                for mode_idx, mode in enumerate(position_modes):
                    for layer in layers:
                        tensor[batch_start + local_idx, mode_idx, :, layer_offsets[layer]] = tracer.cache[mode][layer][local_idx].numpy()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return tensor, {"first_sample_positions": prompt_debug}


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
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
    best = {"threshold": 0.0, "balanced_accuracy": -1.0, "accuracy": 0.0, "tpr_failed": 0.0, "tnr_success": 0.0}
    for threshold in np.unique(scores):
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
                "tpr_failed": float(tpr),
                "tnr_success": float(tnr),
            }
    return best


def summarize(
    *,
    score_tensor: np.ndarray,
    samples: list[dict[str, Any]],
    position_modes: list[str],
    condition_names: list[str],
    layers: list[int],
    min_width: int,
    max_width: int,
) -> dict[str, Any]:
    labels = np.asarray([int(not sample["correct"]) for sample in samples], dtype=np.int32)
    layer_offsets = {layer: idx for idx, layer in enumerate(layers)}
    windows = [
        (start, start + width - 1)
        for start in layers
        for width in range(min_width, max_width + 1)
        if start + width - 1 in layer_offsets
    ]
    rows = []
    for mode_idx, mode in enumerate(position_modes):
        for cond_idx, condition in enumerate(condition_names):
            layer_scores = score_tensor[:, mode_idx, cond_idx, :]
            cumsum = np.concatenate([np.zeros((layer_scores.shape[0], 1), dtype=np.float32), np.cumsum(layer_scores, axis=1)], axis=1)
            for orientation_name, sign in [("arabic_work_P", 1.0), ("drift_D_minus_P", -1.0)]:
                candidates: list[tuple[float, int, int]] = []
                for start, end in windows:
                    s = layer_offsets[start]
                    e = layer_offsets[end]
                    scores = sign * (cumsum[:, e + 1] - cumsum[:, s])
                    auc = rank_auc(labels, scores)
                    if auc is not None:
                        candidates.append((float(auc), int(start), int(end)))
                candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                for auc, start, end in candidates[:5]:
                    s = layer_offsets[start]
                    e = layer_offsets[end]
                    scores = sign * (cumsum[:, e + 1] - cumsum[:, s])
                    success_scores = scores[labels == 0]
                    failed_scores = scores[labels == 1]
                    rows.append(
                        {
                            "position_mode": mode,
                            "condition": condition,
                            "orientation": orientation_name,
                            "window": f"{start}-{end}",
                            "window_start": start,
                            "window_end": end,
                            "auc_failed_high": float(auc),
                            "n": int(len(samples)),
                            "n_success": int((labels == 0).sum()),
                            "n_failed": int((labels == 1).sum()),
                            "threshold": threshold_stats(labels, scores),
                            "success_score_stats": quantiles(success_scores),
                            "failed_score_stats": quantiles(failed_scores),
                        }
                    )
    rows.sort(key=lambda row: -row["auc_failed_high"])
    return {"global_top_windows": rows[:100]}


def maybe_plot(output_dir: Path, score_tensor: np.ndarray, samples: list[dict[str, Any]], summary: dict[str, Any], position_modes: list[str], condition_names: list[str], layers: list[int]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
        return []

    labels = np.asarray([int(not sample["correct"]) for sample in samples], dtype=np.int32)
    layer_offsets = {layer: idx for idx, layer in enumerate(layers)}
    paths = []
    for row in summary["global_top_windows"][:6]:
        mode_idx = position_modes.index(row["position_mode"])
        cond_idx = condition_names.index(row["condition"])
        s = layer_offsets[int(row["window_start"])]
        e = layer_offsets[int(row["window_end"])]
        sign = -1.0 if row["orientation"] == "drift_D_minus_P" else 1.0
        scores = sign * score_tensor[:, mode_idx, cond_idx, s : e + 1].sum(axis=1)
        success = scores[labels == 0]
        failed = scores[labels == 1]
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.boxplot([success, failed], tick_labels=["Success", "Failed"], showfliers=False)
        ax.set_ylabel("score")
        ax.set_title(f"{row['position_mode']} {row['window']} AUC={row['auc_failed_high']:.3f}")
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{row['position_mode']}_{row['condition']}_{row['orientation']}_{row['window']}")[:150]
        path = output_dir / f"boxplot_{safe}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_readme(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen Arabic One-Token OCR Drift",
        "",
        f"- Samples: {manifest['num_samples']} one-token Arabic words rendered on white background.",
        f"- OCR success: {manifest['num_success']} success, {manifest['num_failed']} failed.",
        "- Direction: fixed FLORES Arabic-minus-English sentence direction.",
        "- `norm0_raw`: raw FLORES direction; `norm1_unit`: unit-normalized FLORES direction.",
        "- Positive signed work `P` means MLP work along Arabic-minus-English; drift score is `D=-P`.",
        "",
        "## Best Windows",
        "",
        "| Rank | Position | Condition | Orientation | Window | AUC | Bal. acc. | Success mean | Failed mean |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["global_top_windows"][:30], start=1):
        lines.append(
            "| {rank} | {pos} | `{cond}` | {ori} | {window} | {auc:.4f} | {bal:.4f} | {smean:.4f} | {fmean:.4f} |".format(
                rank=rank,
                pos=row["position_mode"],
                cond=row["condition"],
                ori=row["orientation"],
                window=row["window"],
                auc=row["auc_failed_high"],
                bal=row["threshold"]["balanced_accuracy"],
                smean=row["success_score_stats"]["mean"],
                fmean=row["failed_score_stats"]["mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if Path.cwd() == Path("/home/georgejammal/projects"):
        os.chdir("/tmp")
    args = parse_args()
    output_dir = args.output_root / run_id(args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] processor={args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    _, flores_rows = load_flores_pairs(args.flores_pairs_json)
    word_rows, word_selection_meta = choose_one_token_words(
        tokenizer=processor.tokenizer,
        flores_rows=flores_rows,
        num_words=args.num_words,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    samples = render_dataset(word_rows, output_dir, args)
    write_jsonl(output_dir / "rendered_manifest.jsonl", samples)

    print(f"[load] model={args.model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    print("[ocr] generating baseline OCR", flush=True)
    predictions = run_qwen_ocr(model=model, processor=processor, samples=samples, args=args)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    unembed = get_output_embedding_weight(model)
    directions, direction_meta = build_flores_direction(
        tokenizer=processor.tokenizer,
        unembed=unembed,
        rows=flores_rows,
        num_pairs=args.num_direction_pairs,
    )
    neuron_set_names = parse_csv(args.neuron_sets)
    neuron_sets, neuron_set_meta = load_neuron_sets(neuron_set_names)
    layers = resolve_layers(model, args.layer_start, args.layer_end)
    max_window_width = resolve_max_window_width(args.max_window_width, layers)
    position_modes = parse_csv(args.position_modes)
    q_by_layer, condition_names = build_q_matrices(
        model=model,
        directions=directions,
        neuron_sets=neuron_sets,
        layers=layers,
    )

    print("[score] prefill signed work", flush=True)
    started = time.time()
    score_tensor, token_debug = score_prefill(
        model=model,
        processor=processor,
        samples=predictions,
        q_by_layer=q_by_layer,
        condition_names=condition_names,
        layers=layers,
        position_modes=position_modes,
        args=args,
    )
    np.savez_compressed(
        output_dir / "score_tensor.npz",
        score_tensor=score_tensor,
        labels_failed=np.asarray([int(not row["correct"]) for row in predictions], dtype=np.int32),
        position_modes=np.asarray(position_modes),
        condition_names=np.asarray(condition_names),
        layers=np.asarray(layers, dtype=np.int32),
    )

    summary = summarize(
        score_tensor=score_tensor,
        samples=predictions,
        position_modes=position_modes,
        condition_names=condition_names,
        layers=layers,
        min_width=args.min_window_width,
        max_width=max_window_width,
    )
    write_json(output_dir / "auc_summary.json", summary)
    plot_paths = maybe_plot(output_dir, score_tensor, predictions, summary, position_modes, condition_names, layers) if args.make_plot else []

    manifest = {
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "flores_pairs_json": str(args.flores_pairs_json),
        "word_selection": word_selection_meta,
        "num_samples": len(predictions),
        "num_success": sum(1 for row in predictions if row["correct"]),
        "num_failed": sum(1 for row in predictions if not row["correct"]),
        "prompt": args.prompt,
        "rendering": {
            "width": args.width,
            "height": args.height,
            "font_path": str(args.font_path),
            "background": "white",
            "noise": "none",
        },
        "direction": direction_meta,
        "neuron_sets": neuron_set_meta,
        "layer_range": layers,
        "max_window_width": max_window_width,
        "position_modes": position_modes,
        "condition_names": condition_names,
        "token_debug": token_debug,
        "plot_paths": plot_paths,
        "elapsed_score_seconds": time.time() - started,
        "outputs": {
            "rendered_manifest": str(output_dir / "rendered_manifest.jsonl"),
            "predictions": str(output_dir / "predictions.jsonl"),
            "score_tensor": str(output_dir / "score_tensor.npz"),
            "auc_summary": str(output_dir / "auc_summary.json"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_readme(output_dir / "README.md", manifest, summary)

    print(f"[done] {output_dir}", flush=True)
    for rank, row in enumerate(summary["global_top_windows"][:12], start=1):
        print(
            "[best {rank}] pos={pos} cond={cond} orientation={ori} window={window} auc={auc:.4f} bal={bal:.4f}".format(
                rank=rank,
                pos=row["position_mode"],
                cond=row["condition"],
                ori=row["orientation"],
                window=row["window"],
                auc=row["auc_failed_high"],
                bal=row["threshold"]["balanced_accuracy"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
