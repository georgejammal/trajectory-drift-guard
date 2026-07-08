#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path("/home/georgejammal/projects")
DRIFTING_SECURITY = PROJECT_ROOT / "drifting" / "security"
ALIGNTREE_DATA = DRIFTING_SECURITY / "datasets" / "aligntree_processed"
OUTPUT_ROOT = PROJECT_ROOT / "try_and_change" / "security_drift" / "outputs"
SAMPLE_ROOT = PROJECT_ROOT / "try_and_change" / "security_drift" / "sample_sets"

GEMMA3_PATH = (
    "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/"
    "snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
)
LLAMA32_3B_INSTRUCT_PATH = (
    "/home/georgejammal/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/"
    "snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
)

MODEL_DEFAULTS = {
    "gemma3_4b_it": {
        "model_path": GEMMA3_PATH,
        "neuron_json": DRIFTING_SECURITY
        / "neuron_sets"
        / "gemma3_4b_it_refusal_minus_answerability_single_token_top1000_layers17_33.json",
        "layers": "17-33",
    },
    "llama3p2_3b_instruct": {
        "model_path": LLAMA32_3B_INSTRUCT_PATH,
        "neuron_json": DRIFTING_SECURITY
        / "neuron_sets"
        / "llama3p2_3b_instruct_refusal_minus_answerability_single_token_top1000_layers14_27.json",
        "layers": "14-27",
    },
}

DATASET_MAP = {
    "malwaregen": "malwaregen",
    "promptinject": "promptinject",
    "pair": "pair_test",
    "autodan": "autodan_test",
}

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def dtype_from_name(name: str):
    import torch

    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def model_defaults(model_alias: str) -> dict[str, Any]:
    if model_alias not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model alias: {model_alias}")
    return MODEL_DEFAULTS[model_alias]


def format_prompt(model_alias: str, instruction: str) -> str:
    if model_alias == "gemma3_4b_it":
        return GEMMA_CHAT_TEMPLATE.format(instruction=instruction)
    if model_alias == "llama3p2_3b_instruct":
        return LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)
    raise ValueError(f"Unknown model alias: {model_alias}")


def sample_id(dataset_name: str, index: int, instruction: str) -> str:
    return f"{dataset_name}__{index:05d}__{sha256_text(instruction)[:12]}"


def get_layers(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model.layers
    if hasattr(model, "language_model"):
        language_model = model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            return language_model.model.layers
    raise AttributeError("Could not find decoder layers.")


def load_neurons(
    path: Path,
    layers: tuple[int, ...],
    *,
    score_key: str = "cosine_similarity",
    positive_only: bool = True,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("top_neurons")
    if not isinstance(rows, list):
        raise ValueError(f"Expected top_neurons list in {path}")
    layer_set = set(layers)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rank, row in enumerate(rows, start=1):
        layer = int(row["layer"])
        if layer not in layer_set:
            continue
        score = float(row.get(score_key, row.get("cosine_similarity", 1.0)))
        if positive_only and score <= 0:
            continue
        by_layer[layer].append(
            {
                "neuron": int(row["neuron"]),
                "rank": int(row.get("rank", rank)),
                "score": score,
                "cosine_similarity": float(row.get("cosine_similarity", score)),
                "dot_product": float(row.get("dot_product", 0.0)),
            }
        )
    metadata = {
        "path": str(path),
        "direction_name": payload.get("direction_name"),
        "score_definition": payload.get("score_definition"),
        "write_vector_definition": payload.get("write_vector_definition"),
        "score_key": score_key,
        "positive_only": positive_only,
        "layers": list(layers),
        "loaded_neurons": sum(len(v) for v in by_layer.values()),
        "layer_counts": {str(layer): len(by_layer.get(layer, [])) for layer in layers},
    }
    return dict(sorted(by_layer.items())), metadata


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(sorted(counts.items()))
