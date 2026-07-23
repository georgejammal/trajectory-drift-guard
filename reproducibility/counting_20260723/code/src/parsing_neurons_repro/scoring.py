from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .io import window_layers, write_json
from .models import ModelSpec, load_tensor, weight_map, weight_templates


def cosine_scores(weight: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Weight is [d_model, channels]; each column is an output vector.
    w = weight.float()
    d = direction.float()
    norms = w.norm(dim=0).clamp_min(1e-12)
    dots = torch.matmul(d, w)
    return dots / norms, dots


def select_columns(
    *,
    scores: torch.Tensor,
    dots: torch.Tensor,
    norms: torch.Tensor,
    threshold: float,
    layer: int,
    index_name: str,
) -> list[dict[str, Any]]:
    selected = torch.nonzero(scores >= threshold, as_tuple=False).flatten().tolist()
    rows = []
    for idx in selected:
        rows.append(
            {
                "layer": int(layer),
                index_name: int(idx),
                "score": float(scores[idx].item()),
                "cosine_similarity": float(scores[idx].item()),
                "dot_product": float(dots[idx].item()),
                "weight_norm": float(norms[idx].item()),
            }
        )
    return rows


def score_static_components(
    *,
    model: ModelSpec,
    direction: torch.Tensor,
    direction_metadata: dict[str, Any],
    window: str,
    sigma: float,
    component: str,
    output_path: Path,
) -> dict[str, Any]:
    if component not in {"mlp", "attn"}:
        raise ValueError(f"Unsupported static component: {component}")

    index = weight_map(model.path)
    templates = weight_templates(index)
    key_template = templates["mlp_down"] if component == "mlp" else templates["attn_o"]
    index_name = "neuron" if component == "mlp" else "channel"
    threshold = float(sigma) / math.sqrt(model.hidden_size)
    rows: list[dict[str, Any]] = []
    per_layer: dict[str, int] = {}

    for layer in window_layers(window):
        key = key_template.format(layer=layer)
        weight = load_tensor(model.path, index, key)
        if weight.ndim != 2 or weight.shape[0] != model.hidden_size:
            raise ValueError(f"Expected [{model.hidden_size}, channels] for {key}, got {tuple(weight.shape)}")
        scores, dots = cosine_scores(weight, direction)
        norms = weight.float().norm(dim=0).clamp_min(1e-12)
        layer_rows = select_columns(
            scores=scores,
            dots=dots,
            norms=norms,
            threshold=threshold,
            layer=layer,
            index_name=index_name,
        )
        per_layer[str(layer)] = len(layer_rows)
        rows.extend(layer_rows)
        del weight, scores, dots, norms

    rows.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    payload = {
        "model_alias": model.alias,
        "model_path": str(model.path),
        "component": component,
        "write_vector_key_template": key_template,
        "direction_metadata": direction_metadata,
        "window": window,
        "layers": window_layers(window),
        "sigma": float(sigma),
        "null_cosine_sd": 1.0 / math.sqrt(model.hidden_size),
        "cosine_threshold": threshold,
        "selection_rule": "cosine_similarity >= sigma / sqrt(d_model)",
        "selected_count": len(rows),
        "layer_counts": per_layer,
        "rows": rows,
    }
    write_json(output_path, payload)
    return payload

