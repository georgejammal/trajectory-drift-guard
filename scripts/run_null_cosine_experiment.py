#!/usr/bin/env python3
"""Empirically check the null distribution of static MLP-neuron alignments.

The paper uses the approximation

    cos(d, v_{l,n}) ~ N(0, 1 / d_model)

under the null that the task direction d is independent of the MLP output
weight vector v_{l,n}. This script validates that approximation with two
random-direction sources:

1. isotropic Gaussian unit vectors;
2. random token-embedding difference directions.

Only model weights are read. No generation or activation tracing is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parsing_neurons_repro.models import (  # noqa: E402
    MODEL_PATHS,
    load_tensor,
    model_spec,
    weight_map,
    weight_templates,
)


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen2_5_vl_3b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}


@dataclass
class StreamingStats:
    count: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_x3: float = 0.0
    sum_x4: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        x = values.double().reshape(-1)
        self.count += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_x2 += float((x * x).sum().item())
        self.sum_x3 += float((x * x * x).sum().item())
        self.sum_x4 += float((x * x * x * x).sum().item())

    def finalize(self) -> dict[str, float | int]:
        n = max(self.count, 1)
        mean = self.sum_x / n
        ex2 = self.sum_x2 / n
        var = max(ex2 - mean * mean, 0.0)
        std = math.sqrt(var)
        if std > 0:
            centered4 = (
                self.sum_x4 / n
                - 4 * mean * self.sum_x3 / n
                + 6 * mean * mean * self.sum_x2 / n
                - 3 * mean**4
            )
            kurtosis = centered4 / (std**4)
        else:
            kurtosis = float("nan")
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "variance": var,
            "kurtosis": kurtosis,
        }


def normal_right_tail(kappa: float) -> float:
    return 0.5 * math.erfc(kappa / math.sqrt(2.0))


def unit_gaussian_directions(hidden_size: int, n: int, generator: torch.Generator) -> torch.Tensor:
    dirs = torch.randn(hidden_size, n, generator=generator, dtype=torch.float32)
    return dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)


def unit_embedding_pair_directions(
    embedding: torch.Tensor,
    n: int,
    generator: torch.Generator,
) -> torch.Tensor:
    emb = embedding.float()
    vocab = emb.shape[0]
    left = torch.randint(vocab, (n,), generator=generator)
    right = torch.randint(vocab, (n,), generator=generator)
    diffs = emb[right] - emb[left]
    norms = diffs.norm(dim=1).clamp_min(1e-12)
    keep = norms > 1e-8
    attempts = 0
    while int(keep.sum().item()) < n and attempts < 10:
        missing = n - int(keep.sum().item())
        l2 = torch.randint(vocab, (missing,), generator=generator)
        r2 = torch.randint(vocab, (missing,), generator=generator)
        extra = emb[r2] - emb[l2]
        diffs = torch.cat([diffs[keep], extra], dim=0)
        norms = diffs.norm(dim=1).clamp_min(1e-12)
        keep = norms > 1e-8
        attempts += 1
    diffs = diffs[keep][:n]
    if diffs.shape[0] < n:
        raise RuntimeError("Could not sample enough nonzero embedding-pair directions.")
    dirs = diffs / diffs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return dirs.t().contiguous()


def histogram_edges(theoretical_sd: float, bins: int, sigma_range: float) -> torch.Tensor:
    return torch.linspace(-sigma_range * theoretical_sd, sigma_range * theoretical_sd, bins + 1)


def cosine_batch(weight: torch.Tensor, directions: torch.Tensor, device: torch.device) -> torch.Tensor:
    # weight is [d_model, channels]; each column is an MLP output vector.
    w = weight.to(device=device, dtype=torch.float32, non_blocking=True)
    d = directions.to(device=device, dtype=torch.float32, non_blocking=True)
    norms = w.norm(dim=0).clamp_min(1e-12)
    return torch.matmul(w.t(), d) / norms[:, None]


def run_one_source(
    *,
    model_alias: str,
    directions: torch.Tensor,
    source_name: str,
    layers: list[int],
    bins: int,
    sigma_range: float,
    tail_kappas: list[float],
    device: torch.device,
) -> dict[str, Any]:
    spec = model_spec(model_alias)
    index = weight_map(spec.path)
    templates = weight_templates(index)
    key_template = templates["mlp_down"]
    theoretical_sd = 1.0 / math.sqrt(spec.hidden_size)
    edges = histogram_edges(theoretical_sd, bins, sigma_range)
    hist = torch.zeros(bins, dtype=torch.long)
    stats = StreamingStats()
    tail_counts = {str(k): 0 for k in tail_kappas}
    abs_tail_counts = {str(k): 0 for k in tail_kappas}
    per_layer: dict[str, dict[str, Any]] = {}

    for layer in layers:
        key = key_template.format(layer=layer)
        weight = load_tensor(spec.path, index, key)
        if weight.ndim != 2 or weight.shape[0] != spec.hidden_size:
            raise ValueError(f"{key} expected [{spec.hidden_size}, channels], got {tuple(weight.shape)}")
        cos = cosine_batch(weight, directions, device)
        stats.update(cos)
        hist += torch.histc(cos.float(), bins=bins, min=float(edges[0]), max=float(edges[-1])).cpu().long()
        layer_stats = StreamingStats()
        layer_stats.update(cos)
        per_layer[str(layer)] = layer_stats.finalize()
        per_layer[str(layer)]["channels"] = int(weight.shape[1])
        for kappa in tail_kappas:
            threshold = kappa * theoretical_sd
            tail_counts[str(kappa)] += int((cos >= threshold).sum().item())
            abs_tail_counts[str(kappa)] += int((cos.abs() >= threshold).sum().item())
        del weight, cos
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final = stats.finalize()
    n = int(final["count"])
    final["theoretical_mean"] = 0.0
    final["theoretical_sd"] = theoretical_sd
    final["std_over_theory"] = float(final["std"]) / theoretical_sd
    final["mean_in_theory_sd"] = float(final["mean"]) / theoretical_sd
    final["random_direction_count"] = int(directions.shape[1])
    final["source"] = source_name
    final["layers"] = layers
    final["tail_rates"] = {
        str(k): {
            "empirical_right_tail": tail_counts[str(k)] / n,
            "normal_right_tail": normal_right_tail(k),
            "empirical_two_sided": abs_tail_counts[str(k)] / n,
            "normal_two_sided": 2.0 * normal_right_tail(k),
        }
        for k in tail_kappas
    }
    return {
        "summary": final,
        "histogram": {
            "edges": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in hist.tolist()],
        },
        "per_layer": per_layer,
    }


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if alias not in MODEL_PATHS or MODEL_PATHS[alias].startswith(("google/", "Qwen/")):
            if Path(path).exists():
                MODEL_PATHS[alias] = path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="gemma3_4b_it,gemma3_12b_it,qwen2_5_vl_3b_instruct,qwen3_vl_8b_instruct",
        help="Comma-separated model aliases.",
    )
    parser.add_argument("--num-directions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--sigma-range", type=float, default=5.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/appendix/null_cosine")
    args = parser.parse_args()

    resolve_local_model_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tail_kappas = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
    payload: dict[str, Any] = {
        "experiment": "null_cosine_distribution",
        "seed": args.seed,
        "num_directions_per_source": args.num_directions,
        "tail_kappas": tail_kappas,
        "models": {},
    }
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    payload["device"] = str(device)

    for model_alias in [part.strip() for part in args.models.split(",") if part.strip()]:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        spec = model_spec(model_alias)
        index = weight_map(spec.path)
        templates = weight_templates(index)
        embedding = load_tensor(spec.path, index, templates["embed"])
        layers = list(range(spec.num_layers // 2, spec.num_layers))
        gaussian = unit_gaussian_directions(spec.hidden_size, args.num_directions, generator)
        embedding_pairs = unit_embedding_pair_directions(embedding, args.num_directions, generator)
        del embedding

        spec_payload = asdict(spec)
        spec_payload["path"] = str(spec.path)
        model_payload = {
            "model_spec": spec_payload,
            "layer_policy": "upper_half_inclusive",
            "layers": layers,
            "sources": {},
        }
        for source_name, directions in [
            ("gaussian_unit", gaussian),
            ("embedding_pair", embedding_pairs),
        ]:
            print(f"[{model_alias}] source={source_name} layers={layers[0]}-{layers[-1]} dirs={directions.shape[1]}", flush=True)
            model_payload["sources"][source_name] = run_one_source(
                model_alias=model_alias,
                directions=directions,
                source_name=source_name,
                layers=layers,
                bins=args.bins,
                sigma_range=args.sigma_range,
                tail_kappas=tail_kappas,
                device=device,
            )
        payload["models"][model_alias] = model_payload

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# Null Cosine Distribution",
        "",
        "| Model | Source | $d_{model}$ | Layers | N cosines | Mean | Std | Theory std | Std/Theory | P($s>2\\sigma$) | P($s>3\\sigma$) | P($s>4\\sigma$) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_alias, model_payload in payload["models"].items():
        spec = model_payload["model_spec"]
        layers = model_payload["layers"]
        for source_name, source_payload in model_payload["sources"].items():
            s = source_payload["summary"]
            tails = s["tail_rates"]
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        model_alias,
                        source_name,
                        str(spec["hidden_size"]),
                        f"{layers[0]}--{layers[-1]}",
                        str(s["count"]),
                        f"{s['mean']:.3e}",
                        f"{s['std']:.6f}",
                        f"{s['theoretical_sd']:.6f}",
                        f"{s['std_over_theory']:.3f}",
                        f"{tails['2.0']['empirical_right_tail']:.5f}",
                        f"{tails['3.0']['empirical_right_tail']:.5f}",
                        f"{tails['4.0']['empirical_right_tail']:.5f}",
                    ]
                )
                + " |"
            )
    (args.output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
