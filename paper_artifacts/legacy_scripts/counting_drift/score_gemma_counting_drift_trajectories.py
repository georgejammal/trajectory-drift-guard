#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


DRIFTING_ROOT = Path(__file__).resolve().parents[2]
DRIFTING_SCRIPTS = DRIFTING_ROOT / "scripts"
if str(DRIFTING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DRIFTING_SCRIPTS))

from evaluate_gemma_counting import (  # noqa: E402
    build_chat_prompt,
    gold_from_row,
    image_from_row,
    row_question,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "resources" / "configs" / "gemma_counting_final.json"
DEFAULT_PREDICTIONS_ROOT = (
    DRIFTING_ROOT
    / "outputs/experiment_runs/counting/gemma3_counting_chat_baseline/gemma_counting_baseline_paper/gemma3_4b_it"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "experiment_runs" / "counting"
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score whether Gemma counting trajectories separate baseline-correct "
            "from baseline-wrong examples using selected MLP down-projection neurons."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--datasets", default="countbenchqa,how_many,tallyqa_balanced")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prediction-condition", default="factor_1p0")
    parser.add_argument(
        "--scores-path",
        type=Path,
        default=None,
        help="Optional existing sample_layer_scores.jsonl. If set, skip model loading and only summarize windows.",
    )
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--min-window-width", type=int, default=2)
    parser.add_argument("--max-window-width", type=int, default=7)
    parser.add_argument("--min-layer", type=int, default=18)
    parser.add_argument("--max-layer", type=int, default=31)
    parser.add_argument(
        "--direction-modes",
        default="neighbor_digits_0_9,all_wrong_digits_0_9,mean_word_minus_gold_digit",
        help="Comma-separated direction definitions to evaluate.",
    )
    parser.add_argument(
        "--score-sides",
        default="qpos,allq",
        help="qpos keeps selected neurons with positive q_y; allq uses all selected neurons.",
    )
    parser.add_argument("--make-plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def utc_run_id(raw: str | None) -> str:
    if raw:
        return raw
    return "gemma_counting_drift_trajectories_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_config_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else DRIFTING_ROOT / path)


def normalize_config(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    model_path_env = payload.get("model_path_env")
    if model_path_env and os.environ.get(str(model_path_env)):
        payload["model_path"] = os.environ[str(model_path_env)]
    payload.setdefault("model_path", "google/gemma-3-4b-it")
    if "datasets" in payload:
        payload["datasets"] = {
            name: resolve_config_path(path)
            for name, path in payload["datasets"].items()
        }
    for key in ("neuron_set", "output_root", "log_dir"):
        if key in payload:
            payload[key] = resolve_config_path(payload[key])
    return payload


def read_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "datasets" in payload and "neuron_set" in payload:
        return normalize_config(payload)
    return payload


def load_neurons(path: Path, top_k: int) -> dict[int, list[int]]:
    payload = read_json(path)
    rows = payload.get("top_neurons") or payload.get("global_top_neurons")
    if rows is None:
        raise ValueError(f"Could not find top_neurons/global_top_neurons in {path}")
    rows = rows[:top_k]
    by_layer: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        layer = int(row["layer"])
        neuron = int(row.get("neuron", row.get("neuron_index")))
        by_layer[layer].append(neuron)
    return dict(sorted(by_layer.items()))


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_dataset(loaded: Any, split: str):
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    return loaded


def batch_items(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def unembedding_weight(model: Gemma3ForConditionalGeneration) -> torch.Tensor:
    candidates = [
        ("lm_head", getattr(model, "lm_head", None)),
        ("language_model.lm_head", getattr(getattr(model, "language_model", None), "lm_head", None)),
        (
            "model.language_model.lm_head",
            getattr(getattr(getattr(model, "model", None), "language_model", None), "lm_head", None),
        ),
    ]
    for _, head in candidates:
        if head is not None and hasattr(head, "weight"):
            return head.weight.detach().float()
    raise AttributeError("Could not locate lm_head.weight for Gemma3ForConditionalGeneration")


def token_mean_vector(tokenizer: Any, unembed: torch.Tensor, text: str) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no token ids for {text!r}")
    return unembed[token_ids].float().mean(dim=0)


def wrong_digit_texts(gold: int, mode: str) -> list[str]:
    if mode == "neighbor_digits_0_9":
        texts = []
        if gold - 1 >= 0:
            texts.append(str(gold - 1))
        if gold + 1 <= 9:
            texts.append(str(gold + 1))
        return texts
    if mode == "all_wrong_digits_0_9":
        return [str(value) for value in range(10) if value != gold]
    if mode == "mean_word_minus_gold_digit":
        return [str(gold)]
    raise ValueError(f"Unknown direction mode: {mode}")


def build_directions(
    tokenizer: Any,
    unembed: torch.Tensor,
    modes: list[str],
    gold_values: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    directions: dict[str, dict[int, torch.Tensor]] = {}
    word_cache = {value: token_mean_vector(tokenizer, unembed, NUMBER_WORDS[value]) for value in gold_values}
    digit_cache = {str(value): token_mean_vector(tokenizer, unembed, str(value)) for value in range(10)}
    for mode in modes:
        directions[mode] = {}
        for gold in gold_values:
            answer_vectors = torch.stack([digit_cache[text] for text in wrong_digit_texts(gold, mode)])
            raw = word_cache[gold] - answer_vectors.mean(dim=0)
            directions[mode][gold] = F.normalize(raw, dim=0).to(unembed.device)
    return directions


def build_q_tables(
    model: Gemma3ForConditionalGeneration,
    neurons_by_layer: dict[int, list[int]],
    directions: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict[int, dict[int, torch.Tensor]]]:
    layers = model.model.language_model.layers
    q_tables: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
    for mode, by_gold in directions.items():
        q_tables[mode] = {}
        for gold, direction in by_gold.items():
            q_tables[mode][gold] = {}
            d = direction.to(next(model.parameters()).device).float()
            for layer_idx, neuron_indices in neurons_by_layer.items():
                weight = layers[layer_idx].mlp.down_proj.weight.detach().float()
                cols = weight[:, torch.tensor(neuron_indices, device=weight.device)].T
                q_tables[mode][gold][layer_idx] = (cols @ d).detach().cpu().float()
    return q_tables


class GemmaGatedActivationTracer:
    def __init__(self, model: Gemma3ForConditionalGeneration, neurons_by_layer: dict[int, list[int]]):
        self.model = model
        self.neurons_by_layer = neurons_by_layer
        self.original_forwards: dict[int, Any] = {}
        self.final_positions: torch.Tensor | None = None
        self.cache: dict[int, torch.Tensor] = {}

    def set_final_positions(self, positions: torch.Tensor) -> None:
        self.final_positions = positions.detach().long()
        self.cache = {}

    def __enter__(self) -> "GemmaGatedActivationTracer":
        layers = self.model.model.language_model.layers
        for layer_idx, neuron_indices in self.neurons_by_layer.items():
            mlp = layers[layer_idx].mlp
            self.original_forwards[layer_idx] = mlp.forward
            local_indices = torch.tensor(neuron_indices, dtype=torch.long)

            def patched_forward(module, x, indices=local_indices, layer=layer_idx):
                gate_act = module.act_fn(module.gate_proj(x))
                gated = gate_act * module.up_proj(x)
                if self.final_positions is None:
                    raise RuntimeError("final positions were not set before model forward")
                positions = self.final_positions.to(gated.device)
                batch_ids = torch.arange(gated.shape[0], device=gated.device)
                selected = indices.to(gated.device)
                layer_values = gated[batch_ids, positions][:, selected]
                self.cache[layer] = layer_values.detach().float().cpu()
                return module.down_proj(gated)

            mlp.forward = MethodType(patched_forward, mlp)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        layers = self.model.model.language_model.layers
        for layer_idx, original_forward in self.original_forwards.items():
            layers[layer_idx].mlp.forward = original_forward
        return False


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    reversed_first_one = torch.flip(attention_mask.long(), dims=[1]).argmax(dim=1)
    return attention_mask.shape[1] - 1 - reversed_first_one


def rank_auc(labels: list[int], scores: list[float]) -> float | None:
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    rank = 1
    idx = 0
    while idx < n:
        j = idx
        while j + 1 < n and scores[order[j + 1]] == scores[order[idx]]:
            j += 1
        avg_rank = (rank + rank + (j - idx)) / 2.0
        for k in range(idx, j + 1):
            ranks[order[k]] = avg_rank
        rank += j - idx + 1
        idx = j + 1
    pos_rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_stats(labels: list[int], scores: list[float]) -> dict[str, float]:
    if not labels:
        return {
            "threshold": 0.0,
            "youden_j": 0.0,
            "accuracy": 0.0,
            "tpr": 0.0,
            "fpr": 0.0,
            "balanced_accuracy": 0.0,
        }
    best = {
        "threshold": float(scores[0]) if scores else 0.0,
        "youden_j": -math.inf,
        "accuracy": 0.0,
        "tpr": 0.0,
        "fpr": 0.0,
        "balanced_accuracy": 0.0,
    }
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return {
            "threshold": float(scores[0]) if scores else 0.0,
            "youden_j": 0.0,
            "accuracy": 1.0 if labels else 0.0,
            "tpr": 1.0 if pos else 0.0,
            "fpr": 0.0,
            "balanced_accuracy": 1.0 if labels else 0.0,
        }

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    fn = pos
    tn = neg
    idx = 0
    while idx < len(pairs):
        threshold = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] == threshold:
            if pairs[idx][1] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            idx += 1
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        tnr = tn / neg if neg else 0.0
        acc = (tp + tn) / len(labels) if labels else 0.0
        bal = 0.5 * (tpr + tnr)
        j = tpr - fpr
        if j > best["youden_j"]:
            best = {
                "threshold": float(threshold),
                "youden_j": float(j),
                "accuracy": float(acc),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "balanced_accuracy": float(bal),
            }
    return best


def load_score_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
    sorted_values = sorted(values)
    n = len(sorted_values)

    def q(p: float) -> float:
        if n == 1:
            return sorted_values[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_values[lo]
        frac = pos - lo
        return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac

    return {
        "mean": float(sum(values) / len(values)),
        "median": float(q(0.5)),
        "q25": float(q(0.25)),
        "q75": float(q(0.75)),
    }


def evaluate_windows(
    samples: list[dict[str, Any]],
    modes: list[str],
    score_sides: list[str],
    min_layer: int,
    max_layer: int,
    min_width: int,
    max_width: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    layer_windows: list[tuple[int, int]] = []
    for start in range(min_layer, max_layer + 1):
        for width in range(min_width, max_width + 1):
            end = start + width - 1
            if end <= max_layer:
                layer_windows.append((start, end))

    dataset_names = sorted({sample["dataset"] for sample in samples})
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", samples)]
    for dataset in dataset_names:
        scopes.append((dataset, [sample for sample in samples if sample["dataset"] == dataset]))

    for mode in modes:
        for side in score_sides:
            key = f"{mode}/{side}"
            for scope_name, scope_samples in scopes:
                labels = [int(not sample["baseline_correct"]) for sample in scope_samples]
                if sum(labels) == 0 or sum(labels) == len(labels):
                    continue
                for start, end in layer_windows:
                    scores = []
                    for sample in scope_samples:
                        b_value = sum(
                            sample["layer_work"][key].get(str(layer), 0.0)
                            for layer in range(start, end + 1)
                        )
                        scores.append(-b_value)
                    auc = rank_auc(labels, scores)
                    if auc is None:
                        continue
                    correct_scores = [score for label, score in zip(labels, scores) if label == 0]
                    wrong_scores = [score for label, score in zip(labels, scores) if label == 1]
                    summaries.append(
                        {
                            "scope": scope_name,
                            "direction_mode": mode,
                            "score_side": side,
                            "window_start": start,
                            "window_end": end,
                            "window": f"{start}-{end}",
                            "width": end - start + 1,
                            "auc_wrong_high": float(auc),
                            "n": len(scope_samples),
                            "n_correct": len(correct_scores),
                            "n_wrong": len(wrong_scores),
                            "threshold": threshold_stats(labels, scores),
                            "correct_score_stats": quantiles(correct_scores),
                            "wrong_score_stats": quantiles(wrong_scores),
                        }
                    )
    return sorted(
        summaries,
        key=lambda row: (row["scope"] != "all", -row["auc_wrong_high"], row["window_start"], row["window_end"]),
    )


def write_markdown(summary_path: Path, run: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Gemma Counting Drift Trajectory Scores",
        "",
        f"- Samples: {run['n_samples']} total, {run['n_correct']} baseline-correct, {run['n_wrong']} baseline-wrong.",
        f"- Score: `S_W(x;y) = - sum_{{ell in W}} sum_{{i in S_ell}} a_{{ell i}}(x,t*) q_{{ell i}}(y)`.",
        "- Positive label for AUC: baseline-wrong trajectory; higher score should mean more drift-like.",
        f"- Selected neurons: {run['neuron_json']} top_k={run['top_k']}.",
        "",
        "## Best Overall Windows",
        "",
        "| Rank | Direction | Side | Window | AUC | Correct mean | Wrong mean | Threshold | Bal. acc |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    all_rows = [row for row in summaries if row["scope"] == "all"]
    for rank, row in enumerate(all_rows[:20], start=1):
        lines.append(
            "| {rank} | {direction} | {side} | {window} | {auc:.4f} | {cmean:.4f} | {wmean:.4f} | {thr:.4f} | {bal:.4f} |".format(
                rank=rank,
                direction=row["direction_mode"],
                side=row["score_side"],
                window=row["window"],
                auc=row["auc_wrong_high"],
                cmean=row["correct_score_stats"]["mean"],
                wmean=row["wrong_score_stats"]["mean"],
                thr=row["threshold"]["threshold"],
                bal=row["threshold"]["balanced_accuracy"],
            )
        )
    lines.extend(["", "## Best Per Dataset", ""])
    for dataset in sorted({row["scope"] for row in summaries if row["scope"] != "all"}):
        best = [row for row in summaries if row["scope"] == dataset][:10]
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Rank | Direction | Side | Window | AUC | Correct mean | Wrong mean | Bal. acc |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|")
        for rank, row in enumerate(best, start=1):
            lines.append(
                "| {rank} | {direction} | {side} | {window} | {auc:.4f} | {cmean:.4f} | {wmean:.4f} | {bal:.4f} |".format(
                    rank=rank,
                    direction=row["direction_mode"],
                    side=row["score_side"],
                    window=row["window"],
                    auc=row["auc_wrong_high"],
                    cmean=row["correct_score_stats"]["mean"],
                    wmean=row["wrong_score_stats"]["mean"],
                    bal=row["threshold"]["balanced_accuracy"],
                )
            )
        lines.append("")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(output_dir: Path, samples: list[dict[str, Any]], best: dict[str, Any]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)
        return None

    key = f"{best['direction_mode']}/{best['score_side']}"
    start = int(best["window_start"])
    end = int(best["window_end"])
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    for sample in samples:
        score = -sum(sample["layer_work"][key].get(str(layer), 0.0) for layer in range(start, end + 1))
        if sample["baseline_correct"]:
            correct_scores.append(score)
        else:
            wrong_scores.append(score)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.boxplot([correct_scores, wrong_scores], labels=["Correct", "Wrong"], showfliers=False)
    ax.set_ylabel("Drift score")
    ax.set_title(f"{best['direction_mode']} {best['score_side']} W={start}-{end}")
    ax.axhline(float(best["threshold"]["threshold"]), color="tab:red", linewidth=1, linestyle="--")
    fig.tight_layout()
    path = output_dir / "best_counting_drift_boxplot.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def main() -> None:
    if False:
        os.chdir("/tmp")

    args = parse_args()
    config = read_json(args.config)
    dataset_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    modes = [part.strip() for part in args.direction_modes.split(",") if part.strip()]
    score_sides = [part.strip() for part in args.score_sides.split(",") if part.strip()]
    unknown = sorted(set(dataset_names) - set(config["datasets"]))
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; known={sorted(config['datasets'])}")

    run_id = utc_run_id(args.run_id)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scores_path is not None:
        print(f"[load] existing scores={args.scores_path}", flush=True)
        samples = load_score_samples(args.scores_path)
        summaries = evaluate_windows(
            samples,
            modes=modes,
            score_sides=score_sides,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
            min_width=args.min_window_width,
            max_width=args.max_window_width,
        )
        summary_json = output_dir / "window_auc_summary.json"
        summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        run_manifest = {
            "run_id": run_id,
            "output_dir": str(output_dir),
            "config": str(args.config),
            "model_path": config["model_path"],
            "neuron_json": config["neuron_set"],
            "top_k": args.top_k,
            "predictions_root": str(args.predictions_root),
            "prediction_condition": args.prediction_condition,
            "datasets": dataset_names,
            "direction_modes": modes,
            "score_sides": score_sides,
            "window_search": {
                "min_layer": args.min_layer,
                "max_layer": args.max_layer,
                "min_width": args.min_window_width,
                "max_width": args.max_window_width,
            },
            "score_definition": "S_W(x;y) = - sum_{ell in W} sum_{i in selected MLP down-proj neurons} a_{ell i}(x,t*) <w_{ell i}, d_y>",
            "n_samples": len(samples),
            "n_correct": sum(1 for sample in samples if sample["baseline_correct"]),
            "n_wrong": sum(1 for sample in samples if not sample["baseline_correct"]),
            "samples_path": str(args.scores_path),
            "summary_json": str(summary_json),
        }
        if args.make_plot and summaries:
            run_manifest["plot_path"] = maybe_plot(output_dir, samples, summaries[0])
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_markdown(output_dir / "README.md", run_manifest, summaries)
        print(f"[done] {output_dir}", flush=True)
        if summaries:
            best = summaries[0]
            print(
                "[best] scope={scope} direction={direction_mode} side={score_side} "
                "window={window} auc={auc_wrong_high:.4f} threshold={thr:.4f} bal_acc={bal:.4f}".format(
                    **best,
                    thr=best["threshold"]["threshold"],
                    bal=best["threshold"]["balanced_accuracy"],
                ),
                flush=True,
            )
        return

    neurons_by_layer = load_neurons(Path(config["neuron_set"]), args.top_k)
    print(f"[load] processor={config['model_path']}", flush=True)
    processor = AutoProcessor.from_pretrained(config["model_path"])
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    print(f"[load] model={config['model_path']}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        config["model_path"],
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    unembed = unembedding_weight(model)
    directions = build_directions(
        processor.tokenizer,
        unembed,
        modes,
        gold_values=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    q_tables = build_q_tables(model, neurons_by_layer, directions)

    samples: list[dict[str, Any]] = []
    dataset_cache: dict[str, Any] = {}
    for dataset_name in dataset_names:
        dataset = as_dataset(load_from_disk(config["datasets"][dataset_name]), args.split)
        dataset_cache[dataset_name] = dataset
        pred_path = args.predictions_root / dataset_name / args.prediction_condition / "predictions.jsonl"
        predictions = load_predictions(pred_path)
        if args.limit_per_dataset is not None:
            predictions = predictions[: args.limit_per_dataset]
        for pred in predictions:
            source_index = int(pred["source_index"])
            row = dataset[source_index]
            gold = int(pred["gold"])
            row_gold = int(gold_from_row(row))
            if row_gold != gold:
                raise ValueError(
                    f"Gold mismatch for {dataset_name}:{source_index}: prediction={gold}, row={row_gold}"
                )
            samples.append(
                {
                    "dataset": dataset_name,
                    "source_index": source_index,
                    "question": pred["question"],
                    "gold": gold,
                    "baseline_prediction_text": pred["prediction_text"],
                    "baseline_prediction_number": pred["prediction_number"],
                    "baseline_correct": bool(pred["correct"]),
                }
            )

    scores_path = output_dir / "sample_layer_scores.jsonl"
    start_time = time.time()
    with scores_path.open("w", encoding="utf-8") as handle:
        with GemmaGatedActivationTracer(model, neurons_by_layer) as tracer:
            for batch_idx, batch in enumerate(batch_items(samples, args.batch_size), start=1):
                rows = [dataset_cache[item["dataset"]][item["source_index"]] for item in batch]
                prompts = [
                    build_chat_prompt(
                        processor,
                        row_question(row),
                        config.get("question_prefix", ""),
                        config.get("suffix", " Answer with one number don't add any further explanations."),
                    )
                    for row in rows
                ]
                images = [[image_from_row(row)] for row in rows]
                inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(model.device)
                final_positions = last_nonpad_positions(inputs["attention_mask"]).to(model.device)
                tracer.set_final_positions(final_positions)
                with torch.inference_mode():
                    _ = model(**inputs, use_cache=False)

                for row_idx, sample in enumerate(batch):
                    gold = int(sample["gold"])
                    layer_work: dict[str, dict[str, float]] = {}
                    for mode in modes:
                        for side in score_sides:
                            key = f"{mode}/{side}"
                            layer_work[key] = {}
                            for layer_idx, neuron_indices in neurons_by_layer.items():
                                acts = tracer.cache[layer_idx][row_idx].float()
                                q = q_tables[mode][gold][layer_idx].float()
                                if side == "qpos":
                                    mask = q > 0
                                    value = float((acts[mask] * q[mask]).sum().item())
                                elif side == "allq":
                                    value = float((acts * q).sum().item())
                                else:
                                    raise ValueError(f"Unknown score side: {side}")
                                layer_work[key][str(layer_idx)] = value
                    enriched = {**sample, "layer_work": layer_work}
                    samples[(batch_idx - 1) * args.batch_size + row_idx] = enriched
                    handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

                elapsed = time.time() - start_time
                seen = min(batch_idx * args.batch_size, len(samples))
                print(f"[score] batch={batch_idx} seen={seen}/{len(samples)} elapsed={elapsed:.1f}s", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summaries = evaluate_windows(
        samples,
        modes=modes,
        score_sides=score_sides,
        min_layer=args.min_layer,
        max_layer=args.max_layer,
        min_width=args.min_window_width,
        max_width=args.max_window_width,
    )
    summary_json = output_dir / "window_auc_summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    run_manifest = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "config": str(args.config),
        "model_path": config["model_path"],
        "neuron_json": config["neuron_set"],
        "top_k": args.top_k,
        "predictions_root": str(args.predictions_root),
        "prediction_condition": args.prediction_condition,
        "datasets": dataset_names,
        "direction_modes": modes,
        "score_sides": score_sides,
        "window_search": {
            "min_layer": args.min_layer,
            "max_layer": args.max_layer,
            "min_width": args.min_window_width,
            "max_width": args.max_window_width,
        },
        "score_definition": "S_W(x;y) = - sum_{ell in W} sum_{i in selected MLP down-proj neurons} a_{ell i}(x,t*) <w_{ell i}, d_y>",
        "n_samples": len(samples),
        "n_correct": sum(1 for sample in samples if sample["baseline_correct"]),
        "n_wrong": sum(1 for sample in samples if not sample["baseline_correct"]),
        "samples_path": str(scores_path),
        "summary_json": str(summary_json),
    }
    plot_path = None
    if args.make_plot and summaries:
        plot_path = maybe_plot(output_dir, samples, summaries[0])
        run_manifest["plot_path"] = plot_path
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md = output_dir / "README.md"
    write_markdown(summary_md, run_manifest, summaries)

    print(f"[done] {output_dir}", flush=True)
    if summaries:
        best = summaries[0]
        print(
            "[best] scope={scope} direction={direction_mode} side={score_side} "
            "window={window} auc={auc_wrong_high:.4f} threshold={thr:.4f} bal_acc={bal:.4f}".format(
                **best,
                thr=best["threshold"]["threshold"],
                bal=best["threshold"]["balanced_accuracy"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
