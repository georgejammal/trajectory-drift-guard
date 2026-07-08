#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from generate_gemma3_safety_completions import (
    GEMMA3_PATH,
    MLPActivationIntervention,
    OUT_ROOT,
    build_chat_prompt,
    dtype_from_name,
    load_neurons,
)
from safety_benchmark_common import REFUSAL_PREFIXES, dump_json, write_jsonl


REPO = Path("/home/georgejammal/projects/semantic-to-symbolic")
SAMPLE_ROOT = REPO / "safeguard_eval" / "sample_sets"
RAW_ROOT = REPO / "safeguard_eval" / "benign_data"
REFUSAL_SET = (
    REPO
    / "safeguard_logit_lens"
    / "neuron_sets"
    / "gemma3_4b_it_refusal_minus_answerability_top1000_layers17_33.json"
)

PIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
SIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"

LETTER_RE = re.compile(r"(?i)(?:^|\b)(?:answer|option|choice)?\s*(?:is|:)?\s*([A-D])(?:\b|[\).:])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma3 baseline vs safety intervention on Katz-style benign utility benchmarks."
    )
    parser.add_argument("--model-path", default=GEMMA3_PATH)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--only-condition", choices=["baseline", "intervention"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_id(raw: str | None) -> str:
    return raw or datetime.now(timezone.utc).strftime("gemma3_benign_utility_%Y%m%d_%H%M%S")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_zip(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_extracted(url: str, zip_path: Path, extract_dir: Path) -> Path:
    marker = extract_dir / ".complete"
    if marker.exists():
        return extract_dir
    download_zip(url, zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return extract_dir


def choice_prompt(*, question: str, choices: list[tuple[str, str]], intro: str | None = None) -> str:
    parts = []
    if intro:
        parts.append(intro.strip())
    parts.append(f"Question: {question.strip()}")
    parts.extend(f"{label}. {text.strip()}" for label, text in choices)
    labels = ", ".join(label for label, _ in choices)
    parts.append(f"Choose the best answer. Answer with only the option letter ({labels}).")
    return "\n".join(parts)


def load_piqa() -> list[dict[str, Any]]:
    root = ensure_extracted(PIQA_URL, RAW_ROOT / "physicaliqa-train-dev.zip", RAW_ROOT / "physicaliqa-train-dev")
    data_dir = root / "physicaliqa-train-dev"
    rows_raw = (data_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    labels = (data_dir / "dev-labels.lst").read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for idx, (line, label) in enumerate(zip(rows_raw, labels)):
        item = json.loads(line)
        choices = [("A", item["sol1"]), ("B", item["sol2"])]
        gold = "A" if str(label).strip() == "0" else "B"
        rows.append(
            {
                "benchmark": "piqa",
                "source_index": idx,
                "question": item["goal"],
                "choices": choices,
                "gold": gold,
                "prompt": choice_prompt(question=item["goal"], choices=choices),
            }
        )
    return rows


def load_siqa() -> list[dict[str, Any]]:
    root = ensure_extracted(SIQA_URL, RAW_ROOT / "socialiqa-train-dev.zip", RAW_ROOT / "socialiqa-train-dev")
    data_dir = root / "socialiqa-train-dev"
    rows_raw = (data_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines()
    labels = (data_dir / "dev-labels.lst").read_text(encoding="utf-8").splitlines()
    label_map = {"1": "A", "2": "B", "3": "C"}
    rows: list[dict[str, Any]] = []
    for idx, (line, label) in enumerate(zip(rows_raw, labels)):
        item = json.loads(line)
        choices = [("A", item["answerA"]), ("B", item["answerB"]), ("C", item["answerC"])]
        rows.append(
            {
                "benchmark": "siqa",
                "source_index": idx,
                "question": item["question"],
                "context": item["context"],
                "choices": choices,
                "gold": label_map[str(label).strip()],
                "prompt": choice_prompt(question=item["question"], choices=choices, intro=f"Context: {item['context']}"),
            }
        )
    return rows


def load_openbookqa() -> list[dict[str, Any]]:
    ds = load_dataset("allenai/openbookqa", "main", split="validation")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        choices = list(zip(item["choices"]["label"], item["choices"]["text"]))
        rows.append(
            {
                "benchmark": "openbookqa",
                "source_index": idx,
                "question": item["question_stem"],
                "choices": choices,
                "gold": item["answerKey"],
                "prompt": choice_prompt(question=item["question_stem"], choices=choices),
            }
        )
    return rows


def load_arc() -> list[dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(ds):
        labels = [str(label) for label in item["choices"]["label"]]
        choices = list(zip(labels, item["choices"]["text"]))
        gold = str(item["answerKey"])
        if gold.isdigit() and not labels[0].isdigit():
            gold_idx = int(gold) - 1
            gold = labels[gold_idx]
        rows.append(
            {
                "benchmark": "arc_challenge",
                "source_index": idx,
                "question": item["question"],
                "choices": choices,
                "gold": gold,
                "prompt": choice_prompt(question=item["question"], choices=choices),
            }
        )
    return rows


def build_samples(seed: int, samples_per_benchmark: int) -> list[dict[str, Any]]:
    loaders = {
        "piqa": load_piqa,
        "openbookqa": load_openbookqa,
        "siqa": load_siqa,
        "arc_challenge": load_arc,
    }
    all_samples: list[dict[str, Any]] = []
    for benchmark, loader in loaders.items():
        rows = loader()
        rng = random.Random(f"{seed}:{benchmark}")
        selected = rng.sample(range(len(rows)), min(samples_per_benchmark, len(rows)))
        for rank, idx in enumerate(selected):
            row = rows[idx]
            row = dict(row)
            row["sample_id"] = f"{benchmark}__{row['source_index']:05d}"
            row["selection_seed"] = seed
            row["selection_rank"] = rank
            row["choice_labels"] = [label for label, _ in row["choices"]]
            row["choices"] = [{"label": label, "text": text} for label, text in row["choices"]]
            all_samples.append(row)
    return all_samples


def is_refusal(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if any(stripped.startswith(prefix) for prefix in REFUSAL_PREFIXES):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "i can't answer",
            "i cannot answer",
            "i can't assist",
            "i cannot assist",
            "i'm unable",
            "i am unable",
            "i can't provide",
            "i cannot provide",
        )
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def extract_prediction(completion: str, choices: list[dict[str, str]]) -> str | None:
    labels = [choice["label"] for choice in choices]
    allowed = set(labels)
    text = completion.strip()
    first = re.match(r"^\s*([A-D])(?:\b|[\).:])", text, flags=re.I)
    if first:
        label = first.group(1).upper()
        if label in allowed:
            return label
    for match in LETTER_RE.finditer(text[:200]):
        label = match.group(1).upper()
        if label in allowed:
            return label
    norm = normalize_text(text[:500])
    for choice in choices:
        choice_norm = normalize_text(choice["text"])
        if choice_norm and (norm.startswith(choice_norm) or choice_norm in norm):
            return choice["label"]
    return None


def generate_condition(
    *,
    model: Gemma3ForConditionalGeneration,
    processor: Any,
    samples: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    max_new_tokens: int,
    condition: str,
    intervention: MLPActivationIntervention,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    rows: list[dict[str, Any]] = []
    start = time.time()
    with intervention:
        for batch_idx, batch in enumerate(batched(samples, batch_size), start=1):
            prompts = [build_chat_prompt(processor, sample["prompt"]) for sample in batch]
            inputs = processor(text=prompts, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            input_len = int(inputs["input_ids"].shape[1])
            decoded = processor.batch_decode(
                generated[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample, completion in zip(batch, decoded):
                pred = extract_prediction(completion, sample["choices"])
                row = {
                    **sample,
                    "condition": condition,
                    "completion": completion.strip(),
                    "prediction": pred,
                    "correct": pred == sample["gold"],
                    "refusal": is_refusal(completion),
                }
                rows.append(row)
            write_jsonl(output_path, rows)
            print(
                f"[generate {condition}] batch={batch_idx} seen={len(rows)}/{len(samples)} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    summary: dict[str, Any] = {
        "total": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / max(len(rows), 1),
        "refusal_rate": sum(bool(row["refusal"]) for row in rows) / max(len(rows), 1),
        "unparsed_rate": sum(row["prediction"] is None for row in rows) / max(len(rows), 1),
        "by_benchmark": {},
    }
    for benchmark, group in sorted(by_benchmark.items()):
        predictions = Counter(str(row["prediction"]) for row in group)
        summary["by_benchmark"][benchmark] = {
            "total": len(group),
            "accuracy": sum(bool(row["correct"]) for row in group) / max(len(group), 1),
            "refusal_rate": sum(bool(row["refusal"]) for row in group) / max(len(group), 1),
            "unparsed_rate": sum(row["prediction"] is None for row in group) / max(len(group), 1),
            "prediction_counts": dict(sorted(predictions.items())),
        }
    return summary


def write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    text = f"""# Gemma3 Benign Utility Evaluation

Katz-style benign benchmark check for performance degradation.

## Benchmarks
- PIQA validation/dev: `{manifest["samples_per_benchmark"]}` deterministic samples.
- OpenBookQA validation: `{manifest["samples_per_benchmark"]}` deterministic samples.
- Social IQA validation/dev: `{manifest["samples_per_benchmark"]}` deterministic samples.
- ARC-Challenge validation: `{manifest["samples_per_benchmark"]}` deterministic samples.

## Generation
- Model: `{manifest["model_path"]}`
- Prompting: Gemma chat template, text-only user message, `add_generation_prompt=True`.
- Decoding: greedy, `do_sample=False`.
- Max new tokens: `{manifest["max_new_tokens"]}`.
- Batch size: `{manifest["batch_size"]}`.

## Metrics
- Accuracy: parsed multiple-choice option equals gold option.
- Refusal rate: completion matches the same refusal-prefix heuristic used in the safety scripts,
  with extra benign false-refusal phrases.
- Unparsed rate: no option letter or option text could be extracted from the completion.

## Intervention
- Winning safety intervention: `refusal_minus_answerability_top1000_abs`.
- Site: Gemma language-model MLP gated intermediate before `down_proj`.
- Token scope: all prompt positions and all autoregressive generation steps.
- Neuron set: `{manifest["neuron_set"]}`.
"""
    output_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rid = run_id(args.run_id)
    output_dir = args.output_root / rid / "gemma3_4b_it"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLE_ROOT / f"benign_utility_katz_style_seed{args.seed}_n{args.samples_per_benchmark}.jsonl"
    if sample_path.exists():
        samples = read_jsonl(sample_path)
    else:
        samples = build_samples(args.seed, args.samples_per_benchmark)
        write_jsonl(sample_path, samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": rid,
        "paper_protocol": "Katz et al. benign capability check: PIQA, OpenBookQA, SIQA, ARC; 100 random samples per benchmark; accuracy and refusal rate.",
        "model_path": args.model_path,
        "sample_set": str(sample_path),
        "sample_count": len(samples),
        "seed": args.seed,
        "samples_per_benchmark": args.samples_per_benchmark,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "neuron_set": str(REFUSAL_SET),
        "conditions": ["baseline", "refusal_minus_answerability_top1000_abs"],
    }
    dump_json(output_dir / "manifest.json", manifest)
    write_readme(output_dir, manifest)

    print(f"[load] processor={args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    print(f"[load] model={args.model_path}", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
    ).eval()

    summaries: dict[str, Any] = {}
    conditions = [
        ("baseline", MLPActivationIntervention(model, {}, "none", 1.0, False)),
    ]
    if args.only_condition in (None, "intervention"):
        neurons_by_layer, neuron_metadata = load_neurons(REFUSAL_SET, 1000, "cosine_similarity", "none")
        manifest["intervention_neuron_metadata"] = neuron_metadata
        dump_json(output_dir / "manifest.json", manifest)
        conditions.append(
            (
                "refusal_minus_answerability_top1000_abs",
                MLPActivationIntervention(model, neurons_by_layer, "abs", 1.0, False),
            )
        )
    if args.only_condition == "intervention":
        conditions = conditions[1:]

    for condition, intervention in conditions:
        condition_dir = output_dir / condition
        rows = generate_condition(
            model=model,
            processor=processor,
            samples=samples,
            output_path=condition_dir / "completions.jsonl",
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            condition=condition,
            intervention=intervention,
        )
        condition_summary = summarize(rows)
        dump_json(condition_dir / "summary.json", condition_summary)
        summaries[condition] = condition_summary
        print(json.dumps({condition: condition_summary}, indent=2), flush=True)

    if "baseline" in summaries and "refusal_minus_answerability_top1000_abs" in summaries:
        base = summaries["baseline"]
        intr = summaries["refusal_minus_answerability_top1000_abs"]
        comparison = {
            "baseline_accuracy": base["accuracy"],
            "intervention_accuracy": intr["accuracy"],
            "accuracy_delta": intr["accuracy"] - base["accuracy"],
            "baseline_refusal_rate": base["refusal_rate"],
            "intervention_refusal_rate": intr["refusal_rate"],
            "refusal_rate_delta": intr["refusal_rate"] - base["refusal_rate"],
            "by_benchmark": {},
        }
        for benchmark in sorted(base["by_benchmark"]):
            b = base["by_benchmark"][benchmark]
            i = intr["by_benchmark"][benchmark]
            comparison["by_benchmark"][benchmark] = {
                "baseline_accuracy": b["accuracy"],
                "intervention_accuracy": i["accuracy"],
                "accuracy_delta": i["accuracy"] - b["accuracy"],
                "baseline_refusal_rate": b["refusal_rate"],
                "intervention_refusal_rate": i["refusal_rate"],
                "refusal_rate_delta": i["refusal_rate"] - b["refusal_rate"],
            }
        dump_json(output_dir / "comparison_summary.json", comparison)
        print(json.dumps({"comparison": comparison}, indent=2), flush=True)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
