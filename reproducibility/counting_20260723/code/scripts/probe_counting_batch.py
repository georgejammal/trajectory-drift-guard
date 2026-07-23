#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from parsing_neurons_repro.generation import generate_batch
from parsing_neurons_repro.models import load_processor, load_vlm, model_spec
from parsing_neurons_repro.tasks.counting import (
    DEFAULT_COUNTING_DATASETS,
    load_filtered_dataset,
    row_image,
    row_question,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_COUNTING_DATASETS["countbenchqa"]))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [int(value) for value in args.candidates.split(",")]
    if candidates != sorted(candidates) or not candidates:
        raise ValueError("Candidates must be a non-empty ascending list.")

    dataset, _ = load_filtered_dataset(
        args.dataset_path,
        split="test",
        gold_numbers=list(range(1, 10)),
        limit=max(candidates),
    )
    spec = model_spec(args.model_alias)
    processor = load_processor(spec.path)
    model = load_vlm(args.model_alias, spec.path)
    suffix = " Answer with one number do not add any further explanations."
    successful: list[int] = []

    for batch_size in candidates:
        rows = list(dataset.select(range(batch_size)))
        print(f"[batch-probe:{args.model_alias}] testing batch_size={batch_size}", flush=True)
        try:
            generate_batch(
                model_alias=args.model_alias,
                processor=processor,
                model=model,
                images=[row_image(row) for row in rows],
                questions=[row_question(row) for row in rows],
                suffix=suffix,
                max_new_tokens=args.max_new_tokens,
                qwen_max_pixels=args.qwen_max_pixels,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[batch-probe:{args.model_alias}] OOM batch_size={batch_size}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            break
        successful.append(batch_size)
        print(f"[batch-probe:{args.model_alias}] success batch_size={batch_size}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not successful:
        raise RuntimeError("No candidate batch size succeeded.")
    result = {
        "model_alias": args.model_alias,
        "candidates": candidates,
        "successful": successful,
        "selected_batch_size": successful[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
