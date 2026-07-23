#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from parsing_neurons_repro.models import load_processor, load_vlm, model_spec
from parsing_neurons_repro.tasks.counting import DEFAULT_COUNTING_DATASETS, evaluate_counting_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed three-dataset counting baseline for one model.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    spec = model_spec(args.model_alias)
    processor = load_processor(spec.path)
    model = load_vlm(args.model_alias, spec.path, dtype="bfloat16")
    output_root = args.output_root / args.model_alias
    metrics = []
    for dataset_name, dataset_path in DEFAULT_COUNTING_DATASETS.items():
        metrics.append(
            evaluate_counting_dataset(
                model_alias=args.model_alias,
                processor=processor,
                model=model,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                output_dir=output_root,
                component_mode="baseline",
                mlp_selection=None,
                attn_selection=None,
                batch_size=args.batch_size,
                max_new_tokens=16,
                qwen_max_pixels=401408 if "qwen" in args.model_alias else None,
                adaptive_oom_split=False,
            )
        )
    print({"model_alias": args.model_alias, "datasets": metrics}, flush=True)


if __name__ == "__main__":
    main()
