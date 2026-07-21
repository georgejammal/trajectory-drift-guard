#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
sys.path.insert(0, str(src_root))

from parsing_neurons_repro.models import load_processor, load_vlm  # noqa: E402
from parsing_neurons_repro.tasks.counting import evaluate_counting_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="countbenchqa")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_alias = "qwen2_5_vl_3b_instruct"
    model_path = Path(os.environ["QWEN25VL_3B_INSTRUCT_PATH"])
    dataset_name = args.dataset_name
    dataset_path = Path(os.environ["PARSING_NEURONS_DATA_ROOT"]) / dataset_name

    print(f"node: {os.uname().nodename}", flush=True)
    print(f"gpu visibility: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print(f"batch size: {args.batch_size}", flush=True)

    processor = load_processor(model_path)
    model = load_vlm(model_alias, model_path, dtype="bfloat16")
    metrics = evaluate_counting_dataset(
        model_alias=model_alias,
        processor=processor,
        model=model,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
        component_mode="baseline",
        mlp_selection=None,
        attn_selection=None,
        batch_size=args.batch_size,
        max_new_tokens=16,
        qwen_max_pixels=401408,
        adaptive_oom_split=False,
    )
    print(f"RESULT: {metrics}", flush=True)
    if torch.cuda.is_available():
        print(
            f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; "
            f"peak reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
