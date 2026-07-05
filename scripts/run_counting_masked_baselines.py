#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOCAL_DATA_ROOT = Path("/home/georgejammal/projects/parsing_neurons/data")
if "PARSING_NEURONS_DATA_ROOT" not in os.environ and LOCAL_DATA_ROOT.exists():
    os.environ["PARSING_NEURONS_DATA_ROOT"] = str(LOCAL_DATA_ROOT)

from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_tokenizer, load_vlm, model_spec  # noqa: E402
from parsing_neurons_repro.tasks.counting import DEFAULT_COUNTING_DATASETS, evaluate_counting_suite  # noqa: E402


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen2_5_vl_3b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}

NUMBER_WORDS_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 300,
    "gemma3_12b_it": 100,
    "qwen2_5_vl_3b_instruct": 64,
    "qwen3_vl_8b_instruct": 64,
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer(text, add_special_tokens=False)["input_ids"]]


def single_token_variants(tokenizer: Any, values: list[str]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = {}
    for value in values:
        ids: set[int] = set()
        variants = [value, " " + value]
        for variant in variants:
            variant_ids = token_ids(tokenizer, variant)
            if len(variant_ids) == 1:
                ids.add(variant_ids[0])
        if not ids:
            # Fall back to all sub-token ids for unusual tokenizers. This keeps the
            # mask usable, but all models used here have single-token variants.
            for variant in variants:
                ids.update(token_ids(tokenizer, variant))
        rows[value] = sorted(ids)
    return rows


def allowed_ids_for_constraint(tokenizer: Any, constraint: str) -> tuple[list[int], dict[str, Any]]:
    if constraint == "digit_mask":
        variants = single_token_variants(tokenizer, [str(i) for i in range(10)])
    elif constraint == "word_mask":
        variants = single_token_variants(tokenizer, NUMBER_WORDS_EN)
    else:
        raise ValueError(f"Unknown constraint: {constraint}")

    allowed: set[int] = set()
    for ids in variants.values():
        allowed.update(ids)
    return sorted(allowed), {
        "constraint": constraint,
        "variant_token_ids": variants,
        "allowed_token_ids": sorted(allowed),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="gemma3_4b_it,gemma3_12b_it,qwen2_5_vl_3b_instruct,qwen3_vl_8b_instruct",
    )
    parser.add_argument("--constraints", default="digit_mask,word_mask")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--qwen-max-pixels", type=int, default=401408)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    constraints = [item.strip() for item in args.constraints.split(",") if item.strip()]
    summaries = []

    for model_alias in models:
        spec = model_spec(model_alias)
        print(f"[masked-counting] loading {model_alias} from {spec.path}", flush=True)
        tokenizer = load_tokenizer(spec.path)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        batch_size = DEFAULT_BATCH_SIZES.get(model_alias, 64)

        for constraint in constraints:
            allowed_ids, metadata = allowed_ids_for_constraint(tokenizer, constraint)
            output_dir = args.output_root / "runs" / "counting_masked" / model_alias / constraint
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "constraint_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"[masked-counting:{model_alias}:{constraint}] allowed={len(allowed_ids)} "
                f"batch_size={batch_size}",
                flush=True,
            )
            summary = evaluate_counting_suite(
                model_alias=model_alias,
                processor=processor,
                model=model,
                output_dir=output_dir,
                component_mode="baseline",
                mlp_selection=None,
                attn_selection=None,
                datasets=DEFAULT_COUNTING_DATASETS,
                batch_size=batch_size,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
                qwen_max_pixels=args.qwen_max_pixels,
                allowed_token_ids=allowed_ids,
                decoding_constraint=constraint,
            )
            summary["decoding_constraint"] = constraint
            summary["allowed_token_count"] = len(allowed_ids)
            summaries.append({"model_alias": model_alias, "constraint": constraint, "summary": summary})

        del model, processor, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = args.output_root / "runs" / "counting_masked" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
