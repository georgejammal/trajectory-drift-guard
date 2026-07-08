#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from parsing_neurons_repro.models import MODEL_PATHS
from parsing_neurons_repro.search import run_search_from_args


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/299b6fef6178f0c84214043f933b85cde85bbcf1",
    "qwen2_5_vl_3b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/6628550ee45e342cc5f1692f146b0e0eef2b4d80",
    "qwen3_vl_8b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/9ba50c9f45e55e58ce5108ed88e4273fb7d937b7",
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run static parsing-neuron searches for counting and CC-OCR."
    )
    parser.add_argument("--task", choices=["counting", "ccocr"], required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-root", default=os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs"))
    parser.add_argument(
        "--windows",
        required=True,
        help="Comma-separated zero-based inclusive windows, e.g. '17-27,17-25'.",
    )
    parser.add_argument(
        "--sigmas",
        default="2.5,3,3.5,4,4.5,5",
        help="Comma-separated right-tail sigma thresholds under H0.",
    )
    parser.add_argument(
        "--component-modes",
        default="mlp,mlp_attn",
        help="Comma-separated modes: baseline, mlp, attn, mlp_attn. Baseline is controlled separately by --run-baseline.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--direction",
        choices=["word-minus-digit", "digit-minus-word"],
        default="word-minus-digit",
        help="Counting direction only.",
    )
    parser.add_argument(
        "--counting-language",
        choices=["en", "zh", "pooled_en_zh"],
        default="en",
        help="Counting direction language pool.",
    )
    parser.add_argument("--counting-digits", default="0,1,2,3,4,5,6,7,8,9")

    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian", help="CC-OCR languages.")
    parser.add_argument(
        "--direction-source",
        choices=["flores", "news_commentary"],
        default="flores",
        help="CC-OCR direction-estimation corpus.",
    )
    parser.add_argument(
        "--flores-root",
        default=os.path.join(
            os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"),
            "flores_transfer_pairs",
            "flores101_en_to_cc_ocr_languages_random500",
            "pairs_by_language",
        ),
        help="Directory with en_to_<language>.json files.",
    )
    parser.add_argument(
        "--ncwm-root",
        default=os.path.join(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"), "ncwm"),
        help="News Commentary / NCWM root with en-<lang>/train.en and train.<lang> files.",
    )
    parser.add_argument(
        "--max-gold-tokens-exclusive",
        type=int,
        default=None,
        help="CC-OCR optional filter matching previous controlled runs.",
    )
    return parser.parse_args()


def main() -> None:
    resolve_local_model_paths()
    run_search_from_args(parse_args())


if __name__ == "__main__":
    main()
