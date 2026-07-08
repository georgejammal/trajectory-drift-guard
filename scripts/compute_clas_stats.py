#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from parsing_neurons_repro.clas import compute_clas_stats, load_parallel_flores_texts, load_parallel_xquad_texts
from parsing_neurons_repro.io import clean_float, parse_csv, slug_parts
from parsing_neurons_repro.models import load_tokenizer, load_vlm, model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLAS neuron-category statistics.")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-root", default=os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs"))
    parser.add_argument("--stats-source", choices=["flores", "xquad"], default="flores")
    parser.add_argument("--stats-languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument(
        "--flores-root",
        default=os.path.join(
            os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"),
            "flores_transfer_pairs",
            "flores101_en_to_cc_ocr_languages_random500",
            "pairs_by_language",
        ),
    )
    parser.add_argument("--xquad-dataset", default="google/xquad")
    parser.add_argument("--xquad-split", default="validation")
    parser.add_argument("--stats-layers", required=True)
    parser.add_argument("--stats-samples", type=int, default=100)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--stats-max-length", type=int, default=512)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--stats-token-scope", choices=["last_nonpad", "all_nonpad"], default="last_nonpad")
    parser.add_argument("--stats-statistic", choices=["mean", "mean_abs"], default="mean")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stats_path(args: argparse.Namespace, languages: list[str]) -> Path:
    return (
        Path(args.output_root)
        / "clas"
        / "stats"
        / args.model_alias
        / slug_parts(
            args.stats_source,
            "langs-" + "-".join(language.lower() for language in languages),
            "layers" + args.stats_layers,
            "tau" + clean_float(args.activation_threshold),
            args.stats_token_scope,
            args.stats_statistic,
            "n" + str(args.stats_samples),
        )
        / "stats.json"
    )


def main() -> None:
    args = parse_args()
    target_languages = parse_csv(args.stats_languages)
    path_languages = target_languages
    stats_languages = ["English", *target_languages] if args.stats_source == "flores" else target_languages
    path = stats_path(args, path_languages)
    if path.exists() and not args.force:
        print(path)
        return

    spec = model_spec(args.model_alias)
    tokenizer = load_tokenizer(spec.path)
    model = load_vlm(args.model_alias, spec.path)
    if args.stats_source == "flores":
        texts_by_language = load_parallel_flores_texts(
            flores_root=Path(args.flores_root).resolve(),
            languages=target_languages,
            max_samples=args.stats_samples,
            anchor_language="English",
        )
    else:
        texts_by_language = load_parallel_xquad_texts(
            languages=stats_languages,
            max_samples=args.stats_samples,
            dataset_name=args.xquad_dataset,
            split=args.xquad_split,
        )
    compute_clas_stats(
        model=model,
        tokenizer=tokenizer,
        texts_by_language=texts_by_language,
        layers=args.stats_layers,
        output_path=path,
        batch_size=args.stats_batch_size,
        max_length=args.stats_max_length,
        activation_threshold=args.activation_threshold,
        token_scope=args.stats_token_scope,
        statistic=args.stats_statistic,
    )
    print(path)


if __name__ == "__main__":
    main()
