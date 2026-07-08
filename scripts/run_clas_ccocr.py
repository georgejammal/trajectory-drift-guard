#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from parsing_neurons_repro.io import clean_float, parse_csv, parse_float_grid, slug_parts

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CLAS-style inference-time baseline on CC-OCR."
    )
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--output-root", default=os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs"))
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument(
        "--stats-source",
        choices=["xquad", "flores"],
        default="xquad",
        help="Parallel data source used to estimate CLAS categories.",
    )
    parser.add_argument(
        "--stats-languages",
        default=None,
        help="Languages used to estimate CLAS categories. Defaults to CLAS/XQuAD languages for xquad.",
    )
    parser.add_argument("--xquad-dataset", default="google/xquad")
    parser.add_argument("--xquad-split", default="validation")
    parser.add_argument(
        "--flores-root",
        default=os.path.join(
            os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"),
            "flores_transfer_pairs",
            "flores101_en_to_cc_ocr_languages_random500",
            "pairs_by_language",
        ),
    )
    parser.add_argument(
        "--stats-layers",
        required=True,
        help="Zero-based inclusive layer window for category estimation, e.g. 17-33.",
    )
    parser.add_argument(
        "--intervention-windows",
        required=True,
        help="Comma-separated zero-based inclusive CLAS intervention windows.",
    )
    parser.add_argument(
        "--alphas",
        default="-5,-3,-1,1,3,5",
        help="CLAS does not enumerate this grid in the paper source; this default includes their documented alpha=-5 example.",
    )
    parser.add_argument("--betas", default="0.2,0.4,0.6")
    parser.add_argument("--gammas", default="0.1,0.2,0.4")
    parser.add_argument("--specific-scope", choices=["all", "target"], default="all")
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--stats-token-scope", choices=["last_nonpad", "all_nonpad"], default="last_nonpad")
    parser.add_argument("--stats-statistic", choices=["mean", "mean_abs"], default="mean")
    parser.add_argument("--stats-samples", type=int, default=100)
    parser.add_argument("--stats-batch-size", type=int, default=8)
    parser.add_argument("--stats-max-length", type=int, default=512)
    parser.add_argument("--force-recompute-stats", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-gold-tokens-exclusive", type=int, default=500)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def stats_path(args: argparse.Namespace, output_root: Path, languages: list[str]) -> Path:
    return (
        output_root
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


def ensure_stats(
    *,
    args: argparse.Namespace,
    model: Any,
    tokenizer: "PreTrainedTokenizerBase",
    output_root: Path,
    stats_languages: list[str],
) -> Path:
    from parsing_neurons_repro.clas import compute_clas_stats, load_parallel_flores_texts, load_parallel_xquad_texts

    path = stats_path(args, output_root, stats_languages)
    if path.exists() and not args.force_recompute_stats:
        return path
    if args.stats_source == "xquad":
        texts_by_language = load_parallel_xquad_texts(
            languages=stats_languages,
            max_samples=args.stats_samples,
            dataset_name=args.xquad_dataset,
            split=args.xquad_split,
        )
    else:
        texts_by_language = load_parallel_flores_texts(
            flores_root=Path(args.flores_root).resolve(),
            languages=stats_languages,
            max_samples=args.stats_samples,
            anchor_language="English",
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
    return path


def main() -> None:
    args = parse_args()
    from parsing_neurons_repro.interventions import CLASIntervention
    from parsing_neurons_repro.io import write_json
    from parsing_neurons_repro.models import load_processor, load_tokenizer, load_vlm, model_spec
    from parsing_neurons_repro.tasks.ccocr import DEFAULT_CC_OCR_INDEX, DEFAULT_CC_OCR_ROOT, evaluate_ccocr_suite

    output_root = Path(args.output_root).resolve()
    languages = parse_csv(args.languages)
    default_xquad_languages = "English,Arabic,German,Greek,Spanish,Hindi,Romanian,Russian,Thai,Turkish,Vietnamese,Chinese"
    if args.stats_languages:
        stats_languages = parse_csv(args.stats_languages)
    elif args.stats_source == "xquad":
        stats_languages = parse_csv(default_xquad_languages)
    else:
        stats_languages = languages
    alphas = parse_float_grid(args.alphas)
    betas = parse_float_grid(args.betas)
    gammas = parse_float_grid(args.gammas)
    windows = parse_csv(args.intervention_windows)

    spec = model_spec(args.model_alias)
    tokenizer = load_tokenizer(spec.path)
    processor = load_processor(spec.path)
    model = load_vlm(args.model_alias, spec.path)
    clas_stats_path = ensure_stats(
        args=args,
        model=model,
        tokenizer=tokenizer,
        output_root=output_root,
        stats_languages=stats_languages,
    )

    task_root = output_root / "runs" / "ccocr_clas" / args.model_alias
    summaries: list[dict[str, Any]] = []

    if args.run_baseline:
        summaries.append(
            evaluate_ccocr_suite(
                model_alias=args.model_alias,
                processor=processor,
                model=model,
                output_dir=task_root / "baseline",
                component_mode="baseline",
                mlp_selection=None,
                attn_selection=None,
                cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                index_path=DEFAULT_CC_OCR_INDEX,
                languages=languages,
                batch_size=args.batch_size,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
                max_gold_tokens_exclusive=args.max_gold_tokens_exclusive,
                qwen_max_pixels=args.qwen_max_pixels,
            )
        )

    for language in languages:
        for window in windows:
            for alpha in alphas:
                for beta in betas:
                    for gamma in gammas:
                        run_name = slug_parts(
                            "clas",
                            "layers" + window,
                            "a" + clean_float(alpha),
                            "b" + clean_float(beta),
                            "g" + clean_float(gamma),
                            args.specific_scope,
                        )
                        context = CLASIntervention(
                            model,
                            clas_stats_path,
                            window,
                            target_language=language,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            token_scope="all_positions",
                            specific_scope=args.specific_scope,
                        )
                        summary = evaluate_ccocr_suite(
                            model_alias=args.model_alias,
                            processor=processor,
                            model=model,
                            output_dir=task_root / language.lower() / run_name,
                            component_mode="clas",
                            mlp_selection=None,
                            attn_selection=None,
                            cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                            index_path=DEFAULT_CC_OCR_INDEX,
                            languages=[language],
                            batch_size=args.batch_size,
                            limit=args.limit,
                            max_new_tokens=args.max_new_tokens,
                            max_gold_tokens_exclusive=args.max_gold_tokens_exclusive,
                            qwen_max_pixels=args.qwen_max_pixels,
                            scalar_mode=run_name,
                            custom_context=context,
                        )
                        summary["clas"] = {
                            "stats_path": str(clas_stats_path),
                            "language": language,
                            "window": window,
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "specific_scope": args.specific_scope,
                        }
                        summaries.append(summary)
                        write_json(task_root / "clas_search_summary.json", summaries)

    write_json(task_root / "clas_search_summary.json", summaries)


if __name__ == "__main__":
    main()
