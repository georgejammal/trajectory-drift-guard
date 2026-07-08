#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path
from typing import Any

import torch

from parsing_neurons_repro.incline import INCLINEIntervention, ensure_flores_bridge, ensure_ncwm_bridge
from parsing_neurons_repro.io import clean_float, parse_csv, parse_float_grid, read_json, slug_parts, window_layers, write_json
from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_vlm, model_spec
from parsing_neurons_repro.tasks.ccocr import DEFAULT_CC_OCR_INDEX, DEFAULT_CC_OCR_ROOT, evaluate_ccocr_suite


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen2_5_vl_3b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def infer_all_layers(alias: str) -> str:
    spec = model_spec(alias)
    return f"0-{spec.num_layers - 1}"


def language_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", language.strip().lower()).strip("_")


def parse_layer_windows(raw: str, model_alias: str) -> list[str]:
    if raw.strip().lower() == "all":
        return [infer_all_layers(model_alias)]
    windows = []
    for item in parse_csv(raw):
        windows.append(infer_all_layers(model_alias) if item.lower() == "all" else item)
    return windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an INCLINE-style cross-lingual MLP-output intervention on CC-OCR."
    )
    parser.add_argument("--models", default="gemma3_4b_it")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs/incline")))
    parser.add_argument(
        "--flores-root",
        type=Path,
        default=Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"))
        / "flores_transfer_pairs"
        / "flores101_en_to_cc_ocr_languages_random500"
        / "pairs_by_language",
    )
    parser.add_argument("--bridge-source", choices=["news_commentary", "flores"], default="news_commentary")
    parser.add_argument(
        "--ncwm-root",
        type=Path,
        default=Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data")) / "ncwm",
        help="News Commentary bridge root in the INCLINE data/ncwm layout.",
    )
    parser.add_argument(
        "--stats-layers",
        default="all",
        help="Layers used to fit INCLINE bridge factors. Use 'all' to match the paper's all-layer intervention.",
    )
    parser.add_argument(
        "--intervention-windows",
        default="all",
        help="Comma-separated inclusive layer windows to patch, or 'all'.",
    )
    parser.add_argument(
        "--sigmas",
        default="-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1",
        help="INCLINE sigma grid. The released code sweeps approximately [-1, 1] in 0.1 steps.",
    )
    parser.add_argument(
        "--direction-signs",
        default="1",
        help="Sign multiplier for hW before applying sigma. Use '1,-1' to test both released-script conventions.",
    )
    parser.add_argument(
        "--selected-alphas-json",
        type=Path,
        default=None,
        help="Optional selected-alpha file from ICDAR2019 validation. When set, only the selected alpha per model/language is used.",
    )
    parser.add_argument("--bridge-samples", type=int, default=500)
    parser.add_argument("--bridge-batch-size", type=int, default=4)
    parser.add_argument("--bridge-max-length", type=int, default=512)
    parser.add_argument("--bridge-token-scope", choices=["all_nonpad", "last_nonpad"], default="all_nonpad")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--force-recompute-bridge", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-gold-tokens-exclusive", type=int, default=500)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def selected_alpha(
    selected_payload: dict[str, Any] | None,
    *,
    model_alias: str,
    language: str,
) -> float | None:
    if selected_payload is None:
        return None
    row = selected_payload.get(model_alias, {}).get(language)
    if row is None:
        return None
    if isinstance(row, dict):
        return float(row["alpha"])
    return float(row)


def bridge_path(args: argparse.Namespace, model_alias: str, language: str, stats_layers: str) -> Path:
    return (
        args.output_root
        / "bridges"
        / model_alias
        / language_slug(language)
        / slug_parts(
            args.bridge_source,
            "layers" + stats_layers,
            "n" + str(args.bridge_samples),
            args.bridge_token_scope,
            "maxlen" + str(args.bridge_max_length),
            "ridge" + clean_float(args.ridge),
        )
        / "bridge.pt"
    )


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    models = parse_csv(args.models)
    languages = parse_csv(args.languages)
    sigmas = parse_float_grid(args.sigmas)
    direction_signs = parse_float_grid(args.direction_signs)
    selected_payload = read_json(args.selected_alphas_json) if args.selected_alphas_json is not None else None
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for model_alias in models:
        stats_layers = parse_layer_windows(args.stats_layers, model_alias)[0]
        intervention_windows = parse_layer_windows(args.intervention_windows, model_alias)
        spec = model_spec(model_alias)
        print(f"[incline] loading {model_alias} from {spec.path}", flush=True)
        processor = load_processor(spec.path)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Processor for {model_alias} does not expose a tokenizer.")
        model = load_vlm(model_alias, spec.path)

        model_root = args.output_root / "runs" / "ccocr_incline" / model_alias
        if args.run_baseline:
            summaries.append(
                evaluate_ccocr_suite(
                    model_alias=model_alias,
                    processor=processor,
                    model=model,
                    output_dir=model_root / "baseline",
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
                    resume=args.resume,
                )
            )

        for language in languages:
            if args.bridge_source == "news_commentary":
                current_bridge = ensure_ncwm_bridge(
                    model_alias=model_alias,
                    model=model,
                    tokenizer=tokenizer,
                    language=language,
                    layers=window_layers(stats_layers),
                    output_path=bridge_path(args, model_alias, language, stats_layers),
                    ncwm_root=args.ncwm_root.resolve(),
                    max_samples=args.bridge_samples,
                    batch_size=args.bridge_batch_size,
                    max_length=args.bridge_max_length,
                    token_scope=args.bridge_token_scope,
                    ridge=args.ridge,
                    force=args.force_recompute_bridge,
                )
            else:
                current_bridge = ensure_flores_bridge(
                    model_alias=model_alias,
                    model=model,
                    tokenizer=tokenizer,
                    language=language,
                    layers=window_layers(stats_layers),
                    output_path=bridge_path(args, model_alias, language, stats_layers),
                    flores_root=args.flores_root.resolve(),
                    max_samples=args.bridge_samples,
                    batch_size=args.bridge_batch_size,
                    max_length=args.bridge_max_length,
                    token_scope=args.bridge_token_scope,
                    ridge=args.ridge,
                    force=args.force_recompute_bridge,
                )
            for window in intervention_windows:
                for sign in direction_signs:
                    language_alpha = selected_alpha(selected_payload, model_alias=model_alias, language=language)
                    active_sigmas = [language_alpha] if language_alpha is not None else sigmas
                    for sigma in active_sigmas:
                        run_name = slug_parts(
                            "incline",
                            "stats" + stats_layers,
                            "layers" + window,
                            "sign" + clean_float(sign),
                            "alpha" + clean_float(sigma),
                        )
                        context = INCLINEIntervention(
                            model,
                            current_bridge,
                            window,
                            sigma=sigma,
                            direction_sign=sign,
                            token_scope="last_position",
                        )
                        summary = evaluate_ccocr_suite(
                            model_alias=model_alias,
                            processor=processor,
                            model=model,
                            output_dir=model_root / language_slug(language) / run_name,
                            component_mode="incline",
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
                            resume=args.resume,
                        )
                        summary["incline"] = {
                            "bridge_path": str(current_bridge),
                            "bridge_source": args.bridge_source,
                            "stats_layers": stats_layers,
                            "window": window,
                            "alpha": sigma,
                            "direction_sign": sign,
                            "bridge_samples": args.bridge_samples,
                            "bridge_token_scope": args.bridge_token_scope,
                            "ridge": args.ridge,
                        }
                        summaries.append(summary)
                        write_json(model_root / "incline_search_summary.json", summaries)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    write_json(args.output_root / "runs" / "ccocr_incline_summary.json", summaries)


if __name__ == "__main__":
    main()
