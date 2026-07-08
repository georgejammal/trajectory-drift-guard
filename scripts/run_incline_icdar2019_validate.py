#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch

from parsing_neurons_repro.generation import decode_new_tokens, model_device, move_to_device, prepare_vlm_inputs
from parsing_neurons_repro.incline import INCLINEIntervention, ensure_flores_bridge
from parsing_neurons_repro.io import clean_float, parse_csv, parse_float_grid, read_json, slug_parts, window_layers, write_json
from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_vlm, model_spec
from parsing_neurons_repro.tasks.icdar2019 import (
    DEFAULT_PROMPT,
    normalized_edit_similarity,
    load_language_samples,
    sample_to_json,
)


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


def parse_layer_window(raw: str, model_alias: str) -> str:
    return infer_all_layers(model_alias) if raw.strip().lower() == "all" else raw


def bridge_path(args: argparse.Namespace, model_alias: str, language: str, stats_layers: str) -> Path:
    return (
        args.output_root
        / "bridges"
        / model_alias
        / language.lower()
        / slug_parts(
            "flores-incline",
            "layers" + stats_layers,
            "n" + str(args.bridge_samples),
            args.bridge_token_scope,
            "maxlen" + str(args.bridge_max_length),
            "ridge" + clean_float(args.ridge),
        )
        / "bridge.pt"
    )


def batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_alpha(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    samples: list[Any],
    bridge: Path,
    window: str,
    alpha: float,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    qwen_max_pixels: int | None,
    prompt: str,
    resume: bool,
) -> dict[str, Any]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_score = 0.0
    scored = 0
    context = INCLINEIntervention(
        model,
        bridge,
        window,
        sigma=alpha,
        direction_sign=1.0,
        token_scope="last_position",
    )
    with context:
        for batch_idx, batch in enumerate(batches(samples, batch_size), start=1):
            pending = []
            for sample in batch:
                out_path = prediction_dir / f"{sample.image_id}.json"
                if resume and out_path.exists():
                    rows.append(read_json(out_path))
                else:
                    pending.append(sample)
            if not pending:
                continue
            inputs = prepare_vlm_inputs(
                model_alias=model_alias,
                processor=processor,
                images=[sample.image_path for sample in pending],
                questions=[prompt for _ in pending],
                qwen_max_pixels=qwen_max_pixels,
            )
            inputs = move_to_device(inputs, model_device(model))
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            predictions = decode_new_tokens(
                model_alias=model_alias,
                processor=processor,
                generated=generated,
                inputs=inputs,
            )
            for sample, prediction in zip(pending, predictions):
                score = normalized_edit_similarity(prediction, sample.gold_text)
                record = {
                    **sample_to_json(sample),
                    "prediction": prediction,
                    "score": score,
                }
                write_json(prediction_dir / f"{sample.image_id}.json", record)
                rows.append(record)
            print(
                f"[incline-icdar:{model_alias}:{pending[0].language}:alpha{clean_float(alpha)}] "
                f"batch={batch_idx} done={len(rows)}/{len(samples)}",
                flush=True,
            )
    for row in rows:
        if row:
            total_score += float(row["score"])
            scored += 1
    summary = {
        "model_alias": model_alias,
        "language": samples[0].language if samples else None,
        "alpha": alpha,
        "num_samples": len(samples),
        "num_scored": scored,
        "mean_normalized_edit_similarity": total_score / max(1, scored),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune INCLINE alpha on ICDAR2019 MLT validation samples.")
    parser.add_argument("--models", default="gemma3_4b_it")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean")
    parser.add_argument("--icdar-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data")) / "icdar2019_mlt")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs/incline")))
    parser.add_argument(
        "--flores-root",
        type=Path,
        default=Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"))
        / "flores_transfer_pairs"
        / "flores101_en_to_cc_ocr_languages_random500"
        / "pairs_by_language",
    )
    parser.add_argument("--stats-layers", default="all")
    parser.add_argument("--intervention-window", default="all")
    parser.add_argument("--alphas", default="-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--samples-per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--script-only-gold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bridge-samples", type=int, default=500)
    parser.add_argument("--bridge-batch-size", type=int, default=8)
    parser.add_argument("--bridge-max-length", type=int, default=512)
    parser.add_argument("--bridge-token-scope", choices=["all_nonpad", "last_nonpad"], default="all_nonpad")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--force-recompute-bridge", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    models = parse_csv(args.models)
    languages = parse_csv(args.languages)
    alphas = parse_float_grid(args.alphas)
    all_selected: dict[str, dict[str, Any]] = {}

    for model_alias in models:
        stats_layers = parse_layer_window(args.stats_layers, model_alias)
        window = parse_layer_window(args.intervention_window, model_alias)
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Processor for {model_alias} does not expose a tokenizer.")
        model = load_vlm(model_alias, spec.path)
        model_selected: dict[str, Any] = {}
        for language in languages:
            samples = load_language_samples(
                root=args.icdar_root,
                language=language,
                limit=args.samples_per_language,
                seed=args.seed,
                script_only_gold=args.script_only_gold,
            )
            if len(samples) < args.samples_per_language:
                print(f"[incline-icdar] warning: {language} has only {len(samples)} samples", flush=True)
            bridge = ensure_flores_bridge(
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
            summaries = []
            lang_root = args.output_root / "runs" / "icdar2019_incline_validation" / model_alias / language.lower()
            for alpha in alphas:
                run_name = slug_parts("incline", "layers" + window, "alpha" + clean_float(alpha))
                summaries.append(
                    run_alpha(
                        model_alias=model_alias,
                        processor=processor,
                        model=model,
                        samples=samples,
                        bridge=bridge,
                        window=window,
                        alpha=alpha,
                        output_dir=lang_root / run_name,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        qwen_max_pixels=args.qwen_max_pixels if "qwen" in model_alias else None,
                        prompt=args.prompt,
                        resume=args.resume,
                    )
                )
                write_json(lang_root / "alpha_sweep_summary.json", summaries)
            best = max(summaries, key=lambda row: row["mean_normalized_edit_similarity"])
            model_selected[language] = best
            write_json(lang_root / "selected_alpha.json", best)
        if model_selected:
            mean_alpha = sum(float(row["alpha"]) for row in model_selected.values()) / len(model_selected)
            rounded_mean_alpha = round(mean_alpha, 1)
            model_selected["Russian"] = {
                "language": "Russian",
                "alpha": rounded_mean_alpha,
                "source": "mean_alpha_from_icdar2019_available_languages",
                "available_languages": sorted(model_selected),
            }
        all_selected[model_alias] = model_selected
        write_json(args.output_root / "runs" / "icdar2019_incline_validation" / "selected_alphas.json", all_selected)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
