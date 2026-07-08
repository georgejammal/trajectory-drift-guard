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
from parsing_neurons_repro.io import clean_float, parse_csv, read_json, slug_parts, window_layers, write_json
from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_vlm, model_spec

from run_mdpbench_ccocr_top3 import (  # noqa: E402
    DEFAULT_PROMPT,
    add_caps,
    batches,
    evaluate_official,
    language_samples,
    make_filtered_annotation,
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


def language_slug(language: str) -> str:
    return language.strip().lower()


def selected_alpha(payload: dict[str, Any], model_alias: str, language: str) -> float:
    row = payload[model_alias][language]
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
            "flores-incline",
            "layers" + stats_layers,
            "n" + str(args.bridge_samples),
            args.bridge_token_scope,
            "maxlen" + str(args.bridge_max_length),
            "ridge" + clean_float(args.ridge),
        )
        / "bridge.pt"
    )


def generate_incline(
    *,
    model_alias: str,
    language: str,
    alpha: float,
    bridge: Path,
    window: str,
    samples: list[dict[str, Any]],
    processor: Any,
    model: Any,
    run_dir: Path,
    batch_size: int,
    prompt: str,
    qwen_max_pixels: int | None,
    resume: bool,
) -> dict[str, Any]:
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    queued = []
    for sample in samples:
        output_path = prediction_dir / f"{sample['stem']}.md"
        if resume and output_path.exists():
            continue
        queued.append({**sample, "output_path": output_path})

    context = INCLINEIntervention(
        model,
        bridge,
        window,
        sigma=alpha,
        direction_sign=1.0,
        token_scope="last_position",
    )
    written = 0
    with context:
        for batch_idx, batch in enumerate(batches(queued, batch_size), start=1):
            max_new_tokens = max(sample["gt_token_cap"] for sample in batch)
            inputs = prepare_vlm_inputs(
                model_alias=model_alias,
                processor=processor,
                images=[sample["image_path"] for sample in batch],
                questions=[prompt for _ in batch],
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
            for sample, prediction in zip(batch, predictions):
                if not prediction.endswith("\n"):
                    prediction += "\n"
                sample["output_path"].write_text(prediction, encoding="utf-8")
                written += 1
            print(
                f"[mdpbench-incline:{model_alias}:{language}:alpha{clean_float(alpha)}] "
                f"batch={batch_idx} written={written}/{len(queued)} batch_cap={max_new_tokens}",
                flush=True,
            )
    summary = {
        "task": "mdpbench",
        "method": "incline",
        "model_alias": model_alias,
        "language": language,
        "alpha": alpha,
        "window": window,
        "bridge_path": str(bridge),
        "prediction_dir": str(prediction_dir),
        "num_selected_samples": len(samples),
        "queued": len(queued),
        "written_this_run": written,
    }
    write_json(run_dir / "generation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ICDAR-selected INCLINE alphas on MDPBench.")
    parser.add_argument("--models", default="gemma3_4b_it")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument("--selected-alphas-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs/incline")))
    parser.add_argument("--mdpbench-root", type=Path, default=Path("external/MultimodalOCR/MDPBench"))
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
    parser.add_argument("--bridge-samples", type=int, default=500)
    parser.add_argument("--bridge-batch-size", type=int, default=8)
    parser.add_argument("--bridge-max-length", type=int, default=512)
    parser.add_argument("--bridge-token-scope", choices=["all_nonpad", "last_nonpad"], default="all_nonpad")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gt-token-cap-multiplier", type=float, default=1.2)
    parser.add_argument("--max-gt-token-cap", type=int, default=1600)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--python", default="/home/georgejammal/projects/a100env/bin/python")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    selected = read_json(args.selected_alphas_json)
    models = parse_csv(args.models)
    languages = parse_csv(args.languages)
    mdpbench_root = args.mdpbench_root.resolve()
    annotation_path = mdpbench_root / "dataset" / "MDPBench_public.json"
    image_dir = mdpbench_root / "dataset" / "MDPBench_img_public"
    summaries: list[dict[str, Any]] = []
    eval_summaries: list[dict[str, Any]] = []

    for model_alias in models:
        stats_layers = parse_layer_window(args.stats_layers, model_alias)
        window = parse_layer_window(args.intervention_window, model_alias)
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Processor for {model_alias} does not expose a tokenizer.")
        model = load_vlm(model_alias, spec.path)
        for language in languages:
            alpha = selected_alpha(selected, model_alias, language)
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
            )
            all_samples = language_samples(annotation_path, image_dir, language)
            selected_samples = add_caps(
                all_samples,
                processor,
                multiplier=args.gt_token_cap_multiplier,
                max_cap=args.max_gt_token_cap,
            )
            lang_slug = language_slug(language)
            run_name = slug_parts("incline", "layers" + window, "alpha" + clean_float(alpha))
            model_root = args.output_root / "runs" / "mdpbench_incline" / model_alias / f"{lang_slug}_icdar_alpha"
            filtered_annotation = model_root / "annotations" / f"MDPBench_public_{lang_slug}_cap_le1600.json"
            make_filtered_annotation(selected_samples, filtered_annotation)
            summary = generate_incline(
                model_alias=model_alias,
                language=language,
                alpha=alpha,
                bridge=bridge,
                window=window,
                samples=selected_samples,
                processor=processor,
                model=model,
                run_dir=model_root / run_name,
                batch_size=args.batch_size,
                prompt=args.prompt,
                qwen_max_pixels=args.qwen_max_pixels if "qwen" in model_alias else None,
                resume=args.resume,
            )
            summaries.append(summary)
            write_json(args.output_root / "runs" / "mdpbench_incline" / "generation_summaries.json", summaries)
            if args.evaluate:
                eval_meta = evaluate_official(
                    official_root=mdpbench_root,
                    annotation_path=filtered_annotation,
                    prediction_dir=Path(summary["prediction_dir"]),
                    eval_dir=model_root / "official_eval" / run_name,
                    run_id=f"{model_alias}_{lang_slug}_{run_name}_raw",
                    language=language,
                    python=args.python,
                )
                eval_summaries.append(eval_meta)
                write_json(args.output_root / "runs" / "mdpbench_incline" / "official_eval_summaries.json", eval_summaries)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

