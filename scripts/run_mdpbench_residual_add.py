#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
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

from parsing_neurons_repro.directions import flores_direction  # noqa: E402
from parsing_neurons_repro.generation import decode_new_tokens, model_device, move_to_device, prepare_vlm_inputs  # noqa: E402
from parsing_neurons_repro.interventions import ResidualAddIntervention  # noqa: E402
from parsing_neurons_repro.io import read_json, write_json  # noqa: E402
from parsing_neurons_repro.models import (  # noqa: E402
    MODEL_PATHS,
    load_processor,
    load_tensor,
    load_tokenizer,
    load_vlm,
    model_spec,
    weight_map,
    weight_templates,
)

from run_mdpbench_ccocr_top3 import (  # noqa: E402
    DEFAULT_PROMPT,
    add_caps,
    batches,
    evaluate_official,
    gt_text,
    language_samples,
    make_filtered_annotation,
)


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}

MODEL_ORDER = ["gemma3_4b_it", "gemma3_12b_it", "qwen3_vl_8b_instruct"]
LANGUAGE_TO_DIRECTION = {
    "Arabic": "arabic",
    "Japanese": "japanese",
    "Korean": "korean",
    "Russian": "russian",
}

# Best alpha=1, last-token residual-add layer on filtered CC-OCR.
CCOCR_BEST_RESIDUAL_LAYERS = {
    "gemma3_4b_it": {"Arabic": 29, "Japanese": 17, "Korean": 17, "Russian": 17},
    "gemma3_12b_it": {"Arabic": 43, "Japanese": 43, "Korean": 43, "Russian": 24},
    "qwen3_vl_8b_instruct": {"Arabic": 29, "Japanese": 18, "Korean": 29, "Russian": 24},
}

DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 32,
    "gemma3_12b_it": 16,
    "qwen3_vl_8b_instruct": 16,
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def load_embedding(model_path: Path) -> torch.Tensor:
    index = weight_map(model_path)
    templates = weight_templates(index)
    return load_tensor(model_path, index, templates["embed"])


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def language_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", language.strip().lower()).strip("_")


def clean_alpha(alpha: float) -> str:
    return ("%g" % alpha).replace("-", "m").replace(".", "p")


def flores_pairs_path(flores_root: Path, language: str) -> Path:
    return flores_root / f"en_to_{language.lower()}.json"


def generate_residual_config(
    *,
    model_alias: str,
    language: str,
    layer: int,
    alpha: float,
    token_scope: str,
    direction: torch.Tensor,
    direction_metadata: dict[str, Any],
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

    metadata = {
        "task": "mdpbench",
        "model_alias": model_alias,
        "language": language,
        "layer": int(layer),
        "alpha": float(alpha),
        "token_scope": token_scope,
        "direction_metadata": direction_metadata,
        "intervention": "hidden_states <- hidden_states + alpha * d at layer input",
        "num_selected_samples": len(samples),
        "queued": len(queued),
        "prediction_dir": str(prediction_dir),
    }
    write_json(run_dir / "residual_add_metadata.json", metadata)

    start = time.time()
    written = 0
    with ResidualAddIntervention(model, layers=[layer], direction=direction, alpha=alpha, token_scope=token_scope):
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
                f"[mdpbench-resid:{model_alias}:{language}:layer{layer}] "
                f"batch={batch_idx} written={written}/{len(queued)} batch_cap={max_new_tokens}",
                flush=True,
            )

    metadata.update({"written_this_run": written, "elapsed_seconds": time.time() - start})
    write_json(run_dir / "generation_summary.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    default_flores_root = (
        Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", str(LOCAL_DATA_ROOT)))
        / "flores_transfer_pairs"
        / "flores101_en_to_cc_ocr_languages_random500"
        / "pairs_by_language"
    )
    parser = argparse.ArgumentParser(description="Run alpha=1 residual-add baseline on MDPBench language chunks.")
    parser.add_argument("--models", default="gemma3_4b_it,gemma3_12b_it,qwen3_vl_8b_instruct")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs")))
    parser.add_argument("--mdpbench-root", type=Path, default=Path("external/MultimodalOCR/MDPBench"))
    parser.add_argument("--flores-root", type=Path, default=default_flores_root)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--token-scope", choices=["last_position", "all_positions"], default="last_position")
    parser.add_argument("--gt-token-cap-multiplier", type=float, default=1.2)
    parser.add_argument("--max-gt-token-cap", type=int, default=1600)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--python", default="/home/georgejammal/projects/a100env/bin/python")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    output_root = args.output_root.resolve()
    model_filter = set(parse_csv(args.models))
    languages = parse_csv(args.languages)
    mdpbench_root = args.mdpbench_root.resolve()
    annotation_path = mdpbench_root / "dataset" / "MDPBench_public.json"
    image_dir = mdpbench_root / "dataset" / "MDPBench_img_public"
    all_summaries = []
    eval_summaries = []

    for model_alias in MODEL_ORDER:
        if model_alias not in model_filter:
            continue
        spec = model_spec(model_alias)
        tokenizer = load_tokenizer(spec.path)
        embedding = load_embedding(spec.path)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        batch_size = args.batch_size or DEFAULT_BATCH_SIZES[model_alias]

        for language in languages:
            if language not in LANGUAGE_TO_DIRECTION:
                raise ValueError(f"Unsupported language: {language}")
            all_samples = language_samples(annotation_path, image_dir, language)
            selected_samples = add_caps(
                all_samples,
                processor,
                multiplier=args.gt_token_cap_multiplier,
                max_cap=args.max_gt_token_cap,
            )
            model_root = (
                output_root
                / "runs"
                / "mdpbench_residual_add"
                / model_alias
                / f"{language_slug(language)}_alpha{clean_alpha(args.alpha)}_{args.token_scope}_le1600"
            )
            filtered_annotation = model_root / "annotations" / f"MDPBench_public_{language_slug(language)}_cap_le1600.json"
            make_filtered_annotation(selected_samples, filtered_annotation)
            write_json(
                model_root / "subset_metadata.json",
                {
                    "model_alias": model_alias,
                    "language": language,
                    "num_all_language": len(all_samples),
                    "num_selected": len(selected_samples),
                    "gt_token_cap_multiplier": args.gt_token_cap_multiplier,
                    "max_gt_token_cap": args.max_gt_token_cap,
                    "cap_min": min(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
                    "cap_max": max(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
                },
            )
            direction, direction_metadata = flores_direction(
                tokenizer,
                embedding,
                pairs_path=flores_pairs_path(args.flores_root, language),
                source_field="english",
                target_field="target",
                max_pairs=500,
            )
            layer = CCOCR_BEST_RESIDUAL_LAYERS[model_alias][language]
            run_name = f"resid_alpha{clean_alpha(args.alpha)}_layer{layer}_{args.token_scope}"
            run_dir = model_root / run_name
            print(
                f"[mdpbench-resid:{model_alias}:{language}] selected {len(selected_samples)}/{len(all_samples)} "
                f"samples layer={layer}",
                flush=True,
            )
            summary = generate_residual_config(
                model_alias=model_alias,
                language=language,
                layer=layer,
                alpha=args.alpha,
                token_scope=args.token_scope,
                direction=direction,
                direction_metadata=direction_metadata,
                samples=selected_samples,
                processor=processor,
                model=model,
                run_dir=run_dir,
                batch_size=batch_size,
                prompt=args.prompt,
                qwen_max_pixels=args.qwen_max_pixels if "qwen" in model_alias else None,
                resume=args.resume,
            )
            all_summaries.append(summary)
            write_json(output_root / "runs" / "mdpbench_residual_add" / "generation_summaries.json", all_summaries)
            if args.evaluate:
                run_id = f"{model_alias}_{language_slug(language)}_{run_name}_le1600_raw"
                eval_meta = evaluate_official(
                    official_root=mdpbench_root,
                    annotation_path=filtered_annotation,
                    prediction_dir=Path(summary["prediction_dir"]),
                    eval_dir=model_root / "official_eval" / run_name,
                    run_id=run_id,
                    language=language,
                    python=args.python,
                )
                eval_summaries.append(eval_meta)
                write_json(output_root / "runs" / "mdpbench_residual_add" / "official_eval_summaries.json", eval_summaries)

        del model, processor, tokenizer, embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
