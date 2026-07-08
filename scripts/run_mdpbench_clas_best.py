#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parsing_neurons_repro.generation import decode_new_tokens, model_device, move_to_device, prepare_vlm_inputs  # noqa: E402
from parsing_neurons_repro.interventions import CLASIntervention  # noqa: E402
from parsing_neurons_repro.io import clean_float, write_json  # noqa: E402
from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_vlm, model_spec  # noqa: E402

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

MODEL_ORDER = ["gemma3_4b_it", "gemma3_12b_it", "qwen2_5_vl_3b_instruct", "qwen3_vl_8b_instruct"]
LANGUAGES = ["Arabic", "Japanese", "Korean", "Russian"]

DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 32,
    "gemma3_12b_it": 16,
    "qwen2_5_vl_3b_instruct": 16,
    "qwen3_vl_8b_instruct": 16,
}


@dataclass(frozen=True)
class CLASConfig:
    model_alias: str
    language: str
    stats_layers: str
    window: str
    alpha: float
    ccocr_micro: float

    @property
    def run_name(self) -> str:
        return (
            f"clas_layers{self.window}_a{clean_float(self.alpha)}"
            f"_b0p4_g0p2_all"
        )


BEST_CCOCR_CLAS: list[CLASConfig] = [
    CLASConfig("gemma3_4b_it", "Arabic", "17-33", "17-29", 1.0, 72.94003151876052),
    CLASConfig("gemma3_4b_it", "Japanese", "17-33", "17-29", -1.0, 52.78307607317806),
    CLASConfig("gemma3_4b_it", "Korean", "17-33", "17-29", -1.0, 50.970534792431074),
    CLASConfig("gemma3_4b_it", "Russian", "17-33", "17-29", 1.0, 57.62195116971204),
    CLASConfig("gemma3_12b_it", "Arabic", "0-47", "40-45", -1.0, 79.29142419688387),
    CLASConfig("gemma3_12b_it", "Japanese", "0-47", "40-45", -1.0, 63.61530091165765),
    CLASConfig("gemma3_12b_it", "Korean", "0-47", "40-45", -1.0, 62.91251223861667),
    CLASConfig("gemma3_12b_it", "Russian", "0-47", "40-45", 1.0, 61.257861585231566),
    CLASConfig("qwen2_5_vl_3b_instruct", "Arabic", "0-35", "32-33", -1.0, 70.54911106862768),
    CLASConfig("qwen2_5_vl_3b_instruct", "Japanese", "0-35", "32-33", -1.0, 55.541578052704274),
    CLASConfig("qwen2_5_vl_3b_instruct", "Korean", "0-35", "32-33", -1.0, 62.28577218433165),
    CLASConfig("qwen2_5_vl_3b_instruct", "Russian", "0-35", "32-33", -1.0, 71.4731909964422),
    CLASConfig("qwen3_vl_8b_instruct", "Arabic", "0-35", "32-33", -1.0, 75.87721462729745),
    CLASConfig("qwen3_vl_8b_instruct", "Japanese", "0-35", "32-33", 3.0, 55.651420478335666),
    CLASConfig("qwen3_vl_8b_instruct", "Korean", "0-35", "32-33", 3.0, 62.21725520427406),
    CLASConfig("qwen3_vl_8b_instruct", "Russian", "0-35", "32-33", -1.0, 78.14275124657787),
]


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def language_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", language.strip().lower()).strip("_")


def stats_path(output_root: Path, config: CLASConfig) -> Path:
    return (
        output_root
        / "clas"
        / "stats"
        / config.model_alias
        / (
            "flores_langs-arabic-japanese-korean-russian_"
            f"layers{config.stats_layers}_tau0_last_nonpad_mean_n100"
        )
        / "stats.json"
    )


def generate_clas_config(
    *,
    config: CLASConfig,
    stats_json: Path,
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

    context = CLASIntervention(
        model,
        stats_json,
        config.window,
        target_language=config.language,
        alpha=config.alpha,
        beta=0.4,
        gamma=0.2,
        token_scope="all_positions",
        specific_scope="all",
    )
    start = time.time()
    written = 0
    with context:
        for batch_idx, batch in enumerate(batches(queued, batch_size), start=1):
            max_new_tokens = max(sample["gt_token_cap"] for sample in batch)
            inputs = prepare_vlm_inputs(
                model_alias=config.model_alias,
                processor=processor,
                images=[sample["image_path"] for sample in batch],
                questions=[prompt for _ in batch],
                qwen_max_pixels=qwen_max_pixels,
            )
            inputs = move_to_device(inputs, model_device(model))
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            predictions = decode_new_tokens(
                model_alias=config.model_alias,
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
                f"[mdpbench-clas:{config.model_alias}:{config.language}:{config.run_name}] "
                f"batch={batch_idx} written={written}/{len(queued)} batch_cap={max_new_tokens}",
                flush=True,
            )

    summary = {
        "task": "mdpbench",
        "method": "clas",
        "model_alias": config.model_alias,
        "language": config.language,
        "run_name": config.run_name,
        "stats_path": str(stats_json),
        "stats_layers": config.stats_layers,
        "window": config.window,
        "alpha": config.alpha,
        "beta": 0.4,
        "gamma": 0.2,
        "ccocr_micro": config.ccocr_micro,
        "num_selected_samples": len(samples),
        "queued": len(queued),
        "written_this_run": written,
        "prediction_dir": str(prediction_dir),
        "elapsed_seconds": time.time() - start,
    }
    write_json(run_dir / "generation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run best CC-OCR CLAS configs on MDPBench language chunks.")
    parser.add_argument("--models", default="gemma3_4b_it,gemma3_12b_it,qwen2_5_vl_3b_instruct,qwen3_vl_8b_instruct")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs/clas_flores")))
    parser.add_argument("--mdpbench-root", type=Path, default=Path("external/MultimodalOCR/MDPBench"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gt-token-cap-multiplier", type=float, default=1.2)
    parser.add_argument("--max-gt-token-cap", type=int, default=1600)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--python", default="/home/georgejammal/projects/a100env/bin/python")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    output_root = args.output_root.resolve()
    model_filter = set(parse_csv(args.models))
    language_filter = set(parse_csv(args.languages))
    mdpbench_root = args.mdpbench_root.resolve()
    annotation_path = mdpbench_root / "dataset" / "MDPBench_public.json"
    image_dir = mdpbench_root / "dataset" / "MDPBench_img_public"
    all_summaries = []
    all_eval_summaries = []

    for model_alias in MODEL_ORDER:
        configs = [
            config
            for config in BEST_CCOCR_CLAS
            if config.model_alias == model_alias
            and config.model_alias in model_filter
            and config.language in language_filter
        ]
        if not configs:
            continue
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path)
        batch_size = args.batch_size or DEFAULT_BATCH_SIZES[model_alias]

        for config in configs:
            stats_json = stats_path(output_root, config)
            if not stats_json.exists():
                raise FileNotFoundError(f"Missing CLAS stats for {config}: {stats_json}")
            all_samples = language_samples(annotation_path, image_dir, config.language)
            selected_samples = add_caps(
                all_samples,
                processor,
                multiplier=args.gt_token_cap_multiplier,
                max_cap=args.max_gt_token_cap,
            )
            lang_slug = language_slug(config.language)
            model_root = (
                output_root
                / "runs"
                / "mdpbench_clas"
                / model_alias
                / f"{lang_slug}_best_ccocr_clas_le1600"
            )
            filtered_annotation = model_root / "annotations" / f"MDPBench_public_{lang_slug}_cap_le1600.json"
            make_filtered_annotation(selected_samples, filtered_annotation)
            write_json(
                model_root / "subset_metadata.json",
                {
                    "model_alias": model_alias,
                    "language": config.language,
                    "num_all_language": len(all_samples),
                    "num_selected": len(selected_samples),
                    "gt_token_cap_multiplier": args.gt_token_cap_multiplier,
                    "max_gt_token_cap": args.max_gt_token_cap,
                    "cap_min": min(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
                    "cap_max": max(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
                },
            )
            run_dir = model_root / config.run_name
            print(
                f"[mdpbench-clas:{model_alias}:{config.language}] selected "
                f"{len(selected_samples)}/{len(all_samples)} samples; {config.run_name}",
                flush=True,
            )
            summary = generate_clas_config(
                config=config,
                stats_json=stats_json,
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
            write_json(output_root / "runs" / "mdpbench_clas" / "generation_summaries.json", all_summaries)
            if args.evaluate:
                run_id = f"{model_alias}_{lang_slug}_{config.run_name}_le1600_raw"
                eval_meta = evaluate_official(
                    official_root=mdpbench_root,
                    annotation_path=filtered_annotation,
                    prediction_dir=Path(summary["prediction_dir"]),
                    eval_dir=model_root / "official_eval" / config.run_name,
                    run_id=run_id,
                    language=config.language,
                    python=args.python,
                )
                all_eval_summaries.append(eval_meta)
                write_json(output_root / "runs" / "mdpbench_clas" / "official_eval_summaries.json", all_eval_summaries)

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
