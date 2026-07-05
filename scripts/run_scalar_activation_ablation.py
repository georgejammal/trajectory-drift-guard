#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

LOCAL_DATA_ROOT = REPO_ROOT / "data"
if "PARSING_NEURONS_DATA_ROOT" not in os.environ and LOCAL_DATA_ROOT.exists():
    os.environ["PARSING_NEURONS_DATA_ROOT"] = str(LOCAL_DATA_ROOT)

from parsing_neurons_repro.io import clean_float, write_json  # noqa: E402
from parsing_neurons_repro.models import MODEL_PATHS, load_processor, load_vlm, model_spec  # noqa: E402
from parsing_neurons_repro.tasks.ccocr import DEFAULT_CC_OCR_INDEX, DEFAULT_CC_OCR_ROOT, evaluate_ccocr_suite  # noqa: E402
from parsing_neurons_repro.tasks.counting import evaluate_counting_suite  # noqa: E402

from run_mdpbench_ccocr_top3 import (  # noqa: E402
    DEFAULT_PROMPT,
    Config as MDPConfig,
    add_caps,
    evaluate_official,
    generate_config,
    language_samples,
    make_filtered_annotation,
)


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen2_5_vl_3b_instruct": "/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}

SCALAR_MODES = ["zero", "negative_abs", "relu", "scaled_abs_1p2"]


@dataclass(frozen=True)
class StaticConfig:
    model_alias: str
    direction_slug: str
    language: str | None
    component_mode: str
    sigma: float
    window: str

    @property
    def run_name(self) -> str:
        return f"{self.component_mode}_sigma{clean_float(self.sigma)}_layers{self.window}"


COUNTING_CONFIGS = [
    StaticConfig("gemma3_4b_it", "word-minus-digit_en_digits0-1-2-3-4-5-6-7-8-9", None, "mlp_attn", 3.5, "17-23"),
    StaticConfig("gemma3_12b_it", "word-minus-digit_en_digits0-1-2-3-4-5-6-7-8-9", None, "mlp_attn", 5.0, "24-42"),
    StaticConfig(
        "qwen2_5_vl_3b_instruct",
        "word-minus-digit_pooled_en_zh_digits0-1-2-3-4-5-6-7-8-9",
        None,
        "mlp_attn",
        2.5,
        "32-35",
    ),
    StaticConfig(
        "qwen3_vl_8b_instruct",
        "word-minus-digit_pooled_en_zh_digits0-1-2-3-4-5-6-7-8-9",
        None,
        "mlp_attn",
        2.5,
        "34-35",
    ),
]

OCR_CONFIGS = [
    StaticConfig("gemma3_4b_it", "flores_en-to-arabic", "Arabic", "mlp", 2.5, "17-29"),
    StaticConfig("gemma3_4b_it", "flores_en-to-japanese", "Japanese", "mlp_attn", 5.0, "17-29"),
    StaticConfig("gemma3_4b_it", "flores_en-to-korean", "Korean", "mlp", 5.0, "17-29"),
    StaticConfig("gemma3_4b_it", "flores_en-to-russian", "Russian", "mlp_attn", 5.0, "19-31"),
    StaticConfig("gemma3_12b_it", "flores_en-to-arabic", "Arabic", "mlp_attn", 4.5, "24-42"),
    StaticConfig("gemma3_12b_it", "flores_en-to-japanese", "Japanese", "mlp_attn", 5.0, "24-44"),
    StaticConfig("gemma3_12b_it", "flores_en-to-korean", "Korean", "mlp", 3.5, "24-43"),
    StaticConfig("gemma3_12b_it", "flores_en-to-russian", "Russian", "mlp_attn", 5.0, "24-43"),
    StaticConfig("qwen3_vl_8b_instruct", "flores_en-to-arabic", "Arabic", "mlp_attn", 3.5, "18-29"),
    StaticConfig("qwen3_vl_8b_instruct", "flores_en-to-japanese", "Japanese", "mlp_attn", 3.0, "18-29"),
    StaticConfig("qwen3_vl_8b_instruct", "flores_en-to-korean", "Korean", "mlp_attn", 4.5, "18-29"),
    StaticConfig("qwen3_vl_8b_instruct", "flores_en-to-russian", "Russian", "mlp_attn", 4.5, "18-29"),
]

DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 64,
    "gemma3_12b_it": 16,
    "qwen2_5_vl_3b_instruct": 32,
    "qwen3_vl_8b_instruct": 24,
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def language_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", language.strip().lower()).strip("_")


def selection_paths(config: StaticConfig, output_root: Path, task: str) -> tuple[Path | None, Path | None]:
    base = output_root / "selections" / task / config.model_alias / config.direction_slug
    mlp = None
    attn = None
    if config.component_mode in {"mlp", "mlp_attn"}:
        mlp = base / f"mlp_sigma{clean_float(config.sigma)}_layers{config.window}.json"
        if not mlp.exists():
            raise FileNotFoundError(f"Missing MLP selection: {mlp}")
    if config.component_mode in {"attn", "mlp_attn"}:
        attn = base / f"attn_sigma{clean_float(config.sigma)}_layers{config.window}.json"
        if not attn.exists():
            raise FileNotFoundError(f"Missing attention selection: {attn}")
    return mlp, attn


def run_counting(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    configs = [cfg for cfg in COUNTING_CONFIGS if cfg.model_alias in args.models]
    for model_alias in sorted({cfg.model_alias for cfg in configs}):
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        for cfg in [item for item in configs if item.model_alias == model_alias]:
            mlp, attn = selection_paths(cfg, args.output_root, "counting")
            for scalar_mode in args.scalar_modes:
                run_dir = (
                    args.output_root
                    / "runs"
                    / "scalar_ablation"
                    / "counting"
                    / model_alias
                    / cfg.direction_slug
                    / cfg.run_name
                    / scalar_mode
                )
                print(f"[ablation:counting:{model_alias}:{cfg.run_name}:{scalar_mode}]", flush=True)
                summary = evaluate_counting_suite(
                    model_alias=model_alias,
                    processor=processor,
                    model=model,
                    output_dir=run_dir,
                    component_mode=cfg.component_mode,
                    mlp_selection=mlp,
                    attn_selection=attn,
                    batch_size=args.batch_size or DEFAULT_BATCH_SIZES[model_alias],
                    max_new_tokens=16,
                    qwen_max_pixels=401408 if "qwen" in model_alias else None,
                    scalar_mode=scalar_mode,
                )
                summaries.append({"config": cfg.__dict__ | {"run_name": cfg.run_name}, "scalar_mode": scalar_mode, "summary": summary})
                write_json(args.output_root / "runs" / "scalar_ablation" / "counting_summaries.json", summaries)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def run_ccocr(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    configs = [cfg for cfg in OCR_CONFIGS if cfg.model_alias in args.models]
    for model_alias in sorted({cfg.model_alias for cfg in configs}):
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        for cfg in [item for item in configs if item.model_alias == model_alias]:
            assert cfg.language is not None
            mlp, attn = selection_paths(cfg, args.output_root, "ccocr")
            for scalar_mode in args.scalar_modes:
                run_dir = (
                    args.output_root
                    / "runs"
                    / "scalar_ablation"
                    / "ccocr"
                    / model_alias
                    / language_slug(cfg.language)
                    / cfg.run_name
                    / scalar_mode
                )
                print(f"[ablation:ccocr:{model_alias}:{cfg.language}:{cfg.run_name}:{scalar_mode}]", flush=True)
                summary = evaluate_ccocr_suite(
                    model_alias=model_alias,
                    processor=processor,
                    model=model,
                    output_dir=run_dir,
                    component_mode=cfg.component_mode,
                    mlp_selection=mlp,
                    attn_selection=attn,
                    cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                    index_path=DEFAULT_CC_OCR_INDEX,
                    languages=[cfg.language],
                    batch_size=args.batch_size or DEFAULT_BATCH_SIZES[model_alias],
                    max_new_tokens=512,
                    max_gold_tokens_exclusive=500,
                    qwen_max_pixels=1003520 if "qwen" in model_alias else None,
                    scalar_mode=scalar_mode,
                    resume=args.resume,
                )
                summaries.append({"config": cfg.__dict__ | {"run_name": cfg.run_name}, "scalar_mode": scalar_mode, "summary": summary})
                write_json(args.output_root / "runs" / "scalar_ablation" / "ccocr_summaries.json", summaries)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def run_mdpbench(args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    mdpbench_root = args.mdpbench_root.resolve()
    annotation_path = mdpbench_root / "dataset" / "MDPBench_public.json"
    image_dir = mdpbench_root / "dataset" / "MDPBench_img_public"
    configs = [cfg for cfg in OCR_CONFIGS if cfg.model_alias in args.models]
    for model_alias in sorted({cfg.model_alias for cfg in configs}):
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        for cfg in [item for item in configs if item.model_alias == model_alias]:
            assert cfg.language is not None
            all_samples = language_samples(annotation_path, image_dir, cfg.language)
            selected_samples = add_caps(
                all_samples,
                processor,
                multiplier=args.gt_token_cap_multiplier,
                max_cap=args.max_gt_token_cap,
            )
            model_root = (
                args.output_root
                / "runs"
                / "scalar_ablation"
                / "mdpbench"
                / model_alias
                / f"{language_slug(cfg.language)}_le1600"
                / cfg.run_name
            )
            filtered_annotation = model_root / "annotations" / f"MDPBench_public_{language_slug(cfg.language)}_cap_le1600.json"
            make_filtered_annotation(selected_samples, filtered_annotation)
            mdp_config = MDPConfig(cfg.model_alias, cfg.language, cfg.component_mode, cfg.sigma, cfg.window, 0.0)
            for scalar_mode in args.scalar_modes:
                run_dir = model_root / scalar_mode
                print(f"[ablation:mdpbench:{model_alias}:{cfg.language}:{cfg.run_name}:{scalar_mode}]", flush=True)
                summary = generate_config(
                    config=mdp_config,
                    samples=selected_samples,
                    processor=processor,
                    model=model,
                    run_dir=run_dir,
                    output_root=args.output_root,
                    batch_size=args.batch_size or DEFAULT_BATCH_SIZES[model_alias],
                    prompt=args.prompt,
                    qwen_max_pixels=args.qwen_max_pixels if "qwen" in model_alias else None,
                    resume=args.resume,
                    scalar_mode=scalar_mode,
                )
                eval_meta = evaluate_official(
                    official_root=mdpbench_root,
                    annotation_path=filtered_annotation,
                    prediction_dir=Path(summary["prediction_dir"]),
                    eval_dir=model_root / "official_eval" / scalar_mode,
                    run_id=f"{model_alias}_{language_slug(cfg.language)}_{cfg.run_name}_{scalar_mode}_le1600_raw",
                    language=cfg.language,
                    python=args.python,
                )
                summaries.append(
                    {
                        "config": cfg.__dict__ | {"run_name": cfg.run_name},
                        "scalar_mode": scalar_mode,
                        "generation": summary,
                        "eval": eval_meta,
                    }
                )
                write_json(args.output_root / "runs" / "scalar_ablation" / "mdpbench_summaries.json", summaries)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalar activation ablations for selected parsing neurons.")
    parser.add_argument("--tasks", default="counting,ccocr,mdpbench")
    parser.add_argument("--models", default="gemma3_4b_it,gemma3_12b_it,qwen2_5_vl_3b_instruct,qwen3_vl_8b_instruct")
    parser.add_argument("--scalar-modes", default="zero,negative_abs,relu,scaled_abs_1p2")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--mdpbench-root", type=Path, default=Path("external/MultimodalOCR/MDPBench"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt-token-cap-multiplier", type=float, default=1.2)
    parser.add_argument("--max-gt-token-cap", type=int, default=1600)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--python", default="/home/georgejammal/projects/a100env/bin/python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    args.output_root = args.output_root.resolve()
    args.models = set(parse_csv(args.models))
    args.scalar_modes = parse_csv(args.scalar_modes)
    tasks = set(parse_csv(args.tasks))

    unknown_modes = set(args.scalar_modes) - set(SCALAR_MODES)
    if unknown_modes:
        raise ValueError(f"Unsupported scalar modes: {sorted(unknown_modes)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    if "counting" in tasks:
        run_counting(args)
    if "ccocr" in tasks:
        run_ccocr(args)
    if "mdpbench" in tasks:
        run_mdpbench(args)


if __name__ == "__main__":
    main()
