from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import torch

from .directions import counting_direction, flores_direction
from .io import clean_float, parse_csv, parse_float_grid, slug_parts, write_json
from .models import load_processor, load_tensor, load_tokenizer, load_vlm, model_spec, weight_map, weight_templates
from .scoring import score_static_components
from .tasks.ccocr import DEFAULT_CC_OCR_INDEX, DEFAULT_CC_OCR_ROOT, evaluate_ccocr_suite
from .tasks.counting import DEFAULT_COUNTING_DATASETS, evaluate_counting_suite


def load_embedding(model_path: Path) -> torch.Tensor:
    index = weight_map(model_path)
    templates = weight_templates(index)
    return load_tensor(model_path, index, templates["embed"])


def build_direction(
    *,
    task: str,
    model_alias: str,
    direction_kind: str,
    counting_language: str,
    counting_digits: list[int],
    flores_pairs_path: Path | None,
    flores_source_field: str,
    flores_target_field: str,
    flores_max_pairs: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = model_spec(model_alias)
    tokenizer = load_tokenizer(spec.path)
    embedding = load_embedding(spec.path)
    if task == "counting":
        return counting_direction(
            tokenizer,
            embedding,
            direction=direction_kind,
            language=counting_language,
            digits=counting_digits,
        )
    if task == "ccocr":
        if flores_pairs_path is None:
            raise ValueError("ccocr requires flores_pairs_path.")
        return flores_direction(
            tokenizer,
            embedding,
            pairs_path=flores_pairs_path,
            source_field=flores_source_field,
            target_field=flores_target_field,
            max_pairs=flores_max_pairs,
        )
    raise ValueError(f"Unsupported task: {task}")


def selection_path(
    *,
    output_root: Path,
    task: str,
    model_alias: str,
    direction_slug: str,
    window: str,
    sigma: float,
    component: str,
) -> Path:
    return (
        output_root
        / "selections"
        / task
        / model_alias
        / direction_slug
        / f"{component}_sigma{clean_float(sigma)}_layers{window}.json"
    )


def ensure_selection(
    *,
    output_root: Path,
    task: str,
    model_alias: str,
    direction_slug: str,
    direction: torch.Tensor,
    direction_metadata: dict[str, Any],
    window: str,
    sigma: float,
    component: str,
) -> Path:
    spec = model_spec(model_alias)
    path = selection_path(
        output_root=output_root,
        task=task,
        model_alias=model_alias,
        direction_slug=direction_slug,
        window=window,
        sigma=sigma,
        component=component,
    )
    if not path.exists():
        score_static_components(
            model=spec,
            direction=direction,
            direction_metadata=direction_metadata,
            window=window,
            sigma=sigma,
            component=component,
            output_path=path,
        )
    return path


def run_counting_search(
    *,
    model_alias: str,
    output_root: Path,
    windows: list[str],
    sigmas: list[float],
    component_modes: list[str],
    direction_kind: str,
    counting_language: str,
    counting_digits: list[int],
    batch_size: int,
    limit: int | None,
    max_new_tokens: int,
    qwen_max_pixels: int | None,
    run_baseline: bool,
) -> list[dict[str, Any]]:
    direction, direction_metadata = build_direction(
        task="counting",
        model_alias=model_alias,
        direction_kind=direction_kind,
        counting_language=counting_language,
        counting_digits=counting_digits,
        flores_pairs_path=None,
        flores_source_field="english",
        flores_target_field="target",
        flores_max_pairs=None,
    )
    direction_slug = slug_parts(direction_kind, counting_language, "digits" + "-".join(map(str, counting_digits)))
    spec = model_spec(model_alias)
    tokenizer = load_tokenizer(spec.path)
    embedding = load_embedding(spec.path)
    processor = load_processor(spec.path)
    model = load_vlm(model_alias, spec.path)
    run_summaries: list[dict[str, Any]] = []
    task_root = output_root / "runs" / "counting" / model_alias / direction_slug

    if run_baseline:
        baseline_dir = task_root / "baseline"
        run_summaries.append(
            evaluate_counting_suite(
                model_alias=model_alias,
                processor=processor,
                model=model,
                output_dir=baseline_dir,
                component_mode="baseline",
                mlp_selection=None,
                attn_selection=None,
                datasets=DEFAULT_COUNTING_DATASETS,
                batch_size=batch_size,
                limit=limit,
                max_new_tokens=max_new_tokens,
                qwen_max_pixels=qwen_max_pixels,
            )
        )

    for window in windows:
        for sigma in sigmas:
            mlp_selection = ensure_selection(
                output_root=output_root,
                task="counting",
                model_alias=model_alias,
                direction_slug=direction_slug,
                direction=direction,
                direction_metadata=direction_metadata,
                window=window,
                sigma=sigma,
                component="mlp",
            )
            attn_selection = None
            if "mlp_attn" in component_modes or "attn" in component_modes:
                attn_selection = ensure_selection(
                    output_root=output_root,
                    task="counting",
                    model_alias=model_alias,
                    direction_slug=direction_slug,
                    direction=direction,
                    direction_metadata=direction_metadata,
                    window=window,
                    sigma=sigma,
                    component="attn",
                )
            for component_mode in component_modes:
                run_dir = task_root / slug_parts(component_mode, "sigma" + clean_float(sigma), "layers" + window)
                run_summaries.append(
                    evaluate_counting_suite(
                        model_alias=model_alias,
                        processor=processor,
                        model=model,
                        output_dir=run_dir,
                        component_mode=component_mode,
                        mlp_selection=mlp_selection if component_mode in {"mlp", "mlp_attn"} else None,
                        attn_selection=attn_selection if component_mode in {"attn", "mlp_attn"} else None,
                        datasets=DEFAULT_COUNTING_DATASETS,
                        batch_size=batch_size,
                        limit=limit,
                        max_new_tokens=max_new_tokens,
                        qwen_max_pixels=qwen_max_pixels,
                    )
                )
    write_json(task_root / "search_summary.json", run_summaries)
    return run_summaries


def run_ccocr_search(
    *,
    model_alias: str,
    output_root: Path,
    windows: list[str],
    sigmas: list[float],
    component_modes: list[str],
    languages: list[str],
    flores_root: Path,
    batch_size: int,
    limit: int | None,
    max_new_tokens: int,
    max_gold_tokens_exclusive: int | None,
    qwen_max_pixels: int | None,
    run_baseline: bool,
) -> list[dict[str, Any]]:
    spec = model_spec(model_alias)
    processor = load_processor(spec.path)
    model = load_vlm(model_alias, spec.path)
    task_root = output_root / "runs" / "ccocr" / model_alias
    run_summaries: list[dict[str, Any]] = []

    if run_baseline:
        baseline_dir = task_root / "baseline"
        run_summaries.append(
            evaluate_ccocr_suite(
                model_alias=model_alias,
                processor=processor,
                model=model,
                output_dir=baseline_dir,
                component_mode="baseline",
                mlp_selection=None,
                attn_selection=None,
                cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                index_path=DEFAULT_CC_OCR_INDEX,
                languages=languages,
                batch_size=batch_size,
                limit=limit,
                max_new_tokens=max_new_tokens,
                max_gold_tokens_exclusive=max_gold_tokens_exclusive,
                qwen_max_pixels=qwen_max_pixels,
            )
        )

    for language in languages:
        pairs_path = flores_root / f"en_to_{language.lower()}.json"
        direction, direction_metadata = flores_direction(
            tokenizer,
            embedding,
            pairs_path=pairs_path,
            source_field="english",
            target_field="target",
            max_pairs=500,
        )
        direction_slug = slug_parts("flores", "en-to-" + language.lower())
        for window in windows:
            for sigma in sigmas:
                mlp_selection = ensure_selection(
                    output_root=output_root,
                    task="ccocr",
                    model_alias=model_alias,
                    direction_slug=direction_slug,
                    direction=direction,
                    direction_metadata=direction_metadata,
                    window=window,
                    sigma=sigma,
                    component="mlp",
                )
                attn_selection = None
                if "mlp_attn" in component_modes or "attn" in component_modes:
                    attn_selection = ensure_selection(
                        output_root=output_root,
                        task="ccocr",
                        model_alias=model_alias,
                        direction_slug=direction_slug,
                        direction=direction,
                        direction_metadata=direction_metadata,
                        window=window,
                        sigma=sigma,
                        component="attn",
                    )
                for component_mode in component_modes:
                    run_dir = (
                        task_root
                        / direction_slug
                        / slug_parts(component_mode, "sigma" + clean_float(sigma), "layers" + window)
                    )
                    run_summaries.append(
                        evaluate_ccocr_suite(
                            model_alias=model_alias,
                            processor=processor,
                            model=model,
                            output_dir=run_dir,
                            component_mode=component_mode,
                            mlp_selection=mlp_selection if component_mode in {"mlp", "mlp_attn"} else None,
                            attn_selection=attn_selection if component_mode in {"attn", "mlp_attn"} else None,
                            cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                            index_path=DEFAULT_CC_OCR_INDEX,
                            languages=[language],
                            batch_size=batch_size,
                            limit=limit,
                            max_new_tokens=max_new_tokens,
                            max_gold_tokens_exclusive=max_gold_tokens_exclusive,
                            qwen_max_pixels=qwen_max_pixels,
                        )
                    )
    write_json(task_root / "search_summary.json", run_summaries)
    return run_summaries


def run_search_from_args(args: Any) -> list[dict[str, Any]]:
    windows = parse_csv(args.windows)
    sigmas = parse_float_grid(args.sigmas)
    component_modes = parse_csv(args.component_modes)
    output_root = Path(args.output_root).resolve()
    if args.task == "counting":
        return run_counting_search(
            model_alias=args.model_alias,
            output_root=output_root,
            windows=windows,
            sigmas=sigmas,
            component_modes=component_modes,
            direction_kind=args.direction,
            counting_language=args.counting_language,
            counting_digits=[int(item) for item in parse_csv(args.counting_digits)],
            batch_size=args.batch_size,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            qwen_max_pixels=args.qwen_max_pixels,
            run_baseline=args.run_baseline,
        )
    if args.task == "ccocr":
        return run_ccocr_search(
            model_alias=args.model_alias,
            output_root=output_root,
            windows=windows,
            sigmas=sigmas,
            component_modes=component_modes,
            languages=parse_csv(args.languages),
            flores_root=Path(args.flores_root).resolve(),
            batch_size=args.batch_size,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            max_gold_tokens_exclusive=args.max_gold_tokens_exclusive,
            qwen_max_pixels=args.qwen_max_pixels,
            run_baseline=args.run_baseline,
        )
    raise ValueError(f"Unsupported task: {args.task}")
