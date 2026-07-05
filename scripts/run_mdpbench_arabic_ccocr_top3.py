#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from parsing_neurons_repro.generation import decode_new_tokens, model_device, move_to_device, prepare_vlm_inputs
from parsing_neurons_repro.interventions import build_abs_intervention
from parsing_neurons_repro.io import clean_float, read_json, slug_parts, write_json
from parsing_neurons_repro.models import load_processor, load_vlm, model_spec


DEFAULT_PROMPT = """
You are an advanced hybrid OCR engine capable of processing multilingual text mixed with mathematical notation. Your goal is to transcribe the content with high fidelity. Strict Rules:

1. Multilingual Precision: Transcribe text exactly as it appears in the original language. Do not translate, summarize, or correct original spelling errors.

2. Math Formatting: Identify all mathematical expressions and convert them into LaTeX.

3. Inline Math: Use single dollar signs ($x$) for inline math (formulas within a sentence).

4. Display Math: Use double dollar signs ($$x$$) for display math (standalone formulas on their own lines).

5. Layout & Structure: Use Markdown to preserve the visual structure (headers, paragraphs, lists).

6. Table Formatting: Use HTML tags (e.g., <table>, <tr>, <th>, <td>) to generate any tables found in the text.

7. Output Only: Output the transcribed text directly without any conversational filler.
""".strip()


@dataclass(frozen=True)
class Config:
    model_alias: str
    component_mode: str
    sigma: float
    window: str
    ccocr_micro: float

    @property
    def run_name(self) -> str:
        if self.component_mode == "baseline":
            return "baseline"
        return slug_parts(self.component_mode, "sigma" + clean_float(self.sigma), "layers" + self.window)


TOP3_ARABIC_CCOCR_CONFIGS = [
    Config("gemma3_4b_it", "baseline", 0.0, "none", 0.0),
    Config("gemma3_4b_it", "mlp_attn", 4.5, "17-29", 74.34),
    Config("gemma3_4b_it", "mlp", 2.5, "17-29", 74.07),
    Config("gemma3_4b_it", "mlp", 4.5, "17-29", 73.95),
    Config("gemma3_12b_it", "baseline", 0.0, "none", 0.0),
    Config("gemma3_12b_it", "mlp_attn", 3.5, "24-43", 81.12),
    Config("gemma3_12b_it", "mlp_attn", 3.5, "24-42", 80.76),
    Config("gemma3_12b_it", "mlp_attn", 4.5, "24-42", 80.62),
    Config("qwen3_vl_8b_instruct", "baseline", 0.0, "none", 0.0),
    Config("qwen3_vl_8b_instruct", "mlp_attn", 3.0, "18-29", 79.54),
    Config("qwen3_vl_8b_instruct", "mlp_attn", 4.0, "18-29", 78.79),
    Config("qwen3_vl_8b_instruct", "mlp_attn", 3.5, "18-29", 76.73),
]

DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 24,
    "gemma3_12b_it": 8,
    "qwen3_vl_8b_instruct": 8,
}


def language_slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", language.strip().lower()).strip("_")


def load_annotations(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list annotation payload in {path}")
    return rows


def gt_text(row: dict[str, Any]) -> str:
    return "\n".join(str(block["text"]) for block in row.get("layout_dets", []) if block.get("text"))


def arabic_samples(annotation_path: Path, image_dir: Path) -> list[dict[str, Any]]:
    samples = []
    for idx, row in enumerate(load_annotations(annotation_path)):
        page_info = row.get("page_info") or {}
        attrs = page_info.get("page_attribute") or {}
        if language_slug(str(attrs.get("language", ""))) != "arabic":
            continue
        image_name = page_info.get("image_path")
        if not image_name:
            raise ValueError(f"Arabic annotation {idx} is missing image_path")
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing MDPBench image: {image_path}")
        samples.append(
            {
                "annotation_index": idx,
                "annotation": row,
                "image_name": image_name,
                "stem": Path(image_name).stem,
                "image_path": image_path,
                "gt_text": gt_text(row),
            }
        )
    return samples


def add_caps(
    samples: list[dict[str, Any]],
    processor: Any,
    *,
    multiplier: float,
    max_cap: int,
) -> list[dict[str, Any]]:
    tokenizer = processor.tokenizer
    selected = []
    for sample in samples:
        count = len(tokenizer(sample["gt_text"], add_special_tokens=False).input_ids)
        cap = max(16, int(math.ceil(multiplier * count)))
        if cap <= max_cap:
            selected.append({**sample, "gt_token_count": count, "gt_token_cap": cap})
    return selected


def selection_paths(config: Config, output_root: Path) -> tuple[Path | None, Path | None]:
    base = output_root / "selections" / "ccocr" / config.model_alias / "flores_en-to-arabic"
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


def batches(samples: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(samples, key=lambda item: item["gt_token_cap"])
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def generate_config(
    *,
    config: Config,
    samples: list[dict[str, Any]],
    processor: Any,
    model: Any,
    run_dir: Path,
    output_root: Path,
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

    mlp_selection, attn_selection = selection_paths(config, output_root)
    context = build_abs_intervention(
        model=model,
        component_mode=config.component_mode,
        mlp_selection=mlp_selection,
        attn_selection=attn_selection,
        mlp_token_scope="all_positions",
        attn_token_scope="last_position",
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
                f"[mdpbench:{config.model_alias}:{config.run_name}] "
                f"batch={batch_idx} written={written}/{len(queued)} "
                f"batch_cap={max_new_tokens}",
                flush=True,
            )

    summary = {
        "task": "mdpbench_arabic",
        "model_alias": config.model_alias,
        "component_mode": config.component_mode,
        "sigma": config.sigma,
        "window": config.window,
        "run_name": config.run_name,
        "ccocr_arabic_micro": config.ccocr_micro,
        "num_selected_samples": len(samples),
        "queued": len(queued),
        "written_this_run": written,
        "prediction_dir": str(prediction_dir),
        "mlp_selection": str(mlp_selection) if mlp_selection else None,
        "attn_selection": str(attn_selection) if attn_selection else None,
        "elapsed_seconds": time.time() - start,
    }
    write_json(run_dir / "generation_summary.json", summary)
    return summary


def make_filtered_annotation(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample["annotation"] for sample in samples]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_eval_config(annotation_path: Path, prediction_dir: Path) -> dict[str, Any]:
    prediction_path = prediction_dir if prediction_dir.is_absolute() else prediction_dir.absolute()
    return {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", "CDM"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(annotation_path.resolve())},
                "prediction": {"data_path": str(prediction_path)},
                "match_method": "quick_match",
                "filter": {"language": "Arabic"},
            },
        }
    }


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def evaluate_official(
    *,
    official_root: Path,
    annotation_path: Path,
    prediction_dir: Path,
    eval_dir: Path,
    run_id: str,
    python: str,
) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    config_path = eval_dir / "eval_config.yaml"

    # MDPBench names the result folder from basename(prediction.data_path).
    # Use a run-specific symlink so configs do not collide in result/predictions_result.
    eval_prediction_dir = eval_dir / f"{run_id}_predictions"
    if eval_prediction_dir.exists() or eval_prediction_dir.is_symlink():
        if eval_prediction_dir.is_dir() and not eval_prediction_dir.is_symlink():
            shutil.rmtree(eval_prediction_dir)
        else:
            eval_prediction_dir.unlink()
    os.symlink(prediction_dir.resolve(), eval_prediction_dir, target_is_directory=True)

    config_path.write_text(
        yaml.safe_dump(make_eval_config(annotation_path, eval_prediction_dir), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    result_name = f"{eval_prediction_dir.name}_result"
    official_result_dir = official_root / "result" / result_name
    if official_result_dir.exists():
        shutil.rmtree(official_result_dir)
    validation = run_command([python, "pdf_validation.py", "--config", str(config_path), "--slim"], cwd=official_root)
    (eval_dir / "pdf_validation.log").write_text(validation.stdout, encoding="utf-8")
    if validation.returncode != 0:
        return {
            "run_id": run_id,
            "ok": False,
            "stage": "pdf_validation",
            "returncode": validation.returncode,
            "log": str(eval_dir / "pdf_validation.log"),
        }
    scores = run_command([python, "tools/calculate_scores.py", "--result_folder", str(official_result_dir)], cwd=official_root)
    (eval_dir / "score_summary.txt").write_text(scores.stdout, encoding="utf-8")
    ok = scores.returncode == 0
    metadata = {
        "run_id": run_id,
        "ok": ok,
        "stage": "complete" if ok else "calculate_scores",
        "returncode": scores.returncode,
        "annotation_path": str(annotation_path),
        "prediction_dir": str(prediction_dir),
        "eval_prediction_dir": str(eval_prediction_dir),
        "official_result_dir": str(official_result_dir),
        "config_path": str(config_path),
        "score_summary": str(eval_dir / "score_summary.txt"),
        "pdf_validation_log": str(eval_dir / "pdf_validation.log"),
    }
    write_json(eval_dir / "metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-3 Arabic CC-OCR configs on Arabic MDPBench.")
    parser.add_argument("--models", default="gemma3_4b_it,gemma3_12b_it,qwen3_vl_8b_instruct")
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("PARSING_NEURONS_OUTPUT_ROOT", "outputs")))
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
    model_filter = {item.strip() for item in args.models.split(",") if item.strip()}
    mdpbench_root = args.mdpbench_root.resolve()
    annotation_path = mdpbench_root / "dataset" / "MDPBench_public.json"
    image_dir = mdpbench_root / "dataset" / "MDPBench_img_public"
    all_samples = arabic_samples(annotation_path, image_dir)
    summaries = []
    eval_summaries = []
    for model_alias in ["gemma3_4b_it", "gemma3_12b_it", "qwen3_vl_8b_instruct"]:
        if model_alias not in model_filter:
            continue
        spec = model_spec(model_alias)
        processor = load_processor(spec.path)
        selected_samples = add_caps(
            all_samples,
            processor,
            multiplier=args.gt_token_cap_multiplier,
            max_cap=args.max_gt_token_cap,
        )
        model_root = args.output_root.resolve() / "runs" / "mdpbench" / model_alias / "arabic_ccocr_top3_le1600"
        filtered_annotation = model_root / "annotations" / "MDPBench_public_arabic_cap_le1600.json"
        make_filtered_annotation(selected_samples, filtered_annotation)
        write_json(
            model_root / "subset_metadata.json",
            {
                "model_alias": model_alias,
                "source_annotation": str(annotation_path),
                "filtered_annotation": str(filtered_annotation),
                "language": "Arabic",
                "num_all_arabic": len(all_samples),
                "num_selected": len(selected_samples),
                "gt_token_cap_multiplier": args.gt_token_cap_multiplier,
                "max_gt_token_cap": args.max_gt_token_cap,
                "cap_min": min(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
                "cap_max": max(sample["gt_token_cap"] for sample in selected_samples) if selected_samples else None,
            },
        )
        print(
            f"[mdpbench:{model_alias}] selected {len(selected_samples)}/{len(all_samples)} Arabic samples "
            f"with ceil({args.gt_token_cap_multiplier:g} * gold_tokens) <= {args.max_gt_token_cap}",
            flush=True,
        )
        model = load_vlm(model_alias, spec.path)
        batch_size = args.batch_size or DEFAULT_BATCH_SIZES[model_alias]
        for config in TOP3_ARABIC_CCOCR_CONFIGS:
            if config.model_alias != model_alias:
                continue
            run_dir = model_root / config.run_name
            summary = generate_config(
                config=config,
                samples=selected_samples,
                processor=processor,
                model=model,
                run_dir=run_dir,
                output_root=args.output_root.resolve(),
                batch_size=batch_size,
                prompt=args.prompt,
                qwen_max_pixels=args.qwen_max_pixels if "qwen" in model_alias else None,
                resume=args.resume,
            )
            summaries.append(summary)
            write_json(model_root / "generation_summaries.json", summaries)
            if args.evaluate:
                run_id = f"{model_alias}_arabic_{config.run_name}_le1600_raw"
                eval_meta = evaluate_official(
                    official_root=mdpbench_root,
                    annotation_path=filtered_annotation,
                    prediction_dir=Path(summary["prediction_dir"]),
                    eval_dir=model_root / "official_eval" / config.run_name,
                    run_id=run_id,
                    python=args.python,
                )
                eval_summaries.append(eval_meta)
                write_json(model_root / "official_eval_summaries.json", eval_summaries)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output_root.resolve() / "runs" / "mdpbench" / "arabic_ccocr_top3_le1600_summaries.json", summaries)


if __name__ == "__main__":
    main()
