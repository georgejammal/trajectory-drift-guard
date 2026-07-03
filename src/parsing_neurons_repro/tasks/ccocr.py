from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..generation import generate_batch_adaptive
from ..interventions import build_abs_intervention
from ..io import write_json


DATA_ROOT = Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"))
DEFAULT_CC_OCR_ROOT = DATA_ROOT / "cc_ocr_dataset"
DEFAULT_CC_OCR_INDEX = DEFAULT_CC_OCR_ROOT / "index" / "multi_lan_ocr.json"
DEFAULT_LANGUAGES = ["Arabic", "Japanese", "Korean", "Russian"]


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_index(index_path: Path, languages: list[str] | None = None) -> list[dict[str, Any]]:
    selected_names = set(languages or [])
    data_info = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for row in data_info:
        if not row.get("release", True):
            continue
        if selected_names and row["dataset"] not in selected_names:
            continue
        rows.append(row)
    return rows


def load_labels(dataset_base_dir: Path) -> dict[str, str]:
    return {str(key): str(value) for key, value in json.loads((dataset_base_dir / "label.json").read_text()).items()}


def label_token_filter(
    *,
    samples: list[dict[str, Any]],
    labels: dict[str, str],
    processor: Any,
    max_gold_tokens_exclusive: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_gold_tokens_exclusive is None:
        return samples, {"max_gold_tokens_exclusive": None, "kept": len(samples), "excluded": 0}
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("max_gold_tokens_exclusive requires processor.tokenizer.")
    kept = []
    excluded = 0
    for sample in samples:
        gold = labels.get(sample["image_name"])
        if gold is None:
            excluded += 1
            continue
        token_count = len(tokenizer.encode(gold, add_special_tokens=False))
        sample["gold_token_count"] = token_count
        if token_count < max_gold_tokens_exclusive:
            kept.append(sample)
        else:
            excluded += 1
    return kept, {"max_gold_tokens_exclusive": max_gold_tokens_exclusive, "kept": len(kept), "excluded": excluded}


def load_samples(dataset_base_dir: Path, output_dir: Path, *, limit: int | None, resume: bool) -> list[dict[str, Any]]:
    samples = []
    with (dataset_base_dir / "qa.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_url = row["url"]
            image_name = os.path.basename(unquote(urlparse(image_url).path))
            output_path = output_dir / f"{image_name}.json"
            if resume and output_path.exists():
                continue
            samples.append(
                {
                    "image_path": dataset_base_dir / image_url,
                    "image_name": image_name,
                    "question": row["prompt"],
                    "output_path": output_path,
                }
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def write_prediction(sample: dict[str, Any], prediction_text: str, model_alias: str, elapsed_seconds: float) -> None:
    sample["output_path"].parent.mkdir(parents=True, exist_ok=True)
    record = {
        "image": str(sample["image_path"]),
        "question": sample["question"],
        "model_name": f"local_{model_alias}",
        "response": prediction_text,
        "time": time.time(),
        "elapsed_seconds": elapsed_seconds,
    }
    sample["output_path"].write_text(json.dumps(record, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def evaluate_with_vendored_cc_ocr(index_path: Path, exp_dir: Path, cc_ocr_root: Path) -> Path:
    evaluation_dir = cc_ocr_root / "evaluation"
    sys.path.insert(0, str(evaluation_dir))
    from main import evaluate_and_summary  # type: ignore

    return Path(evaluate_and_summary(str(index_path), str(exp_dir)))


def evaluate_ccocr_language(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    data_info: dict[str, Any],
    cc_ocr_root: Path,
    output_dir: Path,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    batch_size: int = 64,
    max_new_tokens: int = 512,
    limit: int | None = None,
    max_gold_tokens_exclusive: int | None = None,
    qwen_max_pixels: int | None = 1003520,
    qwen_min_pixels: int | None = None,
    adaptive_oom_split: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    dataset_name = data_info["dataset"]
    dataset_base_dir = cc_ocr_root / data_info["base_dir"]
    language_out = output_dir / dataset_name
    labels = load_labels(dataset_base_dir)
    samples = load_samples(dataset_base_dir, language_out, limit=limit, resume=resume)
    samples, filter_info = label_token_filter(
        samples=samples,
        labels=labels,
        processor=processor,
        max_gold_tokens_exclusive=max_gold_tokens_exclusive,
    )

    context = build_abs_intervention(
        model=model,
        component_mode=component_mode,
        mlp_selection=mlp_selection,
        attn_selection=attn_selection,
        mlp_token_scope="all_positions",
        attn_token_scope="last_position",
    )
    total = 0
    start = time.time()
    with context:
        for batch_idx, batch in enumerate(batched(samples, batch_size), start=1):
            batch_start = time.time()
            predictions = generate_batch_adaptive(
                model_alias=model_alias,
                processor=processor,
                model=model,
                images=[sample["image_path"] for sample in batch],
                questions=[sample["question"] for sample in batch],
                max_new_tokens=max_new_tokens,
                qwen_max_pixels=qwen_max_pixels,
                qwen_min_pixels=qwen_min_pixels,
                adaptive_oom_split=adaptive_oom_split,
            )
            elapsed = time.time() - batch_start
            for sample, prediction in zip(batch, predictions):
                write_prediction(sample, prediction, model_alias, elapsed)
                total += 1
            print(
                f"[ccocr:{dataset_name}:{component_mode}] batch={batch_idx} "
                f"written={total}/{len(samples)}",
                flush=True,
            )

    summary = {
        "task": "ccocr",
        "language": dataset_name,
        "component_mode": component_mode,
        "queued": len(samples),
        "written_this_run": total,
        "expected": data_info["num"] if limit is None else min(limit, data_info["num"]),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "gold_token_filter": filter_info,
        "elapsed_seconds": time.time() - start,
    }
    write_json(language_out / "run_summary.json", summary)
    return summary


def evaluate_ccocr_suite(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    output_dir: Path,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    cc_ocr_root: Path = DEFAULT_CC_OCR_ROOT,
    index_path: Path = DEFAULT_CC_OCR_INDEX,
    languages: list[str] | None = None,
    evaluate: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_info = load_index(index_path, languages or DEFAULT_LANGUAGES)
    active_index_path = index_path
    if languages:
        active_index_path = output_dir / "active_index.json"
        write_json(active_index_path, data_info)
    run_summaries = []
    for row in data_info:
        run_summaries.append(
            evaluate_ccocr_language(
                model_alias=model_alias,
                processor=processor,
                model=model,
                data_info=row,
                cc_ocr_root=cc_ocr_root,
                output_dir=output_dir,
                component_mode=component_mode,
                mlp_selection=mlp_selection,
                attn_selection=attn_selection,
                **kwargs,
            )
        )
    summary = {
        "task": "ccocr",
        "component_mode": component_mode,
        "languages": [row["dataset"] for row in data_info],
        "datasets": run_summaries,
    }
    write_json(output_dir / "run_summary.json", summary)
    if evaluate:
        summary_path = evaluate_with_vendored_cc_ocr(active_index_path, output_dir, cc_ocr_root)
        summary["cc_ocr_summary_path"] = str(summary_path)
        write_json(output_dir / "run_summary.json", summary)
    return summary
