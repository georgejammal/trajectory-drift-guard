from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image

from ..generation import generate_batch_adaptive
from ..interventions import build_abs_intervention
from ..io import write_json


DATA_ROOT = Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", "data"))
DEFAULT_COUNTING_DATASETS = {
    "countbenchqa": str(DATA_ROOT / "countbenchqa"),
    "how_many": str(DATA_ROOT / "how_many"),
    "tallyqa_balanced": str(DATA_ROOT / "tallyqa_balanced"),
}

NUMBER_WORDS_EN = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

NUMBER_WORDS_ZH = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def as_dataset(payload: Any, split: str) -> Dataset:
    if isinstance(payload, DatasetDict):
        return payload[split]
    return payload


def row_question(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text", "prompt"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find question field in row keys: {sorted(row)}")


def row_image(row: dict[str, Any]) -> Image.Image | Path:
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if image is not None and hasattr(image, "convert"):
        return image.convert("RGB")
    for key in ("image_path", "path", "filename", "file_name"):
        value = row.get(key)
        if value:
            return Path(value)
    raise KeyError(f"Could not find image field in row keys: {sorted(row)}")


def parse_number_text(text: str) -> tuple[int | None, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    digit_match = re.search(r"\b([0-9]+)\b", lowered)
    if digit_match:
        return int(digit_match.group(1)), "arabic_digit"
    for word, value in NUMBER_WORDS_EN.items():
        if re.search(rf"\b{word}\b", lowered):
            return value, "english_word"
    for word, value in NUMBER_WORDS_ZH.items():
        if word in stripped:
            return value, "chinese_word"
    return None, "unparsed"


def gold_from_row(row: dict[str, Any]) -> int:
    for key in ("number", "gold", "answer", "label", "count"):
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                parsed, _ = parse_number_text(value)
                if parsed is not None:
                    return parsed
            return int(value)
    raise KeyError(f"Could not find gold number field in row keys: {sorted(row)}")


def starts_with_digit_answer(text: str, gold: int) -> bool:
    return re.match(rf"^\s*{gold}\b", text.strip()) is not None


def load_filtered_dataset(
    path: str | Path,
    *,
    split: str,
    gold_numbers: list[int],
    limit: int | None,
) -> tuple[Dataset, list[int]]:
    dataset = as_dataset(load_from_disk(str(path)), split)
    gold_set = set(gold_numbers)
    indices = [idx for idx, row in enumerate(dataset) if gold_from_row(row) in gold_set]
    if limit is not None:
        indices = indices[:limit]
    return dataset.select(indices), indices


def evaluate_counting_dataset(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    dataset_name: str,
    dataset_path: str | Path,
    output_dir: Path,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    split: str = "test",
    gold_numbers: list[int] | None = None,
    suffix: str = " Answer with one number do not add any further explanations.",
    question_prefix: str = "",
    batch_size: int = 64,
    max_new_tokens: int = 16,
    limit: int | None = None,
    qwen_max_pixels: int | None = 401408,
    qwen_min_pixels: int | None = None,
    allowed_token_ids: list[int] | set[int] | None = None,
    decoding_constraint: str | None = None,
    scalar_mode: str = "abs",
    adaptive_oom_split: bool = True,
) -> dict[str, Any]:
    if gold_numbers is None:
        gold_numbers = list(range(1, 10))
    dataset, source_indices = load_filtered_dataset(
        dataset_path,
        split=split,
        gold_numbers=gold_numbers,
        limit=limit,
    )
    condition_dir = output_dir / dataset_name / component_mode
    condition_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = condition_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    total = correct = parsed = strict_digit_correct = 0
    prediction_histogram: Counter[str] = Counter()
    output_form_histogram: Counter[str] = Counter()
    per_label: dict[int, Counter[str]] = defaultdict(Counter)
    start = time.time()

    indexed_rows = list(zip(source_indices, dataset))
    context = build_abs_intervention(
        model=model,
        component_mode=component_mode,
        mlp_selection=mlp_selection,
        attn_selection=attn_selection,
        mlp_token_scope="all_positions",
        attn_token_scope="last_position",
        scalar_mode=scalar_mode,
    )
    with predictions_path.open("w", encoding="utf-8") as handle:
        with context:
            for batch_idx, batch in enumerate(batched(indexed_rows, batch_size), start=1):
                rows = [row for _, row in batch]
                questions = [row_question(row) for row in rows]
                images = [row_image(row) for row in rows]
                decoded = generate_batch_adaptive(
                    model_alias=model_alias,
                    processor=processor,
                    model=model,
                    images=images,
                    questions=questions,
                    suffix=suffix,
                    question_prefix=question_prefix,
                    max_new_tokens=max_new_tokens,
                    qwen_max_pixels=qwen_max_pixels,
                    qwen_min_pixels=qwen_min_pixels,
                    allowed_token_ids=allowed_token_ids,
                    adaptive_oom_split=adaptive_oom_split,
                )
                for (source_index, row), prediction_text in zip(batch, decoded):
                    gold = gold_from_row(row)
                    prediction_number, output_form = parse_number_text(prediction_text)
                    is_correct = prediction_number == gold
                    is_strict_digit_correct = is_correct and starts_with_digit_answer(prediction_text, gold)
                    total += 1
                    correct += int(is_correct)
                    parsed += int(prediction_number is not None)
                    strict_digit_correct += int(is_strict_digit_correct)
                    prediction_histogram[str(prediction_number) if prediction_number is not None else "unparsed"] += 1
                    output_form_histogram[output_form] += 1
                    per_label[gold]["samples"] += 1
                    per_label[gold]["correct"] += int(is_correct)
                    per_label[gold]["parsed"] += int(prediction_number is not None)
                    per_label[gold]["strict_digit_correct"] += int(is_strict_digit_correct)
                    handle.write(
                        json.dumps(
                            {
                                "source_index": int(source_index),
                                "question": row_question(row),
                                "gold": int(gold),
                                "prediction_text": prediction_text,
                                "prediction_number": prediction_number,
                                "output_form": output_form,
                                "correct": is_correct,
                                "strict_digit_correct": is_strict_digit_correct,
                                "component_mode": component_mode,
                                "scalar_mode": scalar_mode,
                                "decoding_constraint": decoding_constraint,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(
                    f"[counting:{dataset_name}:{component_mode}] batch={batch_idx} "
                    f"seen={total}/{len(dataset)} acc={correct / total if total else 0:.4f}",
                    flush=True,
                )

    metrics = {
        "task": "counting",
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "split": split,
        "gold_numbers": gold_numbers,
        "component_mode": component_mode,
        "scalar_mode": scalar_mode,
        "decoding_constraint": decoding_constraint,
        "allowed_token_ids": sorted(int(token_id) for token_id in allowed_token_ids)
        if allowed_token_ids is not None
        else None,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parsed": parsed,
        "parsed_rate": parsed / total if total else 0.0,
        "strict_digit_correct": strict_digit_correct,
        "strict_digit_accuracy": strict_digit_correct / total if total else 0.0,
        "prediction_histogram": dict(prediction_histogram),
        "output_form_histogram": dict(output_form_histogram),
        "per_label_metrics": {
            str(label): {
                "samples": counts["samples"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["samples"] if counts["samples"] else 0.0,
                "parsed": counts["parsed"],
                "parsed_rate": counts["parsed"] / counts["samples"] if counts["samples"] else 0.0,
                "strict_digit_correct": counts["strict_digit_correct"],
                "strict_digit_accuracy": counts["strict_digit_correct"] / counts["samples"] if counts["samples"] else 0.0,
            }
            for label, counts in sorted(per_label.items())
        },
        "predictions_path": str(predictions_path),
        "elapsed_seconds": time.time() - start,
    }
    write_json(condition_dir / "metrics.json", metrics)
    return metrics


def evaluate_counting_suite(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    output_dir: Path,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    datasets: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    datasets = datasets or DEFAULT_COUNTING_DATASETS
    results = []
    for name, path in datasets.items():
        results.append(
            evaluate_counting_dataset(
                model_alias=model_alias,
                processor=processor,
                model=model,
                dataset_name=name,
                dataset_path=path,
                output_dir=output_dir,
                component_mode=component_mode,
                mlp_selection=mlp_selection,
                attn_selection=attn_selection,
                **kwargs,
            )
        )
    macro = sum(item["accuracy"] for item in results) / len(results) if results else 0.0
    summary = {"task": "counting", "component_mode": component_mode, "macro_accuracy": macro, "datasets": results}
    summary["scalar_mode"] = kwargs.get("scalar_mode", "abs")
    write_json(output_dir / component_mode / "suite_metrics.json", summary)
    return summary
