#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from io import BytesIO
import re
import shutil
from pathlib import Path
from typing import Any

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage

from parsing_neurons_repro.io import parse_csv, write_json


LANGUAGE_PATTERNS = {
    "Arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),
    # We require kana for Japanese to avoid confusing CJK-only Chinese text with Japanese.
    "Japanese": re.compile(r"[\u3040-\u30FF]"),
    "Korean": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]"),
}

SCRIPT_PATTERNS = {
    "Arabic": LANGUAGE_PATTERNS["Arabic"],
    "Japanese": LANGUAGE_PATTERNS["Japanese"],
    "Korean": LANGUAGE_PATTERNS["Korean"],
    "Chinese": re.compile(r"[\u4E00-\u9FFF]"),
    "Latin": re.compile(r"[A-Za-z]"),
}


def clean_text(text: Any) -> str:
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def infer_script(text: str) -> str:
    for script, pattern in SCRIPT_PATTERNS.items():
        if pattern.search(text):
            return script
    return "Other"


def text_matches_language(texts: list[str], language: str) -> bool:
    pattern = LANGUAGE_PATTERNS[language]
    return any(pattern.search(text) for text in texts)


def polygon_from_row(row: dict[str, Any], idx: int) -> list[float]:
    polygons = row.get("polygons") or []
    if idx < len(polygons) and len(polygons[idx]) >= 8:
        return [float(value) for value in polygons[idx][:8]]
    bboxes = row.get("bboxes") or []
    if idx < len(bboxes) and len(bboxes[idx]) >= 4:
        x1, y1, x2, y2 = [float(value) for value in bboxes[idx][:4]]
        return [x1, y1, x2, y1, x2, y2, x1, y2]
    return [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]


def format_gt_line(polygon: list[float], script: str, text: str) -> str:
    coords = [str(int(round(value))) for value in polygon[:8]]
    return ",".join([*coords, script, text])


def save_hf_image(image_value: Any, path: Path) -> None:
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            image = PILImage.open(BytesIO(image_value["bytes"])).convert("RGB")
        elif image_value.get("path") is not None:
            image = PILImage.open(image_value["path"]).convert("RGB")
        else:
            raise ValueError("HF image dictionary has neither bytes nor path.")
    else:
        image = image_value.convert("RGB")
    image.save(path, quality=95)


def existing_counts(output_root: Path, languages: list[str]) -> dict[str, int]:
    image_dir = output_root / "train_images"
    counts: dict[str, int] = {}
    for language in languages:
        counts[language] = len(list(image_dir.glob(f"{language.lower()}_*.jpg")))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a language-balanced ICDAR2019-MLT subset from the Hugging Face OCR-Data mirror."
    )
    parser.add_argument("--dataset", default="Yesianrohn/OCR-Data")
    parser.add_argument("--split", default="MLT2019")
    parser.add_argument("--output-root", type=Path, default=Path("data/icdar2019_mlt_hf"))
    parser.add_argument("--languages", default="Arabic,Japanese,Korean")
    parser.add_argument("--samples-per-language", type=int, default=200)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    languages = parse_csv(args.languages)
    unsupported = sorted(set(languages) - set(LANGUAGE_PATTERNS))
    if unsupported:
        raise ValueError(f"Unsupported languages for ICDAR2019 HF materialization: {unsupported}")

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    image_dir = args.output_root / "train_images"
    gt_dir = args.output_root / "train_gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    counts = existing_counts(args.output_root, languages)
    if all(counts[language] >= args.samples_per_language for language in languages):
        print(f"[icdar-hf] already complete: {counts}", flush=True)
        return

    dataset = load_dataset(args.dataset, split=args.split, streaming=True).cast_column("image", HFImage(decode=False))
    metadata: list[dict[str, Any]] = []
    scanned = 0
    for row_idx, row in enumerate(dataset):
        scanned = row_idx + 1
        raw_texts = row.get("texts") or []
        texts = [clean_text(text) for text in raw_texts]
        texts = [text for text in texts if text and text != "###"]
        if not texts:
            continue
        matching = [
            language
            for language in languages
            if counts[language] < args.samples_per_language and text_matches_language(texts, language)
        ]
        if not matching:
            if scanned % 500 == 0:
                print(f"[icdar-hf] scanned={scanned} counts={counts}", flush=True)
            if scanned >= args.max_scan:
                break
            continue

        gt_lines = []
        for text_idx, text in enumerate(texts):
            script = infer_script(text)
            gt_lines.append(format_gt_line(polygon_from_row(row, text_idx), script, text))
        gt_text = "\n".join(gt_lines) + "\n"

        for language in matching:
            sample_idx = counts[language]
            image_id = f"{language.lower()}_{sample_idx:04d}_hf{row_idx:06d}"
            image_path = image_dir / f"{image_id}.jpg"
            gt_path = gt_dir / f"gt_{image_id}.txt"
            save_hf_image(row["image"], image_path)
            gt_path.write_text(gt_text, encoding="utf-8")
            counts[language] += 1
            metadata.append(
                {
                    "language": language,
                    "image_id": image_id,
                    "hf_dataset": args.dataset,
                    "hf_split": args.split,
                    "hf_row_index": row_idx,
                    "num_text_regions": len(texts),
                    "matched_texts": [text for text in texts if LANGUAGE_PATTERNS[language].search(text)],
                }
            )
        print(f"[icdar-hf] scanned={scanned} counts={counts}", flush=True)
        if all(counts[language] >= args.samples_per_language for language in languages):
            break
        if scanned >= args.max_scan:
            break

    write_json(
        args.output_root / "metadata.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "languages": languages,
            "samples_per_language": args.samples_per_language,
            "counts": counts,
            "scanned_rows": scanned,
            "records": metadata,
        },
    )
    missing = {language: args.samples_per_language - counts[language] for language in languages if counts[language] < args.samples_per_language}
    if missing:
        raise RuntimeError(f"Could not materialize enough samples after scanning {scanned} rows: missing={missing}")
    print(f"[icdar-hf] complete: {counts}", flush=True)
    # The HF streaming reader can trigger a PyArrow finalization crash after
    # successful completion on this machine. Exiting directly keeps completed
    # materializations from being reported as failures.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
