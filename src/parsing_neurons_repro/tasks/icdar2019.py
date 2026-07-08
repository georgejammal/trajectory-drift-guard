from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Write the text you see in the image. Do not add any further explanation."

SCRIPT_BY_LANGUAGE = {
    "Arabic": "Arabic",
    "Japanese": "Japanese",
    "Korean": "Korean",
}


@dataclass(frozen=True)
class ICDARSample:
    language: str
    image_path: Path
    gt_path: Path
    image_id: str
    gold_text: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(ca != cb),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_similarity(prediction: str, gold: str) -> float:
    pred = normalize_text(prediction)
    target = normalize_text(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    return max(0.0, 1.0 - levenshtein(pred, target) / max(len(pred), len(target)))


def parse_gt_line(line: str) -> tuple[str | None, str | None]:
    line = line.rstrip("\n\r")
    if not line:
        return None, None
    parts = line.split(",", 9)
    if len(parts) < 10:
        return None, None
    script = parts[8].strip().strip('"')
    transcription = parts[9].strip().strip('"')
    if not transcription or transcription == "###":
        return None, None
    return script, transcription


def gt_transcriptions(gt_path: Path, *, script: str | None = None) -> list[str]:
    texts: list[str] = []
    with gt_path.open(encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            row_script, transcription = parse_gt_line(line)
            if row_script is None or transcription is None:
                continue
            if script is not None and row_script.lower() != script.lower():
                continue
            texts.append(transcription)
    return texts


def discover_dirs(root: Path) -> tuple[Path, Path]:
    image_candidates = [
        root / "train_images",
        root / "train" / "images",
        root / "images",
        root / "ch15_training_images",
        root / "ch15_training",
    ]
    gt_candidates = [
        root / "train_gt",
        root / "train" / "gt",
        root / "gt",
        root / "ch15_training_gt",
        root / "ch15_training_localization_transcription_gt",
    ]
    image_dir = next((path for path in image_candidates if path.exists()), None)
    gt_dir = next((path for path in gt_candidates if path.exists()), None)
    if image_dir is None:
        found = [path for path in root.rglob("*") if path.is_dir() and any(path.glob("*.jpg"))]
        image_dir = found[0] if found else None
    if gt_dir is None:
        found = [
            path
            for path in root.rglob("*")
            if path.is_dir() and any(child.name.lower().startswith("gt_") for child in path.glob("*.txt"))
        ]
        gt_dir = found[0] if found else None
    if image_dir is None or gt_dir is None:
        raise FileNotFoundError(
            "Could not discover ICDAR2019 image/GT directories. Expected a layout like "
            "data/icdar2019_mlt/train_images and data/icdar2019_mlt/train_gt."
        )
    return image_dir, gt_dir


def gt_for_image(gt_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = [
        gt_dir / f"gt_{stem}.txt",
        gt_dir / f"{stem}.txt",
    ]
    if stem.startswith("img_"):
        candidates.append(gt_dir / f"gt_{stem[4:]}.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(gt_dir.rglob(f"*{stem}*.txt"))
    return matches[0] if matches else None


def load_language_samples(
    *,
    root: Path,
    language: str,
    limit: int = 100,
    seed: int = 0,
    script_only_gold: bool = False,
) -> list[ICDARSample]:
    if language not in SCRIPT_BY_LANGUAGE:
        raise ValueError(f"ICDAR2019 does not cover {language}. Available: {sorted(SCRIPT_BY_LANGUAGE)}")
    script = SCRIPT_BY_LANGUAGE[language]
    image_dir, gt_dir = discover_dirs(root)
    candidates: list[ICDARSample] = []
    image_paths = sorted(
        [
            *image_dir.rglob("*.jpg"),
            *image_dir.rglob("*.jpeg"),
            *image_dir.rglob("*.png"),
        ]
    )
    for image_path in image_paths:
        gt_path = gt_for_image(gt_dir, image_path)
        if gt_path is None:
            continue
        script_texts = gt_transcriptions(gt_path, script=script)
        if not script_texts:
            continue
        all_texts = gt_transcriptions(gt_path, script=None)
        gold_texts = script_texts if script_only_gold else all_texts
        if not gold_texts:
            continue
        candidates.append(
            ICDARSample(
                language=language,
                image_path=image_path,
                gt_path=gt_path,
                image_id=image_path.stem,
                gold_text="\n".join(gold_texts),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:limit]


def sample_to_json(sample: ICDARSample) -> dict[str, Any]:
    return {
        "language": sample.language,
        "image_path": str(sample.image_path),
        "gt_path": str(sample.gt_path),
        "image_id": sample.image_id,
        "gold_text": sample.gold_text,
    }

