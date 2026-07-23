#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset, load_from_disk
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/specific/scratches/scratch/georgejammal/dataset_sources")
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--per-label", type=int, default=20)
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "trajectory-drift-guard"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_images(rows: list[dict], image_root: Path) -> None:
    jobs = {image_root / row["image_name"]: row["image_url"] for row in rows}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download, url, path): path for path, url in jobs.items()}
        for future in as_completed(futures):
            future.result()
    for row in rows:
        path = image_root / row["image_name"]
        with Image.open(path) as image:
            image.verify()
        row["image_path"] = str(path)


def existing_test_images(dataset_name: str) -> set[str]:
    dataset = load_from_disk(str(DATA_ROOT / dataset_name))
    rows = dataset["test"] if hasattr(dataset, "keys") else dataset
    return {Path(row["image_path"]).name for row in rows}


def select_rows(candidates: list[dict], targets: dict[int, int], seed: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        grouped[row["number"]].append(row)
    randomizer = random.Random(seed)
    for rows in grouped.values():
        randomizer.shuffle(rows)

    selected: list[dict] = []
    used_images: set[str] = set()
    for label in sorted(targets, key=lambda item: len(grouped[item])):
        quota = targets[label]
        for row in grouped[label]:
            if row["image_name"] in used_images:
                continue
            selected.append(row)
            used_images.add(row["image_name"])
            if sum(item["number"] == label for item in selected) == quota:
                break
        if sum(item["number"] == label for item in selected) != quota:
            raise RuntimeError(f"Could not select {quota} image-disjoint rows for label {label}.")
    return selected


def save_dataset(name: str, rows: list[dict], metadata: dict) -> None:
    output = DATA_ROOT / name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation dataset: {output}")
    fields = [
        {
            "image_path": row["image_path"],
            "question": row["question"],
            "number": row["number"],
            "source_question_id": row["source_question_id"],
            "source_image_id": row["image_name"],
        }
        for row in rows
    ]
    Dataset.from_list(fields).save_to_disk(str(output))
    (output / "validation_metadata.json").write_text(
        json.dumps(metadata | {"labels": dict(sorted(Counter(row["number"] for row in rows).items())), "total": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )


def howmany_candidates() -> list[dict]:
    test_images = existing_test_images("how_many")
    questions = json.loads((SOURCE_ROOT / "vqa/v2_OpenEnded_mscoco_val2014_questions.json").read_text())["questions"]
    annotations = json.loads((SOURCE_ROOT / "vqa/v2_mscoco_val2014_annotations.json").read_text())["annotations"]
    questions_by_id = {row["question_id"]: row for row in questions}
    annotations_by_id = {row["question_id"]: row for row in annotations}
    candidates = []
    for question_id in json.loads((SOURCE_ROOT / "howmanyqa/HowMany-QA/question_ids.json").read_text())["test"]:
        annotation = annotations_by_id[question_id]
        numeric = [
            int(match.group(1))
            for answer in annotation["answers"]
            if (match := re.fullmatch(r"\s*(\d+)\s*", answer["answer"]))
        ]
        if not numeric:
            continue
        frequencies = Counter(numeric)
        label = min(key for key, value in frequencies.items() if value == max(frequencies.values()))
        image_name = f"COCO_val2014_{int(annotation['image_id']):012d}.jpg"
        if label not in range(1, 9) or image_name in test_images:
            continue
        candidates.append(
            {
                "image_name": image_name,
                "image_url": f"http://images.cocodataset.org/val2014/{image_name}",
                "question": questions_by_id[question_id]["question"],
                "number": label,
                "source_question_id": int(question_id),
            }
        )
    return candidates


def tally_candidates() -> list[dict]:
    test_images = existing_test_images("tallyqa_balanced")
    candidates = []
    for source_index, row in enumerate(json.loads((SOURCE_ROOT / "tallyqa/test.json").read_text())):
        label = int(row["answer"])
        image_name = row["image"]
        if label not in range(1, 10) or Path(image_name).name in test_images:
            continue
        candidates.append(
            {
                "image_name": image_name,
                "image_url": f"https://cs.stanford.edu/people/rak248/{image_name}",
                "question": row["question"],
                "number": label,
                "source_question_id": source_index,
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    howmany_targets = {label: args.per_label for label in range(1, 9)}
    tally_targets = {label: args.per_label for label in range(1, 10)}
    howmany = select_rows(howmany_candidates(), howmany_targets, args.seed)
    tally = select_rows(tally_candidates(), tally_targets, args.seed + 1)
    download_images(howmany, DATA_ROOT / "_images_how_many_validation")
    download_images(tally, DATA_ROOT / "_images_tallyqa_validation")
    save_dataset(
        "how_many_validation",
        howmany,
        {"seed": args.seed, "per_label": args.per_label, "source_split": "test", "excluded_test_dataset": "how_many"},
    )
    save_dataset(
        "tallyqa_validation",
        tally,
        {"seed": args.seed + 1, "per_label": args.per_label, "source_split": "test", "excluded_test_dataset": "tallyqa_balanced"},
    )
    print("how_many_validation", len(howmany), dict(sorted(Counter(row["number"] for row in howmany).items())))
    print("tallyqa_validation", len(tally), dict(sorted(Counter(row["number"] for row in tally).items())))


if __name__ == "__main__":
    main()
