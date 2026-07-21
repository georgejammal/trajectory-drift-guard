#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset
from PIL import Image


SOURCE_ROOT = Path("/specific/scratches/scratch/georgejammal/dataset_sources")
DATA_ROOT = Path("/specific/scratches/scratch/georgejammal/trajectory-drift-guard/data")


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "trajectory-drift-guard"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_images(rows: list[dict], image_root: Path) -> None:
    jobs = {}
    for row in rows:
        path = image_root / row["image_name"]
        jobs[path] = row["image_url"]
        row["image_path"] = str(path)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download, url, path): path for path, url in jobs.items()}
        for future in as_completed(futures):
            future.result()
    for row in rows:
        with Image.open(row["image_path"]) as image:
            image.verify()


def save_dataset(name: str, rows: list[dict], labels: dict[int, int]) -> None:
    output = DATA_ROOT / name
    if output.exists():
        shutil.rmtree(output)
    fields = [{key: row[key] for key in ("image_path", "question", "number")} for row in rows]
    Dataset.from_list(fields).save_to_disk(str(output))
    (output / "balance_metadata.json").write_text(
        json.dumps({"labels": labels, "total": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(name, len(rows), dict(sorted(Counter(row["number"] for row in rows).items())), flush=True)


def build_tallyqa() -> None:
    rows = json.loads((SOURCE_ROOT / "tallyqa/test.json").read_text())
    selected = []
    counts = defaultdict(int)
    for row in rows:
        number = int(row["answer"])
        if 1 <= number <= 9 and counts[number] < 50:
            image_name = row["image"]
            base = "https://cs.stanford.edu/people/rak248"
            selected.append(
                {
                    "image_name": image_name,
                    "image_url": f"{base}/{image_name}",
                    "question": row["question"],
                    "number": number,
                }
            )
            counts[number] += 1
    if dict(counts) != {number: 50 for number in range(1, 10)}:
        raise RuntimeError(f"Unexpected TallyQA balance: {dict(counts)}")
    download_images(selected, DATA_ROOT / "_images_tallyqa")
    save_dataset("tallyqa_balanced", selected, dict(counts))


def build_howmany() -> None:
    source = SOURCE_ROOT / "howmanyqa/HowMany-QA/question_ids.json"
    questions = json.loads((SOURCE_ROOT / "vqa/v2_OpenEnded_mscoco_val2014_questions.json").read_text())["questions"]
    annotations = json.loads((SOURCE_ROOT / "vqa/v2_mscoco_val2014_annotations.json").read_text())["annotations"]
    question_by_id = {row["question_id"]: row for row in questions}
    annotation_by_id = {row["question_id"]: row for row in annotations}
    target = {number: 50 for number in range(1, 9)} | {9: 40}
    counts = defaultdict(int)
    selected = []
    for question_id in json.loads(source.read_text())["test"]:
        annotation = annotation_by_id[question_id]
        numeric_answers = [
            int(match.group(1))
            for answer in annotation["answers"]
            if (match := re.fullmatch(r"\s*(\d+)\s*", answer["answer"]))
        ]
        if not numeric_answers:
            continue
        frequencies = Counter(numeric_answers)
        number = min(key for key, value in frequencies.items() if value == max(frequencies.values()))
        if number not in target or counts[number] >= target[number]:
            continue
        image_id = int(annotation["image_id"])
        image_name = f"COCO_val2014_{image_id:012d}.jpg"
        selected.append(
            {
                "image_name": image_name,
                "image_url": f"http://images.cocodataset.org/val2014/{image_name}",
                "question": question_by_id[question_id]["question"],
                "number": number,
            }
        )
        counts[number] += 1
    if dict(counts) != target:
        raise RuntimeError(f"Unexpected HowMany-QA balance: {dict(counts)}")
    download_images(selected, DATA_ROOT / "_images_how_many")
    save_dataset("how_many", selected, dict(counts))


if __name__ == "__main__":
    build_tallyqa()
    build_howmany()
