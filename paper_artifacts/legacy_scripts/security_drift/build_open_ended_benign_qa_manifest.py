#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from security_drift_common import SAMPLE_ROOT, sha256_text, write_json, write_jsonl


SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an open-ended benign QA manifest from SQuAD v1.1 dev.")
    parser.add_argument("--output-jsonl", type=Path, default=SAMPLE_ROOT / "open_ended_benign_squad_v1_seed0_n100.jsonl")
    parser.add_argument("--cache-json", type=Path, default=SAMPLE_ROOT / "cache" / "squad_dev_v1.1.json")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_squad(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SQUAD_DEV_URL, timeout=120) as response:
            path.write_bytes(response.read())
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_squad(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_index = 0
    for article in payload.get("data", []):
        title = str(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", "")).strip()
            if not context:
                continue
            for qa in paragraph.get("qas", []):
                answers = [str(ans.get("text", "")).strip() for ans in qa.get("answers", []) if str(ans.get("text", "")).strip()]
                question = str(qa.get("question", "")).strip()
                if not question or not answers:
                    continue
                source_index += 1
                rows.append(
                    {
                        "source_index": source_index,
                        "source_id": str(qa.get("id", "")),
                        "title": title,
                        "context": context,
                        "question": question,
                        "answers": answers,
                    }
                )
    return rows


def make_prompt(question: str, context: str) -> str:
    return (
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer the question using a short phrase from the context. Do not explain."
    )


def main() -> None:
    args = parse_args()
    payload = load_squad(args.cache_json)
    examples = flatten_squad(payload)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    selected = examples[: args.n]
    rows: list[dict[str, Any]] = []
    for rank, example in enumerate(selected):
        prompt = make_prompt(example["question"], example["context"])
        rows.append(
            {
                "sample_id": f"benign_oeqa__squad_v1__{example['source_index']:05d}__{sha256_text(prompt)[:12]}",
                "trace_group": "benign_open_ended_qa",
                "benchmark": "squad_v1_open",
                "dataset_index": example["source_index"],
                "prompt": prompt,
                "instruction": prompt,
                "behavior": "benign open-ended question answering",
                "target": example["answers"][0],
                "prompt_sha256": sha256_text(prompt),
                "metadata": {
                    "dataset": "SQuAD v1.1 dev",
                    "source_url": SQUAD_DEV_URL,
                    "source_id": example["source_id"],
                    "source_index": example["source_index"],
                    "selection_seed": args.seed,
                    "selection_rank": rank,
                    "title": example["title"],
                    "question": example["question"],
                    "context": example["context"],
                    "answers": example["answers"],
                },
            }
        )
    write_jsonl(args.output_jsonl, rows)
    write_json(
        args.output_jsonl.with_suffix(".summary.json"),
        {
            "output_jsonl": str(args.output_jsonl),
            "source": "SQuAD v1.1 dev",
            "source_url": SQUAD_DEV_URL,
            "seed": args.seed,
            "n": len(rows),
            "benchmark": "squad_v1_open",
        },
    )
    print(f"[done] wrote {len(rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
