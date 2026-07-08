#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select CLAS bridge layers from all-layer category statistics.")
    parser.add_argument("stats_json", type=Path)
    parser.add_argument("--exclude-final", type=int, default=2)
    parser.add_argument(
        "--min-start-frac",
        type=float,
        default=0.70,
        help="Only consider layers at or after this fraction of total depth, matching CLAS's late bridge-layer choice.",
    )
    parser.add_argument(
        "--threshold-frac",
        type=float,
        default=0.90,
        help="Keep layers whose bridge score is at least this fraction of the eligible maximum.",
    )
    return parser.parse_args()


def layer_scores(payload: dict) -> list[tuple[int, float, float, float, float, float]]:
    rows = []
    for layer, counts in sorted(payload["summary_by_layer"].items(), key=lambda item: int(item[0])):
        total = max(1, sum(int(value) for value in counts.values()))
        partial = 100.0 * int(counts.get("partial_shared", 0)) / total
        dead = 100.0 * int(counts.get("dead", 0)) / total
        lang = 100.0 * int(counts.get("language_specific", 0)) / total
        all_shared = 100.0 * int(counts.get("all_shared", 0)) / total
        score = partial - dead - lang
        rows.append((int(layer), score, partial, dead, lang, all_shared))
    return rows


def longest_contiguous(layers: list[int]) -> list[int]:
    if not layers:
        return []
    runs: list[list[int]] = []
    current = [layers[0]]
    for layer in layers[1:]:
        if layer == current[-1] + 1:
            current.append(layer)
        else:
            runs.append(current)
            current = [layer]
    runs.append(current)
    return max(runs, key=lambda run: (len(run), run[-1]))


def main() -> None:
    args = parse_args()
    payload = json.loads(args.stats_json.read_text(encoding="utf-8"))
    scores = layer_scores(payload)
    if not scores:
        raise RuntimeError(f"No layer summary found in {args.stats_json}.")
    num_layers = max(layer for layer, *_ in scores) + 1
    max_layer = num_layers - 1 - args.exclude_final
    min_layer = int(round(args.min_start_frac * (num_layers - 1)))
    eligible = [row for row in scores if min_layer <= row[0] <= max_layer]
    if not eligible:
        raise RuntimeError("No eligible layers after applying bridge-window constraints.")
    max_score = max(row[1] for row in eligible)
    selected_layers = [row[0] for row in eligible if row[1] >= args.threshold_frac * max_score]
    bridge = longest_contiguous(selected_layers)
    if not bridge:
        bridge = [max(eligible, key=lambda row: row[1])[0]]
    print(f"{bridge[0]}-{bridge[-1]}")
    print(
        json.dumps(
            {
                "stats_json": str(args.stats_json),
                "num_layers": num_layers,
                "exclude_final": args.exclude_final,
                "min_layer": min_layer,
                "max_layer": max_layer,
                "threshold_frac": args.threshold_frac,
                "bridge_window": f"{bridge[0]}-{bridge[-1]}",
                "selected_layers": bridge,
                "eligible_scores": [
                    {
                        "layer": layer,
                        "score": score,
                        "partial_shared": partial,
                        "dead": dead,
                        "language_specific": lang,
                        "all_shared": all_shared,
                    }
                    for layer, score, partial, dead, lang, all_shared in eligible
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
