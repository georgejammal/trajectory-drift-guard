#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOCAL_DATA_ROOT = Path("/home/georgejammal/projects/parsing_neurons/data")
if "PARSING_NEURONS_DATA_ROOT" not in os.environ and LOCAL_DATA_ROOT.exists():
    os.environ["PARSING_NEURONS_DATA_ROOT"] = str(LOCAL_DATA_ROOT)

from parsing_neurons_repro.directions import flores_direction  # noqa: E402
from parsing_neurons_repro.interventions import ResidualAddIntervention  # noqa: E402
from parsing_neurons_repro.models import (  # noqa: E402
    MODEL_PATHS,
    load_processor,
    load_tensor,
    load_tokenizer,
    load_vlm,
    model_spec,
    weight_map,
    weight_templates,
)
from parsing_neurons_repro.tasks.ccocr import DEFAULT_CC_OCR_INDEX, DEFAULT_CC_OCR_ROOT, evaluate_ccocr_suite  # noqa: E402


LOCAL_MODEL_PATHS = {
    "gemma3_4b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma3_12b_it": "/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "qwen3_vl_8b_instruct": "/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct",
}

DEFAULT_LAYERS = {
    "gemma3_4b_it": [17, 23, 29],
    "gemma3_12b_it": [24, 34, 43],
    "qwen3_vl_8b_instruct": [18, 24, 29],
}
DEFAULT_BATCH_SIZES = {
    "gemma3_4b_it": 64,
    "gemma3_12b_it": 16,
    "qwen3_vl_8b_instruct": 16,
}


def resolve_local_model_paths() -> None:
    for alias, path in LOCAL_MODEL_PATHS.items():
        if Path(path).exists():
            MODEL_PATHS[alias] = path


def load_embedding(model_path: Path) -> torch.Tensor:
    index = weight_map(model_path)
    templates = weight_templates(index)
    return load_tensor(model_path, index, templates["embed"])


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_layers(raw: str | None, model_alias: str) -> list[int]:
    if raw is None or raw.strip() == "":
        return DEFAULT_LAYERS[model_alias]
    return [int(part) for part in parse_csv(raw)]


def clean_alpha(alpha: float) -> str:
    return ("%g" % alpha).replace("-", "m").replace(".", "p")


def flores_pairs_path(flores_root: Path, language: str) -> Path:
    return flores_root / f"en_to_{language.lower()}.json"


def parse_args() -> argparse.Namespace:
    default_flores_root = (
        Path(os.environ.get("PARSING_NEURONS_DATA_ROOT", str(LOCAL_DATA_ROOT)))
        / "flores_transfer_pairs"
        / "flores101_en_to_cc_ocr_languages_random500"
        / "pairs_by_language"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gemma3_4b_it,gemma3_12b_it,qwen3_vl_8b_instruct")
    parser.add_argument("--languages", default="Arabic,Japanese,Korean,Russian")
    parser.add_argument("--layers", default=None, help="Optional comma-separated layer override for all models.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--token-scope", choices=["last_position", "all_positions"], default="last_position")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--flores-root", type=Path, default=default_flores_root)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-gold-tokens-exclusive", type=int, default=500)
    parser.add_argument("--qwen-max-pixels", type=int, default=1003520)
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_local_model_paths()
    models = parse_csv(args.models)
    languages = parse_csv(args.languages)
    summaries: list[dict[str, Any]] = []

    for model_alias in models:
        spec = model_spec(model_alias)
        tokenizer = load_tokenizer(spec.path)
        embedding = load_embedding(spec.path)
        processor = load_processor(spec.path)
        model = load_vlm(model_alias, spec.path, dtype=args.dtype)
        layers = parse_layers(args.layers, model_alias)
        batch_size = DEFAULT_BATCH_SIZES.get(model_alias, 16)
        print(
            f"[ccocr-resid] model={model_alias} layers={layers} alpha={args.alpha} "
            f"token_scope={args.token_scope} batch={batch_size}",
            flush=True,
        )

        for language in languages:
            direction, metadata = flores_direction(
                tokenizer,
                embedding,
                pairs_path=flores_pairs_path(args.flores_root, language),
                source_field="english",
                target_field="target",
                max_pairs=500,
            )
            direction_slug = f"flores_en-to-{language.lower()}"
            for layer in layers:
                run_dir = (
                    args.output_root
                    / "runs"
                    / "ccocr_residual_add"
                    / model_alias
                    / direction_slug
                    / f"resid_alpha{clean_alpha(args.alpha)}_layer{layer}_{args.token_scope}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "residual_add_metadata.json").write_text(
                    json.dumps(
                        {
                            "model_alias": model_alias,
                            "language": language,
                            "layer": layer,
                            "alpha": args.alpha,
                            "token_scope": args.token_scope,
                            "direction_metadata": metadata,
                            "intervention": "hidden_states <- hidden_states + alpha * d at layer input",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[ccocr-resid:{model_alias}:{language}:layer{layer}] start", flush=True)
                with ResidualAddIntervention(
                    model=model,
                    layers=[layer],
                    direction=direction,
                    alpha=args.alpha,
                    token_scope=args.token_scope,
                ):
                    summary = evaluate_ccocr_suite(
                        model_alias=model_alias,
                        processor=processor,
                        model=model,
                        output_dir=run_dir,
                        component_mode="baseline",
                        mlp_selection=None,
                        attn_selection=None,
                        cc_ocr_root=DEFAULT_CC_OCR_ROOT,
                        index_path=DEFAULT_CC_OCR_INDEX,
                        languages=[language],
                        batch_size=batch_size,
                        limit=args.limit,
                        max_new_tokens=args.max_new_tokens,
                        max_gold_tokens_exclusive=args.max_gold_tokens_exclusive,
                        qwen_max_pixels=args.qwen_max_pixels,
                    )
                summary["residual_add_layer"] = layer
                summary["residual_add_alpha"] = args.alpha
                summary["residual_add_token_scope"] = args.token_scope
                summaries.append({"model_alias": model_alias, "language": language, "layer": layer, "summary": summary})

        del model, processor, tokenizer, embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = args.output_root / "runs" / "ccocr_residual_add" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
