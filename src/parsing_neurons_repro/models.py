from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None

from .io import read_json


MODEL_PATHS = {
    "gemma3_4b_it": os.environ.get(
        "GEMMA3_4B_IT_PATH",
        "google/gemma-3-4b-it",
    ),
    "gemma3_12b_it": os.environ.get(
        "GEMMA3_12B_IT_PATH",
        "google/gemma-3-12b-it",
    ),
    "qwen2_5_vl_3b_instruct": os.environ.get(
        "QWEN25VL_3B_INSTRUCT_PATH",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "qwen3_vl_8b_instruct": os.environ.get(
        "QWEN3VL_8B_INSTRUCT_PATH",
        "Qwen/Qwen3-VL-8B-Instruct",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    path: Path
    family: str
    num_layers: int
    hidden_size: int
    intermediate_size: int


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_model_path(alias_or_path: str | Path) -> Path:
    raw = str(alias_or_path)
    return Path(MODEL_PATHS.get(raw, raw)).expanduser().resolve()


def model_family(alias_or_path: str) -> str:
    name = alias_or_path.lower()
    if "qwen" in name:
        return "qwen"
    if "gemma" in name:
        return "gemma"
    return "auto"


def text_config(model_path: Path) -> dict[str, int]:
    config = read_json(model_path / "config.json")
    cfg = config.get("text_config", config)
    return {
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "hidden_size": int(cfg["hidden_size"]),
        "intermediate_size": int(cfg["intermediate_size"]),
    }


def model_spec(alias: str, model_path: Path | None = None) -> ModelSpec:
    path = resolve_model_path(model_path or alias)
    cfg = text_config(path)
    return ModelSpec(
        alias=alias,
        path=path,
        family=model_family(alias),
        num_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
    )


def load_processor(model_path: Path) -> Any:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    return processor


def load_tokenizer(model_path: Path) -> Any:
    try:
        tokenizer = getattr(load_processor(model_path), "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_vlm(alias: str, model_path: Path, dtype: str = "bfloat16") -> Any:
    family = model_family(alias)
    lower = alias.lower()
    if family == "qwen" and "qwen3" in lower:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("This Transformers build does not expose Qwen3VLForConditionalGeneration.")
        cls = Qwen3VLForConditionalGeneration
    elif family == "qwen":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError("This Transformers build does not expose Qwen2_5_VLForConditionalGeneration.")
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        cls = AutoModelForImageTextToText
    return cls.from_pretrained(
        model_path,
        torch_dtype=dtype_from_name(dtype),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval()


def decoder_layers(model: Any) -> Any:
    candidates = [
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    ]
    for path in candidates:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok:
            return current
    raise AttributeError("Could not locate decoder layers in model.")


def weight_map(model_path: Path) -> dict[str, str]:
    return read_json(model_path / "model.safetensors.index.json")["weight_map"]


def first_existing_key(index: dict[str, str], candidates: list[str]) -> str:
    for key in candidates:
        if key in index:
            return key
    raise KeyError(f"None of the candidate weight keys exist: {candidates}")


def weight_templates(index: dict[str, str]) -> dict[str, str]:
    embed = first_existing_key(
        index,
        [
            "language_model.model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
    )
    mlp0 = first_existing_key(
        index,
        [
            "language_model.model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ],
    )
    attn0 = first_existing_key(
        index,
        [
            "language_model.model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ],
    )

    def templated(key: str, suffix: str) -> str:
        return key[: -len("0." + suffix)] + "{layer}." + suffix

    return {
        "embed": embed,
        "mlp_down": templated(mlp0, "mlp.down_proj.weight"),
        "attn_o": templated(attn0, "self_attn.o_proj.weight"),
    }


def load_tensor(model_path: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(model_path / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)
