from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .clas import forward_text_only, load_parallel_flores_texts
from .io import read_json, window_layers, write_json
from .models import decoder_layers


def _untuple(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _retuple(original: Any, hidden: torch.Tensor) -> Any:
    if isinstance(original, tuple):
        return (hidden, *original[1:])
    return hidden


class MLPOutputCollector:
    """Collect per-layer MLP module outputs for text-only parallel data.

    INCLINE estimates a per-layer linear map from target-language MLP outputs to
    English MLP outputs. The released code uses the mean MLP output over the
    prompt tokens for each example; this collector reproduces that statistic for
    Gemma/Qwen language backbones.
    """

    def __init__(self, model: Any, layers: list[int], *, token_scope: str = "all_nonpad") -> None:
        self.layers = decoder_layers(model)
        self.target_layers = [int(layer) for layer in layers]
        self.token_scope = token_scope
        self.current_attention_mask: torch.Tensor | None = None
        self.current_records: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.handles: list[Any] = []

    def __enter__(self) -> "MLPOutputCollector":
        for layer in self.target_layers:
            self.handles.append(self.layers[layer].mlp.register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.current_attention_mask = attention_mask

    def take_records(self) -> dict[int, list[torch.Tensor]]:
        records = self.current_records
        self.current_records = defaultdict(list)
        return records

    def _summarize(self, hidden: torch.Tensor) -> torch.Tensor:
        values = hidden.detach()
        mask = self.current_attention_mask
        if self.token_scope == "all_nonpad" and mask is not None:
            per_example = []
            bool_mask = mask.to(torch.bool)
            for row, row_mask in zip(values, bool_mask):
                selected = row[row_mask]
                if selected.numel() == 0:
                    selected = row[-1:].reshape(1, -1)
                per_example.append(selected.float().mean(dim=0).cpu())
            return torch.stack(per_example, dim=0)
        if self.token_scope == "last_nonpad" and mask is not None:
            arange = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
            last_indices = (mask.to(torch.long) * arange).argmax(dim=1)
            return values[torch.arange(values.shape[0], device=values.device), last_indices, :].float().cpu()
        return values.float().mean(dim=1).cpu()

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = _untuple(output)
            self.current_records[layer].append(self._summarize(hidden))
            return output

        return hook


def collect_mlp_outputs(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    batch_size: int = 4,
    max_length: int = 512,
    token_scope: str = "all_nonpad",
) -> dict[int, torch.Tensor]:
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    by_layer: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    with MLPOutputCollector(model, layers, token_scope=token_scope) as collector:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            collector.set_attention_mask(inputs.get("attention_mask"))
            with torch.inference_mode():
                forward_text_only(model, inputs)
            for layer, records in collector.take_records().items():
                if records:
                    by_layer[layer].extend(records)
            print(f"[incline:collect] batch={start // batch_size + 1} done={min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return {layer: torch.cat(records, dim=0) for layer, records in by_layer.items()}


def compute_bridge_factors(
    *,
    target_acts: dict[int, torch.Tensor],
    english_acts: dict[int, torch.Tensor],
    ridge: float = 0.0,
) -> dict[int, dict[str, torch.Tensor]]:
    """Compute low-rank factors equivalent to INCLINE least-squares maps.

    INCLINE fits W_l = argmin_W ||X_l W - Y_l||_2, with X_l target-language
    MLP outputs and Y_l English MLP outputs. Since n_samples << d_model here,
    the minimum-norm solution can be applied as

        h W_l = (h X_l^T) (X_l X_l^T)^+ Y_l.

    We cache X_l and B_l = (X_l X_l^T + ridge I)^+ Y_l rather than materialize
    the full d_model x d_model W_l.
    """

    factors: dict[int, dict[str, torch.Tensor]] = {}
    for layer in sorted(target_acts):
        x = target_acts[layer].float()
        y = english_acts[layer].float()
        if x.shape != y.shape:
            raise ValueError(f"Activation shape mismatch at layer {layer}: {x.shape} vs {y.shape}")
        gram = x @ x.T
        if ridge:
            gram = gram + float(ridge) * torch.eye(gram.shape[0], dtype=gram.dtype)
        bridge = torch.linalg.pinv(gram) @ y
        factors[layer] = {"x": x.contiguous(), "bridge": bridge.contiguous()}
    return factors


def incline_training_texts(english: list[str], target: list[str]) -> tuple[list[str], list[str]]:
    """Match the odd/even construction in the released INCLINE scripts."""

    english_out: list[str] = []
    target_out: list[str] = []
    for idx, (en_text, target_text) in enumerate(zip(english, target)):
        en_text = str(en_text).strip()
        target_text = str(target_text).strip()
        if idx % 2 == 0:
            target_out.append(target_text)
            english_out.append(en_text)
        else:
            target_out.append(f"{target_text} {en_text}".strip())
            english_out.append(f"{en_text} {en_text}".strip())
    return english_out, target_out


def tokenized_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
    )
    return len(encoded["input_ids"])


def filter_bridge_text_pairs(
    *,
    tokenizer: Any,
    english: list[str],
    target: list[str],
    max_length: int,
) -> tuple[list[str], list[str], int]:
    """Skip long bridge examples, matching the released INCLINE scripts.

    The upstream implementation discards training examples whose tokenized
    length exceeds its 500-token cap, rather than learning from truncated
    activations. We apply the same rule to the paired English/target bridge
    inputs after the odd/even construction.
    """

    kept_english: list[str] = []
    kept_target: list[str] = []
    skipped = 0
    for english_text, target_text in zip(english, target):
        if tokenized_length(tokenizer, target_text) > max_length:
            skipped += 1
            continue
        if tokenized_length(tokenizer, english_text) > max_length:
            skipped += 1
            continue
        kept_english.append(english_text)
        kept_target.append(target_text)
    return kept_english, kept_target, skipped


def save_bridge(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_payload = {
        "model_alias": payload["model_alias"],
        "language": payload["language"],
        "layers": payload["layers"],
        "num_samples": payload["num_samples"],
        "token_scope": payload["token_scope"],
        "max_length": payload["max_length"],
        "ridge": payload["ridge"],
        "factors": {
            str(layer): {
                "x": tensors["x"],
                "bridge": tensors["bridge"],
            }
            for layer, tensors in payload["factors"].items()
        },
    }
    torch.save(tensor_payload, path)
    metadata = {key: value for key, value in payload.items() if key != "factors"}
    metadata["path"] = str(path)
    write_json(path.with_suffix(".json"), metadata)


def load_bridge(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_flores_bridge(
    *,
    model_alias: str,
    model: Any,
    tokenizer: Any,
    language: str,
    layers: list[int] | str,
    output_path: Path,
    flores_root: Path,
    max_samples: int = 500,
    batch_size: int = 4,
    max_length: int = 512,
    token_scope: str = "all_nonpad",
    ridge: float = 0.0,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    layer_list = window_layers(layers) if isinstance(layers, str) else [int(layer) for layer in layers]
    texts_by_language = load_parallel_flores_texts(
        flores_root=flores_root,
        languages=[language],
        max_samples=max_samples,
        anchor_language="English",
    )
    english = texts_by_language["English"]
    target = texts_by_language[language]
    english, target = incline_training_texts(english, target)
    num_constructed = len(english)
    english, target, skipped = filter_bridge_text_pairs(
        tokenizer=tokenizer,
        english=english,
        target=target,
        max_length=max_length,
    )
    print(
        f"[incline:bridge] kept {len(english)}/{num_constructed} bridge pairs "
        f"for {model_alias}/{language}; skipped {skipped} longer than {max_length} tokens",
        flush=True,
    )
    if not english:
        raise ValueError(
            f"No INCLINE bridge pairs remained for {model_alias}/{language} "
            f"after filtering at max_length={max_length}."
        )
    print(f"[incline:bridge] collecting English activations for {model_alias}/{language}", flush=True)
    english_acts = collect_mlp_outputs(
        model=model,
        tokenizer=tokenizer,
        texts=english,
        layers=layer_list,
        batch_size=batch_size,
        max_length=max_length,
        token_scope=token_scope,
    )
    print(f"[incline:bridge] collecting {language} activations for {model_alias}/{language}", flush=True)
    target_acts = collect_mlp_outputs(
        model=model,
        tokenizer=tokenizer,
        texts=target,
        layers=layer_list,
        batch_size=batch_size,
        max_length=max_length,
        token_scope=token_scope,
    )
    factors = compute_bridge_factors(target_acts=target_acts, english_acts=english_acts, ridge=ridge)
    save_bridge(
        output_path,
        {
            "model_alias": model_alias,
            "language": language,
            "layers": layer_list,
            "num_samples": len(english),
            "num_constructed_samples": num_constructed,
            "num_skipped_long_samples": skipped,
            "token_scope": token_scope,
            "max_length": max_length,
            "ridge": ridge,
            "factors": factors,
        },
    )
    return output_path


class INCLINEIntervention:
    """Patch MLP module outputs with INCLINE's learned cross-lingual map."""

    def __init__(
        self,
        model: Any,
        bridge_path: Path,
        layers: list[int] | str,
        *,
        sigma: float,
        direction_sign: float = 1.0,
        token_scope: str = "last_position",
    ) -> None:
        self.layers = decoder_layers(model)
        self.target_layers = window_layers(layers) if isinstance(layers, str) else [int(layer) for layer in layers]
        payload = load_bridge(bridge_path)
        self.factors = payload["factors"]
        self.sigma = float(sigma)
        self.direction_sign = float(direction_sign)
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def __enter__(self) -> "INCLINEIntervention":
        for layer in self.target_layers:
            if str(layer) not in self.factors:
                raise ValueError(f"Bridge file does not contain layer {layer}.")
            self.handles.append(self.layers[layer].mlp.register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cache.clear()
        return False

    def _layer_factors(self, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, device)
        if key not in self.cache:
            tensors = self.factors[str(layer)]
            self.cache[key] = (
                tensors["x"].to(device=device, dtype=torch.float32),
                tensors["bridge"].to(device=device, dtype=torch.float32),
            )
        return self.cache[key]

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = _untuple(output)
            # Match INCLINE's final prompt-token patching. Decode steps with
            # KV-cache usually have length 1, so they are left unchanged.
            if hidden.ndim != 3 or hidden.shape[1] <= 1:
                return output
            x, bridge = self._layer_factors(layer, hidden.device)
            modified = hidden.clone()
            if self.token_scope == "last_position":
                current = hidden[:, -1, :].float()
                delta = (current @ x.T) @ bridge
                modified[:, -1, :] = hidden[:, -1, :] + (self.sigma * self.direction_sign * delta).to(hidden.dtype)
            elif self.token_scope == "all_positions":
                current = hidden.float()
                flat = current.reshape(-1, current.shape[-1])
                delta = ((flat @ x.T) @ bridge).reshape_as(current)
                modified = hidden + (self.sigma * self.direction_sign * delta).to(hidden.dtype)
            else:
                raise ValueError(f"Unsupported INCLINE token_scope: {self.token_scope}")
            return _retuple(output, modified)

        return hook
