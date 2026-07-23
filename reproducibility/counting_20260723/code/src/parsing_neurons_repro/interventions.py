from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .io import read_json, window_layers
from .models import decoder_layers


def rows_by_layer(selection_path: Path, index_name: str) -> dict[int, list[int]]:
    payload = read_json(selection_path)
    by_layer: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for row in payload["rows"]:
        layer = int(row["layer"])
        idx = int(row[index_name])
        key = (layer, idx)
        if key not in seen:
            by_layer[layer].append(idx)
            seen.add(key)
    return dict(sorted(by_layer.items()))


def transform_selected_scalars(hidden: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "abs":
        return hidden.abs()
    if mode == "zero":
        return torch.zeros_like(hidden)
    if mode == "negative_abs":
        return -hidden.abs()
    if mode == "relu":
        return hidden.clamp_min(0)
    if mode == "scaled_abs_1p2":
        return 1.2 * hidden.abs()
    raise ValueError(f"Unsupported scalar intervention mode: {mode}")


class MLPScalarIntervention:
    """Transform selected gated MLP intermediate coordinates."""

    def __init__(
        self,
        model: Any,
        neurons_by_layer: dict[int, list[int]],
        token_scope: str = "all_positions",
        scalar_mode: str = "abs",
    ) -> None:
        self.layers = decoder_layers(model)
        self.neurons_by_layer = neurons_by_layer
        self.token_scope = token_scope
        self.scalar_mode = scalar_mode
        self.handles: list[Any] = []
        self.index_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "MLPScalarIntervention":
        for layer, neurons in self.neurons_by_layer.items():
            if not neurons:
                continue
            mlp = self.layers[layer].mlp
            if max(neurons) >= mlp.down_proj.in_features:
                raise ValueError(f"MLP neuron index out of range in layer {layer}.")
            self.handles.append(mlp.down_proj.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def _indices(self, layer: int, device: torch.device) -> torch.Tensor:
        key = (layer, device)
        if key not in self.index_cache:
            self.index_cache[key] = torch.tensor(self.neurons_by_layer[layer], dtype=torch.long, device=device)
        return self.index_cache[key]

    def _hook(self, layer: int):
        def hook(module, inputs):
            hidden = inputs[0]
            indices = self._indices(layer, hidden.device)
            modified = hidden.clone()
            if self.token_scope == "last_position":
                modified[:, -1, indices] = transform_selected_scalars(modified[:, -1, indices], self.scalar_mode)
            else:
                modified[..., indices] = transform_selected_scalars(modified[..., indices], self.scalar_mode)
            return (modified,)

        return hook


class AttentionScalarIntervention:
    """Transform selected attention o_proj input channels."""

    def __init__(
        self,
        model: Any,
        channels_by_layer: dict[int, list[int]],
        token_scope: str = "last_position",
        scalar_mode: str = "abs",
    ) -> None:
        self.layers = decoder_layers(model)
        self.channels_by_layer = channels_by_layer
        self.token_scope = token_scope
        self.scalar_mode = scalar_mode
        self.handles: list[Any] = []
        self.index_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "AttentionScalarIntervention":
        for layer, channels in self.channels_by_layer.items():
            if not channels:
                continue
            o_proj = self.layers[layer].self_attn.o_proj
            if max(channels) >= o_proj.in_features:
                raise ValueError(f"Attention channel index out of range in layer {layer}.")
            self.handles.append(o_proj.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def _indices(self, layer: int, device: torch.device) -> torch.Tensor:
        key = (layer, device)
        if key not in self.index_cache:
            self.index_cache[key] = torch.tensor(self.channels_by_layer[layer], dtype=torch.long, device=device)
        return self.index_cache[key]

    def _hook(self, layer: int):
        def hook(module, inputs):
            hidden = inputs[0]
            indices = self._indices(layer, hidden.device)
            modified = hidden.clone()
            if self.token_scope == "all_positions":
                modified[..., indices] = transform_selected_scalars(modified[..., indices], self.scalar_mode)
            else:
                modified[:, -1, indices] = transform_selected_scalars(modified[:, -1, indices], self.scalar_mode)
            return (modified,)

        return hook


class CombinedIntervention:
    def __init__(self, *contexts: Any) -> None:
        self.contexts = [context for context in contexts if context is not None]

    def __enter__(self) -> "CombinedIntervention":
        for context in self.contexts:
            context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for context in reversed(self.contexts):
            context.__exit__(exc_type, exc, tb)
        return False


def clas_indices_by_layer(
    stats_path: Path,
    *,
    layers: list[int],
    target_language: str | None = None,
    specific_scope: str = "all",
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Load CLAS category masks from a stats file.

    CLAS distinguishes partial-shared neurons from language-specific neurons.
    The paper treats these as MLP intermediate coordinates; in this codebase
    those are the input channels to ``mlp.down_proj``.
    """

    payload = read_json(stats_path)
    layer_set = {int(layer) for layer in layers}
    partial: dict[int, list[int]] = defaultdict(list)
    specific: dict[int, list[int]] = defaultdict(list)
    for row in payload["rows"]:
        layer = int(row["layer"])
        if layer not in layer_set:
            continue
        category = row["category"]
        neuron = int(row["neuron"])
        if category == "partial_shared":
            partial[layer].append(neuron)
        elif category == "language_specific":
            if specific_scope == "all":
                specific[layer].append(neuron)
            elif specific_scope == "target":
                if target_language is None:
                    raise ValueError("target_language is required when specific_scope='target'.")
                active = row.get("active_languages", [])
                if len(active) == 1 and str(active[0]).lower() == target_language.lower():
                    specific[layer].append(neuron)
            else:
                raise ValueError(f"Unsupported CLAS specific_scope: {specific_scope}")
    return dict(sorted(partial.items())), dict(sorted(specific.items()))


class CLASIntervention:
    """Cross-Lingual Activation Steering for MLP intermediate activations.

    This is a paper-faithful reimplementation of CLAS as a baseline. Given
    category masks estimated from parallel multilingual inputs, it rescales
    partial-shared and language-specific MLP coordinates and blends the result
    with the original activation, matching the CLAS paper notation:

        h1 = h * (1 + beta * M_partial)
        h2 = h1 * (1 - gamma * M_specific)
        h_final = (1 - alpha) * h + alpha * h2

    where ``h`` is the gated MLP intermediate vector entering ``down_proj``.
    """

    def __init__(
        self,
        model: Any,
        stats_path: Path,
        layers: list[int] | str,
        *,
        target_language: str | None = None,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.2,
        token_scope: str = "all_positions",
        specific_scope: str = "all",
    ) -> None:
        self.layers = decoder_layers(model)
        self.target_layers = window_layers(layers) if isinstance(layers, str) else [int(layer) for layer in layers]
        self.partial_by_layer, self.specific_by_layer = clas_indices_by_layer(
            stats_path,
            layers=self.target_layers,
            target_language=target_language,
            specific_scope=specific_scope,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.index_cache: dict[tuple[str, int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "CLASIntervention":
        for layer in self.target_layers:
            partial = self.partial_by_layer.get(layer, [])
            specific = self.specific_by_layer.get(layer, [])
            if not partial and not specific:
                continue
            mlp = self.layers[layer].mlp
            max_index = max(partial + specific)
            if max_index >= mlp.down_proj.in_features:
                raise ValueError(f"CLAS neuron index out of range in layer {layer}.")
            self.handles.append(mlp.down_proj.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def _indices(self, kind: str, layer: int, device: torch.device) -> torch.Tensor:
        key = (kind, layer, device)
        if key not in self.index_cache:
            source = self.partial_by_layer if kind == "partial" else self.specific_by_layer
            self.index_cache[key] = torch.tensor(source.get(layer, []), dtype=torch.long, device=device)
        return self.index_cache[key]

    def _apply(self, hidden: torch.Tensor, partial: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
        modified = hidden.clone()
        if partial.numel():
            modified[..., partial] = modified[..., partial] * (1.0 + self.alpha * self.beta)
        if specific.numel():
            modified[..., specific] = modified[..., specific] * (1.0 - self.alpha * self.gamma)
        return modified

    def _hook(self, layer: int):
        def hook(module, inputs):
            hidden = inputs[0]
            partial = self._indices("partial", layer, hidden.device)
            specific = self._indices("specific", layer, hidden.device)
            if self.token_scope == "last_position":
                modified = hidden.clone()
                modified[:, -1:, :] = self._apply(modified[:, -1:, :], partial, specific)
                return (modified,)
            return (self._apply(hidden, partial, specific),)

        return hook


def build_abs_intervention(
    *,
    model: Any,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    mlp_token_scope: str = "all_positions",
    attn_token_scope: str = "last_position",
    scalar_mode: str = "abs",
) -> CombinedIntervention:
    contexts = []
    if component_mode in {"mlp", "mlp_attn"}:
        if mlp_selection is None:
            raise ValueError("MLP selection path is required for component mode with MLP.")
        contexts.append(
            MLPScalarIntervention(
                model,
                rows_by_layer(mlp_selection, "neuron"),
                token_scope=mlp_token_scope,
                scalar_mode=scalar_mode,
            )
        )
    if component_mode in {"attn", "mlp_attn"}:
        if attn_selection is None:
            raise ValueError("Attention selection path is required for component mode with attention.")
        contexts.append(
            AttentionScalarIntervention(
                model,
                rows_by_layer(attn_selection, "channel"),
                token_scope=attn_token_scope,
                scalar_mode=scalar_mode,
            )
        )
    if component_mode == "baseline":
        return CombinedIntervention()
    return CombinedIntervention(*contexts)


class ResidualAddIntervention:
    """Add alpha * direction to the residual stream at selected layer inputs."""

    def __init__(
        self,
        model: Any,
        layers: list[int],
        direction: torch.Tensor,
        alpha: float = 1.0,
        token_scope: str = "last_position",
    ) -> None:
        self.decoder_layers = decoder_layers(model)
        self.target_layers = [int(layer) for layer in layers]
        self.direction = direction.float()
        self.alpha = float(alpha)
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.direction_cache: dict[torch.device, torch.Tensor] = {}

    def __enter__(self) -> "ResidualAddIntervention":
        for layer in self.target_layers:
            self.handles.append(self.decoder_layers[layer].register_forward_pre_hook(self._hook()))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def _direction(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if device not in self.direction_cache:
            self.direction_cache[device] = self.direction.to(device=device)
        return self.direction_cache[device].to(dtype=dtype)

    def _hook(self):
        def hook(module, inputs):
            if not inputs:
                return inputs
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.ndim < 2:
                return inputs
            direction = self._direction(hidden.device, hidden.dtype)
            modified = hidden.clone()
            if self.token_scope == "all_positions":
                modified = modified + self.alpha * direction.view(*([1] * (hidden.ndim - 1)), -1)
            else:
                modified[:, -1, :] = modified[:, -1, :] + self.alpha * direction
            return (modified, *inputs[1:])

        return hook
