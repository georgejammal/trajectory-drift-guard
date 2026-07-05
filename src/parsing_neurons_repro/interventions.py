from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .io import read_json
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


class MLPAbsIntervention:
    """Apply a <- |a| to selected gated MLP intermediate coordinates."""

    def __init__(self, model: Any, neurons_by_layer: dict[int, list[int]], token_scope: str = "all_positions") -> None:
        self.layers = decoder_layers(model)
        self.neurons_by_layer = neurons_by_layer
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.index_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "MLPAbsIntervention":
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
                modified[:, -1, indices] = modified[:, -1, indices].abs()
            else:
                modified[..., indices] = modified[..., indices].abs()
            return (modified,)

        return hook


class AttentionAbsIntervention:
    """Apply z <- |z| to selected attention o_proj input channels at the last token."""

    def __init__(self, model: Any, channels_by_layer: dict[int, list[int]], token_scope: str = "last_position") -> None:
        self.layers = decoder_layers(model)
        self.channels_by_layer = channels_by_layer
        self.token_scope = token_scope
        self.handles: list[Any] = []
        self.index_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def __enter__(self) -> "AttentionAbsIntervention":
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
                modified[..., indices] = modified[..., indices].abs()
            else:
                modified[:, -1, indices] = modified[:, -1, indices].abs()
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


def build_abs_intervention(
    *,
    model: Any,
    component_mode: str,
    mlp_selection: Path | None,
    attn_selection: Path | None,
    mlp_token_scope: str = "all_positions",
    attn_token_scope: str = "last_position",
) -> CombinedIntervention:
    contexts = []
    if component_mode in {"mlp", "mlp_attn"}:
        if mlp_selection is None:
            raise ValueError("MLP selection path is required for component mode with MLP.")
        contexts.append(MLPAbsIntervention(model, rows_by_layer(mlp_selection, "neuron"), token_scope=mlp_token_scope))
    if component_mode in {"attn", "mlp_attn"}:
        if attn_selection is None:
            raise ValueError("Attention selection path is required for component mode with attention.")
        contexts.append(AttentionAbsIntervention(model, rows_by_layer(attn_selection, "channel"), token_scope=attn_token_scope))
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
