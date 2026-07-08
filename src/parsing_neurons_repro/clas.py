from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from .io import read_records, window_layers, write_json
from .models import decoder_layers


def language_model_candidates(model: Any) -> list[Any]:
    candidates = [model]
    for path in [
        ("language_model",),
        ("language_model", "model"),
        ("model", "language_model"),
        ("model", "language_model", "model"),
        ("model",),
    ]:
        current = model
        ok = True
        for attr in path:
            if not hasattr(current, attr):
                ok = False
                break
            current = getattr(current, attr)
        if ok and current not in candidates:
            candidates.append(current)
    return candidates


def forward_text_only(model: Any, inputs: dict[str, torch.Tensor]) -> None:
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "use_cache": False,
    }
    last_error: Exception | None = None
    for candidate in language_model_candidates(model):
        try:
            candidate(**{key: value for key, value in kwargs.items() if value is not None})
            return
        except Exception as exc:  # noqa: BLE001 - try compatible model entrypoints.
            last_error = exc
    raise RuntimeError("Could not run a text-only forward pass through the language model.") from last_error


def load_parallel_flores_texts(
    *,
    flores_root: Path,
    languages: list[str],
    max_samples: int = 100,
    anchor_language: str = "English",
) -> dict[str, list[str]]:
    """Load aligned English/target texts from per-language FLORES pair files."""

    by_language: dict[str, list[str]] = {}
    anchor_rows: list[dict[str, Any]] | None = None
    for language in languages:
        rows = read_records(flores_root / f"en_to_{language.lower()}.json")[:max_samples]
        if anchor_rows is None:
            anchor_rows = rows
            by_language[anchor_language] = [str(row["english"]) for row in rows]
        by_language[language] = [str(row["target"]) for row in rows]
    return by_language


XQUAD_CONFIGS = {
    "English": "xquad.en",
    "Arabic": "xquad.ar",
    "German": "xquad.de",
    "Greek": "xquad.el",
    "Spanish": "xquad.es",
    "Hindi": "xquad.hi",
    "Romanian": "xquad.ro",
    "Russian": "xquad.ru",
    "Thai": "xquad.th",
    "Turkish": "xquad.tr",
    "Vietnamese": "xquad.vi",
    "Chinese": "xquad.zh",
}


def xquad_prompt(row: dict[str, Any]) -> str:
    return f"Context: {row['context']}\nQuestion: {row['question']}\nAnswer:"


def load_parallel_xquad_texts(
    *,
    languages: list[str],
    max_samples: int = 100,
    dataset_name: str = "google/xquad",
    split: str = "validation",
) -> dict[str, list[str]]:
    """Load aligned XQuAD prompts across languages.

    CLAS computes category masks from 100 parallel XQuAD examples. We align
    examples by XQuAD id and build the same kind of QA prompt used at inference.
    """

    tables: dict[str, dict[str, dict[str, Any]]] = {}
    for language in languages:
        if language not in XQUAD_CONFIGS:
            raise ValueError(f"Unsupported XQuAD language {language!r}. Known: {sorted(XQUAD_CONFIGS)}")
        dataset = load_dataset(dataset_name, XQUAD_CONFIGS[language], split=split)
        tables[language] = {str(row["id"]): dict(row) for row in dataset}

    common_ids: set[str] | None = None
    for rows_by_id in tables.values():
        ids = set(rows_by_id)
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise RuntimeError("No aligned XQuAD ids found across requested languages.")

    selected_ids = sorted(common_ids)[:max_samples]
    return {
        language: [xquad_prompt(tables[language][sample_id]) for sample_id in selected_ids]
        for language in languages
    }


class MLPActivationStats:
    """Accumulate MLP intermediate activation means by language and layer."""

    def __init__(
        self,
        model: Any,
        layers: list[int],
        *,
        token_scope: str = "last_nonpad",
        statistic: str = "mean",
    ) -> None:
        self.decoder_layers = decoder_layers(model)
        self.target_layers = [int(layer) for layer in layers]
        self.token_scope = token_scope
        self.statistic = statistic
        self.handles: list[Any] = []
        self.current_language: str | None = None
        self.current_attention_mask: torch.Tensor | None = None
        self.sums: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.counts: dict[int, Counter[str]] = defaultdict(Counter)

    def __enter__(self) -> "MLPActivationStats":
        for layer in self.target_layers:
            self.handles.append(self.decoder_layers[layer].mlp.down_proj.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    def set_batch(self, *, language: str, attention_mask: torch.Tensor | None) -> None:
        self.current_language = language
        self.current_attention_mask = attention_mask

    def means(self) -> dict[int, dict[str, torch.Tensor]]:
        out: dict[int, dict[str, torch.Tensor]] = {}
        for layer, lang_sums in self.sums.items():
            out[layer] = {}
            for language, total in lang_sums.items():
                count = max(1, int(self.counts[layer][language]))
                out[layer][language] = total / count
        return out

    def _summarize(self, hidden: torch.Tensor) -> torch.Tensor:
        values = hidden.detach()
        mask = self.current_attention_mask
        if self.token_scope == "last_nonpad" and mask is not None:
            arange = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
            last_indices = (mask.to(torch.long) * arange).argmax(dim=1)
            values = values[torch.arange(values.shape[0], device=values.device), last_indices, :]
        elif self.token_scope == "all_nonpad" and mask is not None:
            values = values[mask.to(torch.bool)]
        else:
            values = values.reshape(-1, values.shape[-1])

        if self.statistic == "mean_abs":
            values = values.abs()
        elif self.statistic != "mean":
            raise ValueError(f"Unsupported CLAS statistic: {self.statistic}")
        return values.float().sum(dim=0).cpu(), torch.tensor(values.shape[0])

    def _hook(self, layer: int):
        def hook(module, inputs):
            if self.current_language is None:
                return inputs
            batch_sum, count = self._summarize(inputs[0])
            if self.current_language not in self.sums[layer]:
                self.sums[layer][self.current_language] = torch.zeros_like(batch_sum)
            self.sums[layer][self.current_language] += batch_sum
            self.counts[layer][self.current_language] += int(count.item())
            return inputs

        return hook


def categorize_means(
    *,
    means: dict[int, dict[str, torch.Tensor]],
    languages: list[str],
    activation_threshold: float = 0.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, int]] = {}
    language_order = list(languages)
    for layer in sorted(means):
        stacked = torch.stack([means[layer][language] for language in language_order], dim=0)
        active = stacked > float(activation_threshold)
        layer_counts: Counter[str] = Counter()
        for neuron in range(stacked.shape[1]):
            active_languages = [language_order[idx] for idx, flag in enumerate(active[:, neuron].tolist()) if flag]
            if not active_languages:
                category = "dead"
            elif len(active_languages) == len(language_order):
                category = "all_shared"
            elif len(active_languages) == 1:
                category = "language_specific"
            else:
                category = "partial_shared"
            layer_counts[category] += 1
            rows.append(
                {
                    "layer": int(layer),
                    "neuron": int(neuron),
                    "category": category,
                    "active_languages": active_languages,
                    "means": {
                        language: float(stacked[lang_idx, neuron].item())
                        for lang_idx, language in enumerate(language_order)
                    },
                }
            )
        summaries[str(layer)] = dict(sorted(layer_counts.items()))
    return {"rows": rows, "summary_by_layer": summaries}


def compute_clas_stats(
    *,
    model: Any,
    tokenizer: Any,
    texts_by_language: dict[str, list[str]],
    layers: list[int] | str,
    output_path: Path,
    batch_size: int = 8,
    max_length: int = 512,
    activation_threshold: float = 0.0,
    token_scope: str = "last_nonpad",
    statistic: str = "mean",
) -> dict[str, Any]:
    layer_list = window_layers(layers) if isinstance(layers, str) else [int(layer) for layer in layers]
    languages = list(texts_by_language)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    with MLPActivationStats(model, layer_list, token_scope=token_scope, statistic=statistic) as collector:
        for language, texts in texts_by_language.items():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                collector.set_batch(language=language, attention_mask=inputs.get("attention_mask"))
                with torch.inference_mode():
                    forward_text_only(model, inputs)
        means = collector.means()

    payload = categorize_means(means=means, languages=languages, activation_threshold=activation_threshold)
    payload["metadata"] = {
        "method": "CLAS reimplementation",
        "languages": languages,
        "layers": layer_list,
        "batch_size": batch_size,
        "max_length": max_length,
        "activation_threshold": activation_threshold,
        "token_scope": token_scope,
        "statistic": statistic,
        "num_samples_by_language": {language: len(texts) for language, texts in texts_by_language.items()},
    }
    write_json(output_path, payload)
    return payload
