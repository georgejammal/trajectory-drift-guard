from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .io import read_records


NCWM_LANGUAGE_CODES = {
    "Arabic": "ar",
    "Japanese": "ja",
    "Russian": "ru",
}


NUMBER_WORDS_EN = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

NUMBER_WORDS_ZH = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [int(item) for item in tokenizer(text, add_special_tokens=False)["input_ids"]]


def mean_embedding(embedding: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        raise ValueError("Cannot average an empty tokenization.")
    return embedding[torch.tensor(ids, dtype=torch.long)].float().mean(dim=0)


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector.float(), dim=0)


def number_word(number: int, language: str) -> str:
    if language == "en":
        return NUMBER_WORDS_EN[number]
    if language == "zh":
        return NUMBER_WORDS_ZH[number]
    raise ValueError(f"Unsupported counting language: {language}")


def counting_direction(
    tokenizer: Any,
    embedding: torch.Tensor,
    *,
    direction: str,
    language: str,
    digits: list[int] | tuple[int, ...] = tuple(range(10)),
) -> tuple[torch.Tensor, dict[str, Any]]:
    languages = ["en", "zh"] if language == "pooled_en_zh" else [language]
    components: list[torch.Tensor] = []
    metadata_components: list[dict[str, Any]] = []
    for lang in languages:
        for digit in digits:
            digit_vec = mean_embedding(embedding, token_ids(tokenizer, str(digit)))
            word = number_word(int(digit), lang)
            word_vec = mean_embedding(embedding, token_ids(tokenizer, word))
            if direction == "word-minus-digit":
                component = word_vec - digit_vec
            elif direction == "digit-minus-word":
                component = digit_vec - word_vec
            else:
                raise ValueError(f"Unsupported counting direction: {direction}")
            components.append(component)
            metadata_components.append(
                {
                    "language": lang,
                    "digit": int(digit),
                    "word": word,
                    "digit_token_ids": token_ids(tokenizer, str(digit)),
                    "word_token_ids": token_ids(tokenizer, word),
                    "component_norm": float(component.norm().item()),
                }
            )
    raw = torch.stack(components, dim=0).mean(dim=0)
    return normalized(raw), {
        "kind": "counting",
        "direction": direction,
        "language": language,
        "digits": [int(d) for d in digits],
        "formula": f"normalize(mean({direction}))",
        "norm_before_normalization": float(raw.norm().item()),
        "components": metadata_components,
    }


def flores_direction(
    tokenizer: Any,
    embedding: torch.Tensor,
    *,
    pairs_path: Path,
    source_field: str = "english",
    target_field: str = "target",
    max_pairs: int | None = 500,
    list_key: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rows = read_records(pairs_path, list_key=list_key)
    if max_pairs is not None:
        rows = rows[:max_pairs]
    components: list[torch.Tensor] = []
    used: list[dict[str, Any]] = []
    for row in rows:
        source = str(row[source_field])
        target = str(row[target_field])
        source_ids = token_ids(tokenizer, source)
        target_ids = token_ids(tokenizer, target)
        if not source_ids or not target_ids:
            continue
        component = mean_embedding(embedding, target_ids) - mean_embedding(embedding, source_ids)
        components.append(component)
        used.append(
            {
                "source": source,
                "target": target,
                "source_token_count": len(source_ids),
                "target_token_count": len(target_ids),
                "component_norm": float(component.norm().item()),
            }
        )
    if not components:
        raise RuntimeError(f"No usable FLORES pairs found in {pairs_path}")
    raw = torch.stack(components, dim=0).mean(dim=0)
    return normalized(raw), {
        "kind": "multilingual_ocr",
        "direction": f"{target_field}-minus-{source_field}",
        "pairs_path": str(pairs_path),
        "source_field": source_field,
        "target_field": target_field,
        "num_pairs": len(used),
        "formula": "normalize(mean(E(target_sentence)-E(source_sentence)))",
        "norm_before_normalization": float(raw.norm().item()),
        "sample_components": used[:20],
    }


def news_commentary_direction(
    tokenizer: Any,
    embedding: torch.Tensor,
    *,
    ncwm_root: Path,
    language: str,
    max_pairs: int | None = 500,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if language not in NCWM_LANGUAGE_CODES:
        raise ValueError(
            f"News Commentary direction data is not available for {language!r}. "
            f"Available languages: {sorted(NCWM_LANGUAGE_CODES)}"
        )
    code = NCWM_LANGUAGE_CODES[language]
    pair_root = ncwm_root / f"en-{code}"
    english_path = pair_root / "train.en"
    target_path = pair_root / f"train.{code}"
    if not english_path.exists() or not target_path.exists():
        raise FileNotFoundError(
            f"Missing News Commentary files for {language}: expected "
            f"{english_path} and {target_path}."
        )
    english = [line.strip() for line in english_path.read_text(encoding="utf-8").splitlines()]
    target = [line.strip() for line in target_path.read_text(encoding="utf-8").splitlines()]
    pairs = [(en, tgt) for en, tgt in zip(english, target) if en and tgt]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    components: list[torch.Tensor] = []
    used: list[dict[str, Any]] = []
    for source, target_text in pairs:
        source_ids = token_ids(tokenizer, source)
        target_ids = token_ids(tokenizer, target_text)
        if not source_ids or not target_ids:
            continue
        component = mean_embedding(embedding, target_ids) - mean_embedding(embedding, source_ids)
        components.append(component)
        used.append(
            {
                "source": source,
                "target": target_text,
                "source_token_count": len(source_ids),
                "target_token_count": len(target_ids),
                "component_norm": float(component.norm().item()),
            }
        )
    if not components:
        raise RuntimeError(f"No usable News Commentary pairs found for {language} in {pair_root}")
    raw = torch.stack(components, dim=0).mean(dim=0)
    return normalized(raw), {
        "kind": "multilingual_ocr",
        "direction": "target-minus-english",
        "source": "news_commentary",
        "ncwm_root": str(ncwm_root),
        "language": language,
        "language_code": code,
        "english_path": str(english_path),
        "target_path": str(target_path),
        "num_pairs": len(used),
        "formula": "normalize(mean(E(target_sentence)-E(english_sentence)))",
        "norm_before_normalization": float(raw.norm().item()),
        "sample_components": used[:20],
    }
