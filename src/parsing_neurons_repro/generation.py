from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from .models import model_family

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover
    process_vision_info = None


def model_device(model: Any) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def move_to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def close_images(images: list[list[Image.Image]]) -> None:
    for group in images:
        for image in group:
            image.close()


class AllowedTokenLogitsProcessor(LogitsProcessor):
    """Mask generation to a fixed set of token ids."""

    def __init__(self, allowed_token_ids: list[int] | set[int]) -> None:
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids must not be empty.")
        self.allowed_token_ids = sorted({int(token_id) for token_id in allowed_token_ids})
        self._mask_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        key = (scores.device, scores.shape[-1])
        if key not in self._mask_cache:
            mask = torch.full((scores.shape[-1],), torch.finfo(scores.dtype).min, device=scores.device)
            valid = [token_id for token_id in self.allowed_token_ids if 0 <= token_id < scores.shape[-1]]
            if not valid:
                raise ValueError("No allowed token ids are valid for this vocabulary.")
            mask[torch.tensor(valid, dtype=torch.long, device=scores.device)] = 0
            self._mask_cache[key] = mask
        return scores + self._mask_cache[key]


def build_gemma_prompt(processor: Any, question: str, *, suffix: str = "", question_prefix: str = "") -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_prefix}{question}{suffix}"},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_qwen_message(
    image: Image.Image | Path | str,
    question: str,
    *,
    suffix: str = "",
    qwen_max_pixels: int | None = None,
    qwen_min_pixels: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(image, Path):
        image_value: Any = f"file://{image.resolve()}"
    else:
        image_value = image
    payload: dict[str, Any] = {"type": "image", "image": image_value}
    if qwen_max_pixels is not None:
        payload["max_pixels"] = int(qwen_max_pixels)
    if qwen_min_pixels is not None:
        payload["min_pixels"] = int(qwen_min_pixels)
    return [{"role": "user", "content": [payload, {"type": "text", "text": f"{question}{suffix}"}]}]


def prepare_vlm_inputs(
    *,
    model_alias: str,
    processor: Any,
    images: list[Image.Image | Path | str],
    questions: list[str],
    suffix: str = "",
    question_prefix: str = "",
    qwen_max_pixels: int | None = None,
    qwen_min_pixels: int | None = None,
) -> Any:
    family = model_family(model_alias)
    if family == "qwen":
        if process_vision_info is None:
            raise RuntimeError("qwen-vl-utils is required for Qwen-VL generation.")
        messages = [
            build_qwen_message(
                image,
                question,
                suffix=suffix,
                qwen_max_pixels=qwen_max_pixels,
                qwen_min_pixels=qwen_min_pixels,
            )
            for image, question in zip(images, questions)
        ]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    prompts = [
        build_gemma_prompt(processor, question, suffix=suffix, question_prefix=question_prefix)
        for question in questions
    ]
    pil_images: list[list[Image.Image]] = []
    for image in images:
        if isinstance(image, Image.Image):
            pil_images.append([image.convert("RGB")])
        else:
            with Image.open(image) as opened:
                pil_images.append([opened.convert("RGB")])
    try:
        return processor(text=prompts, images=pil_images, padding=True, return_tensors="pt")
    finally:
        close_images(pil_images)


def decode_new_tokens(*, model_alias: str, processor: Any, generated: torch.Tensor, inputs: Any) -> list[str]:
    if model_family(model_alias) == "qwen":
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
    else:
        input_len = int(inputs["input_ids"].shape[1])
        trimmed = generated[:, input_len:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [text.strip() for text in decoded]


def generate_batch(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    images: list[Image.Image | Path | str],
    questions: list[str],
    suffix: str = "",
    question_prefix: str = "",
    max_new_tokens: int = 16,
    qwen_max_pixels: int | None = None,
    qwen_min_pixels: int | None = None,
    allowed_token_ids: list[int] | set[int] | None = None,
) -> list[str]:
    inputs = prepare_vlm_inputs(
        model_alias=model_alias,
        processor=processor,
        images=images,
        questions=questions,
        suffix=suffix,
        question_prefix=question_prefix,
        qwen_max_pixels=qwen_max_pixels,
        qwen_min_pixels=qwen_min_pixels,
    )
    inputs = move_to_device(inputs, model_device(model))
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if allowed_token_ids is not None:
        generate_kwargs["logits_processor"] = LogitsProcessorList([AllowedTokenLogitsProcessor(allowed_token_ids)])
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    return decode_new_tokens(model_alias=model_alias, processor=processor, generated=generated, inputs=inputs)


def generate_batch_adaptive(
    *,
    model_alias: str,
    processor: Any,
    model: Any,
    images: list[Image.Image | Path | str],
    questions: list[str],
    suffix: str = "",
    question_prefix: str = "",
    max_new_tokens: int = 16,
    qwen_max_pixels: int | None = None,
    qwen_min_pixels: int | None = None,
    allowed_token_ids: list[int] | set[int] | None = None,
    adaptive_oom_split: bool = True,
) -> list[str]:
    try:
        return generate_batch(
            model_alias=model_alias,
            processor=processor,
            model=model,
            images=images,
            questions=questions,
            suffix=suffix,
            question_prefix=question_prefix,
            max_new_tokens=max_new_tokens,
            qwen_max_pixels=qwen_max_pixels,
            qwen_min_pixels=qwen_min_pixels,
            allowed_token_ids=allowed_token_ids,
        )
    except torch.OutOfMemoryError:
        if not adaptive_oom_split or len(images) == 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        midpoint = len(images) // 2
        return generate_batch_adaptive(
            model_alias=model_alias,
            processor=processor,
            model=model,
            images=images[:midpoint],
            questions=questions[:midpoint],
            suffix=suffix,
            question_prefix=question_prefix,
            max_new_tokens=max_new_tokens,
            qwen_max_pixels=qwen_max_pixels,
            qwen_min_pixels=qwen_min_pixels,
            allowed_token_ids=allowed_token_ids,
            adaptive_oom_split=adaptive_oom_split,
        ) + generate_batch_adaptive(
            model_alias=model_alias,
            processor=processor,
            model=model,
            images=images[midpoint:],
            questions=questions[midpoint:],
            suffix=suffix,
            question_prefix=question_prefix,
            max_new_tokens=max_new_tokens,
            qwen_max_pixels=qwen_max_pixels,
            qwen_min_pixels=qwen_min_pixels,
            allowed_token_ids=allowed_token_ids,
            adaptive_oom_split=adaptive_oom_split,
        )
