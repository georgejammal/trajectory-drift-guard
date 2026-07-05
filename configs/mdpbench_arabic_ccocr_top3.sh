#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/georgejammal/projects/a100env/bin/python}"

export GEMMA3_4B_IT_PATH="${GEMMA3_4B_IT_PATH:-/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767}"
export GEMMA3_12B_IT_PATH="${GEMMA3_12B_IT_PATH:-/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80}"
export QWEN3VL_8B_INSTRUCT_PATH="${QWEN3VL_8B_INSTRUCT_PATH:-/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct}"
export PARSING_NEURONS_DATA_ROOT="${PARSING_NEURONS_DATA_ROOT:-/home/georgejammal/projects/parsing_neurons_repro/data}"

PYTHONPATH=src "$PYTHON" scripts/run_mdpbench_arabic_ccocr_top3.py \
  --models "${MODELS:-gemma3_4b_it,gemma3_12b_it,qwen3_vl_8b_instruct}" \
  --batch-size "${BATCH_SIZE:-300}" \
  --gt-token-cap-multiplier "${GT_TOKEN_CAP_MULTIPLIER:-1.2}" \
  --max-gt-token-cap "${MAX_GT_TOKEN_CAP:-1600}" \
  --qwen-max-pixels "${QWEN_MAX_PIXELS:-1003520}"
