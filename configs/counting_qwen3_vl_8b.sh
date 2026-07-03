#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PYTHONPATH=src "$PYTHON" scripts/run_search.py \
  --task counting \
  --model-alias qwen3_vl_8b_instruct \
  --direction word-minus-digit \
  --counting-language pooled_en_zh \
  --windows 34-35,33-35,32-35,31-35,30-35 \
  --sigmas 2.5,3,3.5,4,4.5,5 \
  --component-modes mlp,mlp_attn \
  --batch-size 64 \
  --max-new-tokens 16 \
  --qwen-max-pixels 401408
