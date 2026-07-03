#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
PYTHONPATH=src "$PYTHON" scripts/run_search.py \
  --task ccocr \
  --model-alias qwen3_vl_8b_instruct \
  --languages Arabic,Japanese,Korean,Russian \
  --windows 18-29,18-30,18-28,20-30,22-30 \
  --sigmas 2.5,3,3.5,4,4.5,5 \
  --component-modes mlp,mlp_attn \
  --batch-size 192 \
  --max-new-tokens 512 \
  --max-gold-tokens-exclusive 500 \
  --qwen-max-pixels 1003520
