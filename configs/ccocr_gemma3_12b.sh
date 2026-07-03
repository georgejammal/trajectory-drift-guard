#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
PYTHONPATH=src "$PYTHON" scripts/run_search.py \
  --task ccocr \
  --model-alias gemma3_12b_it \
  --languages Arabic,Japanese,Korean,Russian \
  --windows 24-46,24-44,24-43,25-44,24-42 \
  --sigmas 2.5,3,3.5,4,4.5,5 \
  --component-modes mlp,mlp_attn \
  --batch-size 256 \
  --max-new-tokens 512 \
  --max-gold-tokens-exclusive 500
