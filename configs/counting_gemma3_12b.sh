#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
PYTHONPATH=src "$PYTHON" scripts/run_search.py \
  --task counting \
  --model-alias gemma3_12b_it \
  --direction word-minus-digit \
  --counting-language en \
  --windows 24-46,24-44,24-42,26-46,28-46 \
  --sigmas 2.5,3,3.5,4,4.5,5 \
  --component-modes mlp,mlp_attn \
  --batch-size 100 \
  --max-new-tokens 16
