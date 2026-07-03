#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
PYTHONPATH=src "$PYTHON" scripts/run_search.py \
  --task counting \
  --model-alias gemma3_4b_it \
  --direction word-minus-digit \
  --counting-language en \
  --windows 17-27,17-25,17-23,19-27,21-27 \
  --sigmas 2.5,3,3.5,4,4.5,5 \
  --component-modes mlp,mlp_attn \
  --batch-size 300 \
  --max-new-tokens 16
