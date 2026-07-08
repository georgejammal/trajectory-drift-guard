#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 scripts/run_clas_ccocr.py \
  --model-alias gemma3_4b_it \
  --stats-source xquad \
  --stats-layers 17-33 \
  --intervention-windows 17-29,19-31 \
  --alphas=-5,-3,-1,1,3,5 \
  --betas 0.4 \
  --gammas 0.2 \
  --specific-scope all \
  --languages Arabic,Japanese,Korean,Russian \
  --stats-samples 100 \
  --stats-batch-size 8 \
  --batch-size 64 \
  --max-new-tokens 512 \
  --max-gold-tokens-exclusive 500 \
  --no-run-baseline
