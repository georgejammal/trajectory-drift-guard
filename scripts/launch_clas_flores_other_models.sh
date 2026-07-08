#!/usr/bin/env bash
set -euo pipefail

cd /home/georgejammal/projects/parsing_neurons_repro

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export GEMMA3_4B_IT_PATH="${GEMMA3_4B_IT_PATH:-/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767}"
export GEMMA3_12B_IT_PATH="${GEMMA3_12B_IT_PATH:-/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80}"
export QWEN25VL_3B_INSTRUCT_PATH="${QWEN25VL_3B_INSTRUCT_PATH:-/home/georgejammal/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
export QWEN3VL_8B_INSTRUCT_PATH="${QWEN3VL_8B_INSTRUCT_PATH:-/home/georgejammal/projects/parsing_neurons/models/qwen3-vl-8b-instruct}"

PYTHON="${PYTHON:-/home/georgejammal/projects/a100env/bin/python}"
COMMON=(
  scripts/run_clas_ccocr.py
  --output-root outputs/clas_flores
  --stats-source flores
  --stats-languages Arabic,Japanese,Korean,Russian
  --languages Arabic,Japanese,Korean,Russian
  --alphas=-5,-3,-1,1,3,5
  --betas 0.4
  --gammas 0.2
  --stats-samples 100
  --stats-batch-size 16
  --stats-max-length 512
  --force-recompute-stats
  --max-gold-tokens-exclusive 500
  --max-new-tokens 512
)

run_model() {
  local alias="$1"
  local stats_layers="$2"
  local windows="$3"
  local batch_size="$4"
  shift 4
  echo "=== $(date -Is) ${alias} stats=${stats_layers} windows=${windows} batch=${batch_size} ==="
  "$PYTHON" "${COMMON[@]}" \
    --model-alias "$alias" \
    --stats-layers "$stats_layers" \
    --intervention-windows "$windows" \
    --batch-size "$batch_size" \
    "$@"
}

# Gemma-12B: compute categories over all 48 text-decoder layers, then apply
# CLAS on the last six eligible bridge layers before the final two layers.
run_model gemma3_12b_it 0-47 40-45 16

# CLAS reports Qwen bridge layers 24--25 in a 28-layer model: the two layers
# immediately before the excluded final two layers. Both Qwen-VL variants here
# have 36 text layers, so the analogous bridge window is 32--33.
run_model qwen2_5_vl_3b_instruct 0-35 32-33 32 --qwen-max-pixels 1003520
run_model qwen3_vl_8b_instruct 0-35 32-33 16 --qwen-max-pixels 1003520

echo "=== $(date -Is) all done ==="
