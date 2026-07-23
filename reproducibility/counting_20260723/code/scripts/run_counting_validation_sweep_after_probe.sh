#!/usr/bin/env bash
set -euo pipefail

model_alias="$1"
candidates="$2"
windows="$3"
counting_language="$4"
output_root="$5"

source tau_env_all.sh

qwen_args=()
if [[ "$model_alias" == qwen* ]]; then
  qwen_args=(--qwen-max-pixels 401408)
fi

probe_path="$output_root/batch_probes/$model_alias.json"
PYTHONPATH=src python scripts/probe_counting_batch.py \
  --model-alias "$model_alias" \
  --candidates "$candidates" \
  --dataset-path data/how_many_validation \
  --output "$probe_path" \
  --max-new-tokens 16 \
  "${qwen_args[@]}"

batch_size="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size"])' "$probe_path")"
echo "[validation-sweep:$model_alias] selected batch_size=$batch_size"

PYTHONPATH=src python scripts/run_search.py \
  --task counting \
  --model-alias "$model_alias" \
  --direction word-minus-digit \
  --counting-language "$counting_language" \
  --windows "$windows" \
  --sigmas 2.5,3,3.5,4,4.5,5,5.5 \
  --component-modes mlp_attn \
  --batch-size "$batch_size" \
  --max-new-tokens 16 \
  --output-root "$output_root" \
  --counting-datasets "how_many_validation=data/how_many_validation,tallyqa_validation=data/tallyqa_validation" \
  "${qwen_args[@]}"
