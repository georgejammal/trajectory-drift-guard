#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

wait_for_live_ccocr() {
  while pgrep -f "scripts/run_search.py --task ccocr" >/dev/null; do
    echo "[$(date -Is)] waiting for existing CC-OCR run to finish..."
    sleep 60
  done
}

run_if_missing() {
  local alias="$1"
  local script="$2"
  local summary="${PARSING_NEURONS_OUTPUT_ROOT:-outputs}/runs/ccocr/${alias}/search_summary.json"

  if [[ -f "$summary" ]]; then
    echo "[$(date -Is)] skip ${alias}; found ${summary}"
    return
  fi

  echo "[$(date -Is)] start ${alias} via ${script}"
  bash "$script"
  echo "[$(date -Is)] done ${alias}"
}

wait_for_live_ccocr

run_if_missing gemma3_4b_it configs/ccocr_gemma3_4b.sh
run_if_missing gemma3_12b_it configs/ccocr_gemma3_12b.sh
run_if_missing qwen2_5_vl_3b_instruct configs/ccocr_qwen2_5_vl_3b.sh
run_if_missing qwen3_vl_8b_instruct configs/ccocr_qwen3_vl_8b.sh
