#!/usr/bin/env bash
set -euo pipefail

cd /home/georgejammal/projects/parsing_neurons_repro

WAIT_PID_FILE="${WAIT_PID_FILE:-logs/mdpbench_residual_add_alpha1.pid}"
PYTHON="${PYTHON:-/home/georgejammal/projects/a100env/bin/python}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

if [[ -s "$WAIT_PID_FILE" ]]; then
  WAIT_PID="$(cat "$WAIT_PID_FILE")"
  echo "[scheduler] waiting for residual MDPBench PID ${WAIT_PID}"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$SLEEP_SECONDS"
  done
  echo "[scheduler] residual MDPBench PID ${WAIT_PID} finished"
else
  echo "[scheduler] no residual MDPBench PID file found; starting immediately"
fi

PYTHONPATH=src:scripts "$PYTHON" -u scripts/run_scalar_activation_ablation.py \
  --tasks counting,ccocr,mdpbench \
  --scalar-modes zero,negative_abs,relu,scaled_abs_1p2
