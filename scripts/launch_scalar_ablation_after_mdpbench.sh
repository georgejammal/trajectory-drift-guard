#!/usr/bin/env bash
set -euo pipefail

cd /home/georgejammal/projects/parsing_neurons_repro

mkdir -p logs
ts="$(date -u +%Y%m%d_%H%M%S)"
log="logs/scalar_activation_ablation_after_mdpbench_${ts}.log"

setsid bash scripts/wait_then_run_scalar_activation_ablation.sh > "$log" 2>&1 < /dev/null &
pid="$!"

echo "$pid" > logs/scalar_activation_ablation_after_mdpbench.pid
echo "$log" > logs/scalar_activation_ablation_after_mdpbench.logpath
echo "$pid"
echo "$log"
