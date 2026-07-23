#!/usr/bin/env bash
# Usage: sbatch reproducibility/counting_20260723/slurm/reproduce_counting.sh MODEL CANDIDATES WINDOW LANGUAGE
# Example: sbatch reproducibility/counting_20260723/slurm/reproduce_counting.sh gemma3_4b_it 32,64,96,128 17-23 en
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a5000|a6000|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/repro_counting_%j.out
#SBATCH --error=logs/repro_counting_%j.err

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL CANDIDATES WINDOW LANGUAGE" >&2
  exit 2
fi

ROOT="/specific/scratches/scratch/georgejammal/trajectory-drift-guard"
cd "$ROOT"
source "$ROOT/tau_env_all.sh"

MODEL="$1"
CANDIDATES="$2"
WINDOW="$3"
LANGUAGE="$4"
OUTPUT_ROOT="$ROOT/outputs/reproductions/counting_20260723"

mkdir -p "$OUTPUT_ROOT" logs
bash scripts/run_counting_sweep_after_probe.sh "$MODEL" "$CANDIDATES" "$WINDOW" "$LANGUAGE" "$OUTPUT_ROOT"
