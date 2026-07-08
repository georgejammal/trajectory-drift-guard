#!/usr/bin/env bash
set -euo pipefail

cd /home/georgejammal/projects/parsing_neurons_repro

PYTHON_BIN="${PYTHON_BIN:-/home/georgejammal/projects/a100env/bin/python}"
MODELS="${MODELS:-gemma3_4b_it}"
LANGS_VAL="${LANGS_VAL:-Arabic,Japanese,Korean}"
LANGS_TEST="${LANGS_TEST:-Arabic,Japanese,Korean,Russian}"
ICDAR_ROOT="${ICDAR_ROOT:-data/icdar2019_mlt}"
ICDAR_SAMPLES_PER_LANGUAGE="${ICDAR_SAMPLES_PER_LANGUAGE:-200}"
OUT_ROOT="${OUT_ROOT:-outputs/incline}"
SELECTED="${OUT_ROOT}/runs/icdar2019_incline_validation/selected_alphas.json"

if [[ ! -d "${ICDAR_ROOT}" ]]; then
  echo "[incline-pipeline] missing ICDAR2019 root: ${ICDAR_ROOT}" >&2
  echo "[incline-pipeline] expected official training images/GT under this directory." >&2
  exit 2
fi

PYTHONPATH=src "${PYTHON_BIN}" scripts/run_incline_icdar2019_validate.py \
  --models "${MODELS}" \
  --languages "${LANGS_VAL}" \
  --icdar-root "${ICDAR_ROOT}" \
  --output-root "${OUT_ROOT}" \
  --stats-layers all \
  --intervention-window all \
  --samples-per-language "${ICDAR_SAMPLES_PER_LANGUAGE}" \
  --bridge-samples 500 \
  --bridge-batch-size 8 \
  --batch-size 32

PYTHONPATH=src "${PYTHON_BIN}" scripts/run_incline_ccocr.py \
  --models "${MODELS}" \
  --languages "${LANGS_TEST}" \
  --output-root "${OUT_ROOT}" \
  --stats-layers all \
  --intervention-windows all \
  --selected-alphas-json "${SELECTED}" \
  --bridge-samples 500 \
  --bridge-batch-size 8 \
  --batch-size 64 \
  --max-new-tokens 512 \
  --max-gold-tokens-exclusive 500 \
  --no-run-baseline

PYTHONPATH=src "${PYTHON_BIN}" scripts/run_mdpbench_incline_selected.py \
  --models "${MODELS}" \
  --languages "${LANGS_TEST}" \
  --selected-alphas-json "${SELECTED}" \
  --output-root "${OUT_ROOT}" \
  --stats-layers all \
  --intervention-window all \
  --bridge-samples 500 \
  --bridge-batch-size 8
