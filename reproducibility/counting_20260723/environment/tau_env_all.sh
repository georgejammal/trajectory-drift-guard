#!/usr/bin/env bash

export PROJECT_ROOT="/specific/scratches/scratch/georgejammal/trajectory-drift-guard"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export PARSING_NEURONS_DATA_ROOT="$PROJECT_ROOT/data"
export PARSING_NEURONS_OUTPUT_ROOT="$PROJECT_ROOT/outputs"
export GEMMA3_4B_IT_PATH="$PROJECT_ROOT/models/gemma-3-4b-it"
export GEMMA3_12B_IT_PATH="$PROJECT_ROOT/models/gemma-3-12b-it"
export QWEN25VL_3B_INSTRUCT_PATH="$PROJECT_ROOT/models/Qwen2.5-VL-3B-Instruct"
export QWEN3VL_8B_INSTRUCT_PATH="$PROJECT_ROOT/models/Qwen3-VL-8B-Instruct"
export LLAMA32_3B_INSTRUCT_PATH="$PROJECT_ROOT/models/Llama-3.2-3B-Instruct"
export TOKENIZERS_PARALLELISM=false

source "$PROJECT_ROOT/.venv-all/bin/activate"
cd "$PROJECT_ROOT"
