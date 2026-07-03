# Parsing Neurons Reproduction

This repository contains clean experiment code for the static parsing-neuron
experiments used in the paper draft. It focuses on the experiments that produce
paper-facing results:

- visual counting on CountBenchQA, HowManyQA, and TallyQA;
- multilingual OCR on CC-OCR for Arabic, Japanese, Korean, and Russian.

Safety uses the same direction, static scoring, and intervention abstractions,
but is intentionally not wired into this first cleanup pass.

## Core Idea

For a task-defined parsing direction `d` in embedding space, we statically score
model components by cosine similarity with `d`.

- **MLP neuron**: one column of an MLP down-projection matrix. During inference
  it receives a scalar gated activation `a`, so its additive contribution is
  `a * w`.
- **Attention channel**: one column of an attention output projection `o_proj`.
  During inference it receives one scalar coordinate from the attention output,
  so its additive contribution is also scalar times fixed vector.

Under the null hypothesis that a component vector is unrelated to the parsing
direction, the cosine is approximately distributed with standard deviation
`1 / sqrt(d_model)`. The search selects components whose cosine is at least
`sigma / sqrt(d_model)`, with sigma values usually swept from `2.5` upward in
steps of `0.5`.

The intervention is the same everywhere: for selected scalar coordinates, apply
`a <- |a|` at inference time. Counting and OCR use MLP-only and MLP+attention
variants.

## Install

```bash
cd trajectory-drift-guard
python -m pip install -e .
python -m pip install qwen-vl-utils
```

For static component scoring, models must be available as local Hugging Face
snapshot directories containing `config.json` and `model.safetensors.index.json`.
Set these paths with:

- `GEMMA3_4B_IT_PATH`
- `GEMMA3_12B_IT_PATH`
- `QWEN25VL_3B_INSTRUCT_PATH`
- `QWEN3VL_8B_INSTRUCT_PATH`

Dataset defaults are resolved under `PARSING_NEURONS_DATA_ROOT`, defaulting to
`data/`. The expected entries are:

- `data/countbenchqa`
- `data/how_many`
- `data/tallyqa_balanced`
- `data/cc_ocr_dataset`
- `data/flores_transfer_pairs/flores101_en_to_cc_ocr_languages_random500/pairs_by_language`

Outputs are written under `PARSING_NEURONS_OUTPUT_ROOT`, defaulting to
`outputs/`.

## Run Searches

Counting, Gemma-3-4B:

```bash
bash configs/counting_gemma3_4b.sh
```

Counting, Qwen3-VL-8B:

```bash
bash configs/counting_qwen3_vl_8b.sh
```

CC-OCR, Gemma-3-12B:

```bash
bash configs/ccocr_gemma3_12b.sh
```

A direct command has the form:

```bash
PYTHONPATH=src python3 scripts/run_search.py \
  --task counting \
  --model-alias gemma3_4b_it \
  --direction word-minus-digit \
  --counting-language en \
  --windows 17-27,17-25 \
  --sigmas 2.5,3,3.5,4 \
  --component-modes mlp,mlp_attn
```

Outputs are written under `outputs/`:

- `outputs/selections/...`: static MLP/attention selections;
- `outputs/runs/...`: generations, metrics, and CC-OCR official summaries.

## Code Layout

- `src/parsing_neurons_repro/directions.py`: counting and FLORES directions.
- `src/parsing_neurons_repro/scoring.py`: static cosine scoring under the null.
- `src/parsing_neurons_repro/interventions.py`: shared absolute-value hooks.
- `src/parsing_neurons_repro/generation.py`: Gemma/Qwen generation wrappers.
- `src/parsing_neurons_repro/tasks/counting.py`: counting datasets and metrics.
- `src/parsing_neurons_repro/tasks/ccocr.py`: CC-OCR generation and official evaluation.
- `src/parsing_neurons_repro/search.py`: window/sigma/component search orchestration.
