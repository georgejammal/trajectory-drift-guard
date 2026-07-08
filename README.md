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

## CLAS Baseline

The repository also includes a reimplementation of Cross-Lingual Activation
Steering (CLAS) for comparison as an inference-time multilingual baseline. CLAS
first estimates stable MLP-neuron categories from 100 parallel XQuAD examples
across the 12 languages used in the paper:

- `partial_shared`: active for more than one, but not all, languages;
- `language_specific`: active for exactly one language;
- `all_shared` and `dead`: tracked for analysis but not steered by default.

During inference, CLAS modifies the gated MLP intermediate activation entering
`down_proj`:

```text
h1 = h * (1 + beta * M_partial)
h2 = h1 * (1 - gamma * M_specific)
h_final = (1 - alpha) * h + alpha * h2
```

This is implemented in `src/parsing_neurons_repro/clas.py` and
`CLASIntervention` in `src/parsing_neurons_repro/interventions.py`.

Example CC-OCR run:

```bash
PYTHONPATH=src python3 scripts/run_clas_ccocr.py \
  --model-alias gemma3_4b_it \
  --stats-source xquad \
  --stats-layers 17-33 \
  --intervention-windows 17-29,19-31 \
  --alphas -5,-3,-1,1,3,5 \
  --betas 0.4 \
  --gammas 0.2 \
  --languages Arabic,Japanese,Korean,Russian \
  --max-gold-tokens-exclusive 500
```

The CLAS paper reports a grid over
`beta in {0.2, 0.4, 0.6}` and `gamma in {0.1, 0.2, 0.4}`, selecting
`beta=0.4`, `gamma=0.2`. The released paper source states that `alpha` is tuned
separately on 200 samples per language and gives `alpha=-5.0` as an example,
but does not enumerate the full alpha grid; the wrapper therefore exposes
`--alphas` and defaults to a symmetric grid around the documented value.

Outputs are written under `outputs/runs/ccocr_clas/...`, and CLAS category
files are written under `outputs/clas/stats/...`.

## INCLINE Baseline

The repository links the original INCLINE code as a submodule under
`external/INCLINE` for reference only. We do not modify it. The runnable
CC-OCR adaptation is in
`src/parsing_neurons_repro/incline.py` and `scripts/run_incline_ccocr.py`.

This wrapper reproduces INCLINE's core operation: from parallel FLORES
English/target-language text, it collects per-layer MLP module outputs, fits a
least-squares map from target-language MLP outputs to English MLP outputs, and
patches the MLP output at the final prompt token during CC-OCR generation.
Bridge examples longer than `--bridge-max-length` are skipped rather than
truncated, matching the released INCLINE scripts, and bridge factors are stored
in float32.

Example run:

```bash
PYTHONPATH=src /home/georgejammal/projects/a100env/bin/python scripts/run_incline_ccocr.py \
  --models gemma3_4b_it \
  --languages Arabic,Japanese,Korean,Russian \
  --stats-layers all \
  --intervention-windows all \
  --sigmas=-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --bridge-samples 500 \
  --max-gold-tokens-exclusive 500
```

For the paper comparison, use the validation-first protocol. INCLINE's
alignment maps are learned from 500 FLORES English-target pairs, the
intervention strength is selected on 200 ICDAR2019 RRC-MLT validation images
for Arabic/Japanese/Korean, and the Russian strength is set to the mean selected
value because ICDAR2019 does not include Russian. The selected strengths are
then evaluated on CC-OCR and MDPBench.

If the official RRC download is unavailable, materialize the Hugging Face
`Yesianrohn/OCR-Data` `MLT2019` mirror into the same ICDAR-style layout:

```bash
PYTHONPATH=src /home/georgejammal/projects/a100env/bin/python \
  scripts/materialize_icdar2019_mlt_hf.py \
  --output-root data/icdar2019_mlt_hf \
  --languages Arabic,Japanese,Korean \
  --samples-per-language 200
```

```bash
ICDAR_ROOT=data/icdar2019_mlt_hf \
MODELS=gemma3_4b_it \
bash configs/incline_icdar2019_validate_then_test.sh
```

## Code Layout

- `src/parsing_neurons_repro/directions.py`: counting and FLORES directions.
- `src/parsing_neurons_repro/scoring.py`: static cosine scoring under the null.
- `src/parsing_neurons_repro/interventions.py`: shared absolute-value hooks.
- `src/parsing_neurons_repro/clas.py`: CLAS category estimation baseline.
- `src/parsing_neurons_repro/incline.py`: INCLINE least-squares MLP-output baseline.
- `src/parsing_neurons_repro/generation.py`: Gemma/Qwen generation wrappers.
- `src/parsing_neurons_repro/tasks/counting.py`: counting datasets and metrics.
- `src/parsing_neurons_repro/tasks/ccocr.py`: CC-OCR generation and official evaluation.
- `src/parsing_neurons_repro/search.py`: window/sigma/component search orchestration.
