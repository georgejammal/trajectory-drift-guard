# Counting Reproducibility Bundle

This directory freezes the code and verified metadata used for the current
visual-counting experiments. It is a reproducibility artifact, not a new
implementation: do not edit files under `code/` during a rerun.

## Scope

- **Task:** visual counting with absolute-value interventions on selected MLP
  and attention-output coordinates.
- **Test benchmarks:** CountBenchQA (437), HowManyQA (440), and balanced
  TallyQA (450), all restricted to labels 1--9.
- **Validation:** disjoint HowManyQA (160) and TallyQA (180) subsets.
- **Models with completed validation-to-test pairs:** Gemma-3-4B-IT,
  Gemma-3-12B-IT, and Qwen3-VL-8B-Instruct.

## Exact Method

1. Construct `word-minus-digit` from input embeddings for digits 0--9.
   Qwen models use pooled English and Chinese number words; Gemma uses English.
2. Score MLP `down_proj` columns and attention `o_proj` columns against the
   direction using `cosine >= sigma / sqrt(d_model)`.
3. Replace selected MLP activations at all positions and selected attention
   output activations at the final prompt position with their absolute value.
4. Generate greedily (`do_sample=False`) for at most 16 tokens with the fixed
   suffix ` Answer with one number do not add any further explanations.`
5. Report parsed numerical accuracy. Qwen uses `max_pixels=401408`; Gemma has
   no Qwen image-pixel setting.

## Run a Reproduction

From the repository root, copy the `code/` snapshot only if you need to run
the frozen version outside this checkout. In this checkout, the canonical
paths are unchanged and the wrapper below can be submitted directly:

```bash
sbatch reproducibility/counting_20260723/slurm/reproduce_counting.sh \
  gemma3_4b_it 32,64,96,128 17-23 en
```

The wrapper probes a safe batch size, then runs the complete test sweep. Use
`scripts/run_counting_validation_sweep_after_probe.sh` to select a window and
sigma on validation data before reporting a final test configuration.

## What Is and Is Not Reconstructible

- `provenance.json` captures pinned package versions, code hashes, split
  seeds, deterministic decoding, manifests, and all known Slurm provenance.
- The current reproducible environment is Python 3.11.5, PyTorch 2.6.0+cu124,
  CUDA runtime 12.4, and Transformers 4.57.1.
- Historical A100 CUDA, driver, PyTorch, and Transformers versions were never
  saved. They cannot be reconstructed from GitHub or this checkout.
- Several completed Gemma/Qwen3 sweeps did not persist Slurm job IDs or GPU
  model names; these are intentionally recorded as `unavailable` rather than
  guessed. The fully recorded Qwen2.5 L40S run is preserved in `results/`.

## Contents

- `code/`: frozen counting scripts and core modules.
- `environment/`: activation script and fully pinned package list.
- `data_manifests/`: shared OCR manifests retained because the same environment
  and repository release also produced the OCR runs.
- `results/`: recorded Qwen2.5 L40S intervention metadata.
- `provenance.json`: machine-readable reproducibility record.
- `results_validation_and_test.tsv`: completed selection/test pairs.
- `slurm/`: a portable TAU Slurm submission template.
