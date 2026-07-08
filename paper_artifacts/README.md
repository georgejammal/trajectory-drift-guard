# Paper Artifact Map

This directory stores the lightweight provenance needed to reproduce the paper
tables and diagnostic figures. It is intentionally not a full cache of every
generation produced during exploration.

## Included

- `figures/`: figures referenced by `main.tex`.
- `sample_sets/`: security sample manifests used for harmful, held-out drift,
  benign MCQA, and open-ended benign QA evaluations.
- `security_main_eval/`: Gemma-3-4B-IT and Llama-3.2-3B-Instruct safety
  manifests, native summaries, and judge summaries for the main harmful ASR
  table. Raw harmful completions are intentionally not included.
- `security_benign_utility/`: benign utility manifests, completions, and
  summaries for Gemma and Llama.
- `security_drift/`: held-out safety drift summaries/features, pre-MLP
  refusal diagnostics, and negative-absolute-value safety ablation outputs.
- `counting_drift/`: counting drift diagnostic summaries and manifests used in
  the appendix.
- `ocr_drift/`: one-token Arabic OCR drift diagnostic summaries and manifests
  used in the appendix.
- `legacy_scripts/`: scripts from the earlier task-specific folders that
  generated the security, counting-drift, and OCR-drift diagnostics.

The clean shared implementation for current counting and multilingual OCR runs
lives under `src/parsing_neurons_repro/` and `scripts/`. The exact sample IDs
and filtering logic for the current paper tables live in:

- `outputs/paper_tables/benchmark_sample_manifest.json`
- `outputs/paper_tables/best_multilingual_ocr_configs.json`
- `outputs/appendix/null_cosine/`
- selected `outputs/runs/**/search_summary.json`,
  `outputs/runs/**/run_summary.json`, and `outputs/runs/**/generation_summary.json`
  files.

## Excluded

GCG artifacts are intentionally excluded from this backup, following the current
paper scope. Raw harmful completions, large raw datasets, and model checkpoints
are also excluded; the repo stores dataset identifiers, filter rules, sample
indices, and scored summaries rather than publishing full external benchmark
data or harmful generations.

## Paper Mapping

- `tab:pre-mlp-refusal-margin`: `security_drift/pre_mlp_residual_failed_refusal_corrected_20260623/`
- `tab:security-window-drift-auc`: `security_drift/heldout_malware_promptinject_seed0_all_unused_baseline/`
- `fig:safety-drift-failed-success`: `figures/window_signed_drift_failed_vs_success.png`
- `tab:safety-two-judges-no-prefilter`: `security_main_eval/`
- `tab:ablation-neg-abs`: `security_drift/first100x4_neg_abs_sign_test/`
- `tab:visual-counting-compact`: `outputs/runs/counting/**/search_summary.json`
- `fig:counting-drift`: `figures/counting_drift.png` and `counting_drift/`
- `tab:multilingual-ocr`: `outputs/paper_tables/best_multilingual_ocr_configs.json`,
  `outputs/runs/ccocr/**/search_summary.json`, and
  `outputs/runs/mdpbench/**/generation_summary.json`
- `tab:safety-benign-utility`: `security_benign_utility/`
- `fig:safety-drift-benign-control`: `figures/window_signed_drift_benign_success_failed.png`
- `tab:empirical-null-cosines`: `outputs/appendix/null_cosine/`
- `tab:appendix-main-sample-counts`: `outputs/paper_tables/benchmark_sample_manifest.json`
- `tab:appendix-diagnostic-sample-counts`: `outputs/paper_tables/benchmark_sample_manifest.json`
- `tab:appendix-counting-drift`: `counting_drift/`
- `tab:appendix-ocr-drift`: `ocr_drift/`
