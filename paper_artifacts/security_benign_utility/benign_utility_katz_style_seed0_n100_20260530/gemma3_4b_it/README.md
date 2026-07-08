# Gemma3 Benign Utility Evaluation

Katz-style benign benchmark check for performance degradation.

## Benchmarks
- PIQA validation/dev: `100` deterministic samples.
- OpenBookQA validation: `100` deterministic samples.
- Social IQA validation/dev: `100` deterministic samples.
- ARC-Challenge validation: `100` deterministic samples.

## Generation
- Model: `/home/georgejammal/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767`
- Prompting: Gemma chat template, text-only user message, `add_generation_prompt=True`.
- Decoding: greedy, `do_sample=False`.
- Max new tokens: `512`.
- Batch size: `128`.

## Metrics
- Accuracy: parsed multiple-choice option equals gold option.
- Refusal rate: completion matches the same refusal-prefix heuristic used in the safety scripts,
  with extra benign false-refusal phrases.
- Unparsed rate: no option letter or option text could be extracted from the completion.

## Intervention
- Winning safety intervention: `refusal_minus_answerability_top1000_abs`.
- Site: Gemma language-model MLP gated intermediate before `down_proj`.
- Token scope: all prompt positions and all autoregressive generation steps.
- Neuron set: `/home/georgejammal/projects/semantic-to-symbolic/safeguard_logit_lens/neuron_sets/gemma3_4b_it_refusal_minus_answerability_top1000_layers17_33.json`.
