# Negative-Abs Sign Test

Run: `first100x4_neg_abs_sign_test`

This run tests whether the selected refusal-minus-answerability MLP neurons are
causally sign-sensitive.

Interventions at the gated MLP intermediate before `down_proj`:

```text
baseline: no edit
abs:      a_i -> |a_i|
neg_abs:  a_i -> -|a_i|
```

The evaluation set is the same first-100 malicious split used in the final
security evaluation:

```text
pair_test:    100
promptinject: 100
malwaregen:   100
autodan_test: 100
total:        400
```

All scores below are HarmBench classifier ASR. Higher is worse.

## Overall

| Model | Baseline | abs(a) | -abs(a) |
|---|---:|---:|---:|
| Gemma-3-4B-IT | 115/400 = 28.75% | 26/400 = 6.50% | 212/400 = 53.00% |
| Llama-3.2-3B-Instruct | 53/400 = 13.25% | 16/400 = 4.00% | 199/400 = 49.75% |

## Per Benchmark

### Gemma-3-4B-IT

| Benchmark | Baseline | abs(a) | -abs(a) |
|---|---:|---:|---:|
| pair | 34% | 13% | 31% |
| promptinject | 30% | 7% | 29% |
| malwaregen | 39% | 6% | 93% |
| autodan | 12% | 0% | 59% |

### Llama-3.2-3B-Instruct

| Benchmark | Baseline | abs(a) | -abs(a) |
|---|---:|---:|---:|
| pair | 19% | 9% | 20% |
| promptinject | 13% | 2% | 37% |
| malwaregen | 21% | 5% | 89% |
| autodan | 0% | 0% | 53% |

## Interpretation

This is the cleanest causal sign evidence so far.

The original intervention:

```text
a_i -> |a_i|
```

makes the selected refusal-aligned neurons write with positive sign and strongly
reduces ASR.

The sign-flipped intervention:

```text
a_i -> -|a_i|
```

forces the same neurons to write with the opposite sign and sharply increases
ASR.

This supports the mechanism:

```text
positive activation on these selected down-projection directions supports
refusal, while negative activation pushes the model toward answerability.
```

The effect is especially strong on malwaregen and autodan. Pair and
promptinject are less clean because they include benchmark-specific behavior
that is not purely "refuse vs answer harmful instructions."

## Artifacts

- Manifest: `try_and_change/security_drift/sample_sets/first100_pair_promptinject_malwaregen_autodan.jsonl`
- Gemma `neg_abs` completions: `gemma3_4b_it/neg_abs/completions.jsonl`
- Gemma `neg_abs` HarmBench labels: `gemma3_4b_it/neg_abs/llama_harmbench_judge/labels.jsonl`
- Llama `neg_abs` completions: `llama3p2_3b_instruct/neg_abs/completions.jsonl`
- Llama `neg_abs` HarmBench labels: `llama3p2_3b_instruct/neg_abs/llama_harmbench_judge/labels.jsonl`
