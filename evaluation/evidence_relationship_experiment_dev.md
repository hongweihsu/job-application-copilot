# Evidence Relationship Prompt Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)
Candidate prompt: `requirement-match-v2-evidence-relationships`

## Results

| Version | Accuracy | Macro F1 | Citation precision | Citation coverage | False supported | Unsupported claims | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 85.7% | 82.3% | 50.0% | 100.0% | 0.0% | 3.6% | 13.06s | 17.32s |
| v2 | 75.0% | 67.5% | 83.3% | 76.2% | 0.0% | 0.0% | 12.99s | 43.22s |

## Candidate API usage

- LLM: 28 requests, 15833 input tokens, 33013 output tokens
- Embedding: 28 requests, 1398 input tokens, 80 cached evidence units

## Decision

Decision: **needs error analysis**.

- Accuracy preserved: False
- Macro F1 preserved: False
- Citation precision improved: True
- Citation coverage preserved: False
- Safety preserved: True

The v1 baseline is read from the previously committed report; it is not rerun. Only dev was evaluated. Validation and test remain untouched.

The candidate improved citation precision and safety but became too conservative. Evidence that both supports one material part and disproves another was forced into a single mutually exclusive relationship group. Several valid partial cases were therefore downgraded to missing. The next schema revision needs explicit partial evidence rather than more prompt tuning.
