# LLM Retrieval Pipeline Comparison

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)
LLM: `gpt-5-mini`; final Top-k: 5

## End-to-end results

| Pipeline | Accuracy | Macro F1 | Citation precision | Citation coverage | False supported | Unsupported claims | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_llm | 78.6% | 76.9% | 54.5% | 95.2% | 0.0% | 0.0% | 11.08s | 17.27s |
| embedding_llm | 85.7% | 82.3% | 50.0% | 100.0% | 0.0% | 3.6% | 13.06s | 17.32s |

## Actual API usage

- bm25_llm: 28 LLM requests, 9779 input tokens, 25926 output tokens
- embedding_llm: 28 LLM requests, 10065 input tokens, 26184 output tokens
- embedding: 28 requests, 1398 input tokens, 80 cached evidence units

## Decision

Decision: **needs error analysis**.

- Classification improved: True
- Citation precision preserved: False
- Safety preserved: False

Only dev was evaluated. Validation and test remain untouched until the end-to-end candidate and decision gate are reviewed.
