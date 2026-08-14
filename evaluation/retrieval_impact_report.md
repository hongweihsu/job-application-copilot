# Retrieval Impact Report

Dataset: `golden_dataset.json` (32 cases)
Tokenizer: `o200k_base` for `gpt-5-mini`

## Retrieval quality

| Comparison | Recall@5 | Delta |
|---|---:|---:|
| Deterministic baseline | 72.7% | — |
| BM25 Top-5 | 72.7% | +0.0% |

## Prompt size

| Pipeline | Evidence units sent | Estimated prompt tokens |
|---|---:|---:|
| Full context | 64 | 5138 |
| BM25 Top-5 | 26 | 4637 |

BM25 reduced evidence units by 59.4% and estimated prompt tokens by 9.8% (501 tokens) while changing Recall@5 by +0.0%.

Token counts include system and user prompt text. They exclude API message framing and model output tokens, so they are reproducible estimates rather than billing data.

The golden cases contain short synthetic resumes, so token savings on longer real resumes may differ.
