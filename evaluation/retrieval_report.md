# Retrieval Evaluation

Dataset: `golden_dataset.json` (32 cases)

Only cases with labelled evidence are included in retrieval metrics. Missing cases remain in the case-level output but do not have a relevant document to retrieve.

## Summary

| Method | Split | Answerable | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | dev | 8 | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% |
| deterministic | validation | 7 | 57.1% | 57.1% | 57.1% | 71.4% | 71.4% | 71.4% |
| deterministic | test | 7 | 71.4% | 71.4% | 71.4% | 71.4% | 71.4% | 71.4% |
| deterministic | all | 22 | 72.7% | 72.7% | 72.7% | 77.3% | 77.3% | 77.3% |
| bm25 | dev | 8 | 62.5% | 75.0% | 75.0% | 56.2% | 75.0% | 68.8% |
| bm25 | validation | 7 | 42.9% | 57.1% | 57.1% | 64.3% | 71.4% | 64.3% |
| bm25 | test | 7 | 85.7% | 85.7% | 85.7% | 78.6% | 85.7% | 85.7% |
| bm25 | all | 22 | 63.6% | 72.7% | 72.7% | 65.9% | 77.3% | 72.7% |

## Comparison

Across all answerable cases, BM25 changes Recall@5 by +0.0% and MRR by -4.5% versus the deterministic baseline.

These metrics evaluate retrieval only. They do not measure requirement classification, recommendation quality, or LLM generation quality.
