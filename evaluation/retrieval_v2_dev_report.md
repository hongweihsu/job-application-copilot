# Retrieval Evaluation V2

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)

## Summary

| Method | Answerable | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 21 | 33.3% | 33.3% | 33.3% | 35.7% | 38.1% | 38.1% | 71.4% | 0.54 |
| bm25 | 21 | 40.5% | 76.2% | 84.9% | 25.7% | 95.2% | 67.0% | 100.0% | 4.61 |

## Baseline comparison

BM25 changes Recall@5 by +51.6%, Precision@5 by -10.0%, and MRR by +28.9%.

No-gold candidate rate is the percentage of cases with no supporting gold evidence where the retriever still returns candidates. Such candidates may be irrelevant, adjacent, or contradictory; the metric does not by itself distinguish those types.

Only the dev split is intended for retriever error analysis and implementation changes.
