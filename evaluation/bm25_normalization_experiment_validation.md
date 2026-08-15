# BM25 Word Normalization Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `validation` (3 profiles, 12 cases)

This experiment normalizes a pre-registered set of common word forms in both queries and resume evidence. It does not use stopwords, aliases for domain concepts, or embeddings.

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_raw | 45.0% | 75.0% | 90.0% | 26.0% | 90.0% | 72.5% | 100.0% | 4.67 |
| bm25_normalized | 45.0% | 75.0% | 90.0% | 26.0% | 90.0% | 72.5% | 100.0% | 4.75 |

## Delta versus raw BM25

- Recall@5: +0.0%
- Precision@5: +0.0%
- MRR: +0.0%
- Average candidates: +0.08

## Decision

Decision: **rejected after validation**.

The decision requires Recall@5 to remain at least equal to raw BM25 and either MRR or Precision@5 to improve.

The normalization mapping was frozen after dev evaluation; no aliases were added or removed based on validation cases.

Test was not evaluated.
