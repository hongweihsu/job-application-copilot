# BM25 Word Normalization Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)

This experiment normalizes a pre-registered set of common word forms in both queries and resume evidence. It does not use stopwords, aliases for domain concepts, or embeddings.

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_raw | 40.5% | 76.2% | 84.9% | 25.7% | 95.2% | 67.0% | 100.0% | 4.61 |
| bm25_normalized | 44.4% | 79.4% | 88.9% | 26.3% | 95.2% | 74.0% | 100.0% | 4.75 |

## Delta versus raw BM25

- Recall@5: +4.0%
- Precision@5: +0.6%
- MRR: +7.0%
- Average candidates: +0.14

## Decision

Decision: **candidate for validation**.

The decision requires Recall@5 to remain at least equal to raw BM25 and either MRR or Precision@5 to improve.

Observed dev gains include AWS observability moving from first relevant rank 2 to rank 1, complete retrieval of both incident-detection evidence units, mobile monitoring moving from rank 3 to rank 1, and an additional MLOps training evidence unit entering Top-5.

Validation and test were not evaluated.
