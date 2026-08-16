# Embedding Retrieval Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `validation` (3 profiles, 12 cases)
Embedding model: `text-embedding-3-small`

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_raw | 45.0% | 75.0% | 90.0% | 26.0% | 90.0% | 72.5% | 100.0% | 4.67 |
| embedding | 45.0% | 85.0% | 100.0% | 28.0% | 100.0% | 77.5% | 100.0% | 5.00 |

## Delta versus raw BM25

- Recall@5: +10.0%
- Precision@5: +2.0%
- MRR: +5.0%

## Embedding usage

- API requests: 12
- Input tokens: 562
- Cached resume evidence units: 32

Document embeddings are cached by exact text for the duration of the run. Each resume is embedded once even though it is evaluated against four requirements.

## Decision

Decision: **approved for promotion**.

A standalone candidate must preserve raw BM25 Recall@5 and improve MRR or Precision@5. Even if it fails this gate, case-level complementary wins may still justify a later hybrid experiment.

The embedding model, cosine similarity, and Top-k were frozen after dev; no configuration was changed using validation results.

Test was not evaluated.
