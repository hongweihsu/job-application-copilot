# Embedding Retrieval Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)
Embedding model: `text-embedding-3-small`

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_raw | 40.5% | 76.2% | 84.9% | 25.7% | 95.2% | 67.0% | 100.0% | 4.61 |
| embedding | 73.8% | 88.1% | 96.0% | 27.6% | 100.0% | 94.0% | 100.0% | 5.00 |

## Delta versus raw BM25

- Recall@5: +11.1%
- Precision@5: +1.9%
- MRR: +27.1%

## Embedding usage

- API requests: 28
- Input tokens: 1398
- Cached resume evidence units: 80

Document embeddings are cached by exact text for the duration of the run. Each dev resume is embedded once even though it is evaluated against four requirements.

## Decision

Decision: **candidate for validation**.

A standalone candidate must preserve raw BM25 Recall@5 and improve MRR or Precision@5. Even if it fails this gate, case-level complementary wins may still justify a later hybrid experiment.

Dev improvements include semantic stakeholder communication, Azure scope, mobile performance, expanded SIEM evidence, and the RAG prototype case. Remaining partial misses include one MLOps training unit and one cross-functional leadership unit.

Validation and test were not evaluated.
