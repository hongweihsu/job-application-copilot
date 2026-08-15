# BM25 Stopword Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)

This experiment removes conservative boilerplate terms from the query only. Resume document tokens and BM25 parameters are unchanged.

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | Hit@5 | MRR | No-gold candidate rate | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25_raw | 40.5% | 76.2% | 84.9% | 25.7% | 95.2% | 67.0% | 100.0% | 4.61 |
| bm25_stopwords | 40.5% | 69.0% | 73.0% | 41.8% | 85.7% | 66.8% | 100.0% | 2.64 |

## Delta versus raw BM25

- Recall@5: -11.9%
- Precision@5: +16.1%
- MRR: -0.2%
- Average candidates: -1.96

## Decision

**Rejected as a standalone replacement for raw BM25.** Precision improves and candidate volume falls, but Recall@5 drops below the pre-registered 84.9% floor.

Four cases lose gold evidence: AWS observability, stakeholder communication, business partnering, and mobile performance. In several of them, raw BM25 found semantic gold evidence only through accidental matches on words such as `and` or `with`. Filtering exposes the semantic gap but does not solve it.

The stopword variant remains useful as an experimental component and may be combined with word normalization or embedding retrieval later. It is not wired into the LLM pipeline.

Scope-bearing terms including `production`, `ownership`, `formal`, and `administration` are intentionally retained.

Validation and test were not evaluated.
