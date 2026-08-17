# Hybrid Retrieval Experiment

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (7 profiles, 28 cases)

BM25 and embedding rankings are fused with Reciprocal Rank Fusion (RRF). Raw BM25 scores and cosine similarities are never added directly.

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | Avg candidates |
|---|---:|---:|---:|---:|---:|---:|
| bm25_raw | 40.5% | 76.2% | 84.9% | 25.7% | 67.0% | 4.61 |
| embedding | 73.8% | 88.1% | 96.0% | 27.6% | 94.0% | 5.00 |
| hybrid_rrf | 54.8% | 82.5% | 89.7% | 25.7% | 79.4% | 5.00 |

## Configuration

- RRF k: 60
- Child retrieval depth: 20
- Final Top-k: 5

## Embedding usage

- API requests: 56
- Input tokens: 1679
- Cached resume evidence units: 80

## Decision

Decision: **not better than best component**.

The hybrid must preserve the best component's Recall@5 and improve either MRR or Precision@5. Validation and test were not evaluated.

Equal-weight RRF weakened the stronger embedding ranking. It dropped one MLOps evidence unit, completely lost the generative-AI gold evidence from Top-5, and pushed several otherwise rank-1 results lower. This configuration will not advance to validation.
