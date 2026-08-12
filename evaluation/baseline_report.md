# Deterministic Baseline Evaluation

Dataset: `golden_dataset.json` (32 cases)

## Summary

| Split | Cases | Accuracy | Macro F1 | Recall@5 | Citation precision | False supported |
|---|---:|---:|---:|---:|---:|---:|
| dev | 12 | 91.7% | 91.5% | 87.5% | 100.0% | 0.0% |
| validation | 10 | 70.0% | 66.7% | 57.1% | 100.0% | 0.0% |
| test | 10 | 80.0% | 80.2% | 71.4% | 100.0% | 0.0% |
| all | 32 | 81.2% | 80.7% | 72.7% | 100.0% | 0.0% |

## Held-out test classification

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| supported | 100.0% | 75.0% | 85.7% | 4 |
| partial | 100.0% | 66.7% | 80.0% | 3 |
| missing | 60.0% | 100.0% | 75.0% | 3 |

## Held-out test confusion matrix

| Expected \ Predicted | supported | partial | missing |
|---|---:|---:|---:|
| supported | 3 | 0 | 1 |
| partial | 0 | 2 | 1 |
| missing | 0 | 0 | 3 |

## Interpretation

These numbers describe a transparent keyword-overlap baseline, not production model quality. The held-out test split is reserved for comparisons with BM25, vector, and hybrid retrieval. Dataset examples are synthetic and manually labelled.
