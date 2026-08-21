# Partial Evidence Targeted Smoke Test

Dataset: `golden_dataset_v2.json`
Evaluated split: `dev` (5 targeted cases)
Prompt: `requirement-match-v3-partial-evidence`

## Case results

| Case | Expected | Predicted | Supporting | Partial | Related | Contradictory |
|---|---|---|---|---|---|---|
| v2-dev-security-terraform-ownership | partial | partial | — | resume-s6 | resume-s2, resume-s3, resume-s5, resume-s1 | — |
| v2-dev-mobile-ios-ownership | partial | partial | — | resume-s4 | resume-s1, resume-s2, resume-s5, resume-s11 | — |
| v2-dev-ml-kubernetes-operations | partial | partial | — | resume-s8 | resume-s3, resume-s12, resume-s11, resume-s4 | — |
| v2-dev-delivery-cloud-migration | partial | missing | — | — | resume-s3, resume-s10, resume-s2, resume-s8 | resume-s4 |
| v2-dev-mobile-react-native | missing | partial | — | resume-s5 | resume-s1, resume-s2, resume-s12, resume-s11 | — |

## Decision

Decision: **needs error analysis**.

- Correct cases: 3/5
- Partial relationships correct: False
- Missing safety control correct: False

## API usage

- LLM: 5 requests, 3199 input tokens, 5543 output tokens
- Embedding: 5 requests, 642 input tokens

This smoke test is only a precondition for a full dev evaluation. Validation and test were not evaluated.

A partial relationship also needs an activity-strength boundary. Meaningful delivery participation with limited ownership may be partial, while merely evaluating, observing, or studying a technology without doing the requested work is not.
