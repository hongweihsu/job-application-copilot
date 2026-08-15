# BM25 Dev Error Analysis

Dataset: `golden_dataset_v2.json`  
Split: `dev` only (7 profiles, 28 cases)  
Retriever: raw BM25 (`k1=1.5`, `b=0.75`, `top_k=5`)

Validation and test results were not inspected for this analysis.

## Baseline

| Metric | Deterministic | Raw BM25 | Delta |
|---|---:|---:|---:|
| Recall@5 | 33.3% | 84.9% | +51.6 pp |
| Precision@5 | 35.7% | 25.7% | -10.0 pp |
| MRR | 38.1% | 67.0% | +28.9 pp |
| No-gold candidate rate | 71.4% | 100.0% | +28.6 pp |
| Average candidates | 0.54 | 4.61 | +4.07 |

Raw BM25 substantially improves coverage and first-relevant ranking, but retrieves almost the full
Top-5 for every query. Its main weakness is noisy candidate selection rather than complete failure
to find lexical matches.

## Answerable-case coverage

There are 21 dev cases with supporting gold evidence.

| Outcome | Cases | Meaning |
|---|---:|---|
| Complete gold set retrieved | 16 | Every labelled supporting evidence unit appears in Top-5 |
| Partial gold set retrieved | 4 | At least one supporting unit is found, but another is missing |
| No gold evidence retrieved | 1 | Top-5 contains no supporting unit |

First relevant evidence ranks:

| Rank | Cases |
|---:|---:|
| 1 | 10 |
| 2 | 6 |
| 3 | 2 |
| 5 | 2 |
| Not found | 1 |

## Recall failures

| Case | Gold | Retrieved gold | Failure mode |
|---|---|---|---|
| `v2-dev-security-incident-detection` | `s4`, `s10` | `s10` | `SIEM analysis` has no lexical connection to Splunk searches and cloud audit logs in `s4` |
| `v2-dev-mobile-performance-reliability` | `s6`, `s9` | `s9` | `performance optimization` does not match reduced startup time, profiling, or lazy initialization in `s6` |
| `v2-dev-ml-production-mlops` | `s2`, `s3`, `s4` | `s4` | training/deployment word forms and the broad `MLOps` concept do not connect all three evidence units |
| `v2-dev-ml-production-generative-ai` | `s5` | none | `RAG` does not match the expanded phrase `retrieval-augmented generation`; the negated prototype sentence ranks first |
| `v2-dev-delivery-agile-leadership` | `s2`, `s7` | `s2` | resolving priority conflicts is semantic leadership evidence without a direct `leading` or `team` match |

Only one answerable case is a complete Top-5 miss. Four failures are multi-evidence cases where BM25
finds one component but cannot connect a paraphrased component.

## No-gold cases

All seven no-gold dev cases return at least one candidate.

| Case | Rank-1 candidate | Why it is not supporting evidence |
|---|---|---|
| `v2-dev-backend-kubernetes` | `s6` | Explicitly says Kubernetes clusters were not operated |
| `v2-dev-frontend-nextjs` | `s9` | Proof-of-concept review; explicitly not built or shipped |
| `v2-dev-data-machine-learning` | `s8` | Explicitly denies training, deployment, and monitoring in production |
| `v2-dev-security-kubernetes-admin` | `s5` | Dashboard review without cluster administration |
| `v2-dev-mobile-react-native` | `s5` | Hackathon evaluation without a customer release |
| `v2-dev-delivery-budget-ownership` | `s5` | Invoice tracking without budget or forecast ownership |
| `v2-dev-delivery-software-engineering` | `s9` | Explicitly denies production software engineering experience |

These rank-1 candidates are usually contradictory or scope-limited, not random irrelevant text.
Lexical retrieval is behaving as designed: the sentences are highly relevant to the query topic.
Removing them at retrieval time could deprive the downstream LLM of useful negative evidence.
Therefore no-gold candidate rate should not be optimized in isolation until the dataset separately
labels supporting, contradicting, and irrelevant evidence.

## Root causes

### 1. Query boilerplate creates low-value matches

Terms such as `experience`, `required`, `must`, `hands-on`, `professional`, `strong`, `ability`, and
`recent` add little technical meaning. They allow generic profile and work-history sentences to
enter Top-5. This contributes to low Precision@5 and the high average candidate count.

Scope-bearing terms such as `production`, `ownership`, `administration`, and `formal` should not be
treated as ordinary stopwords. They distinguish supported from partial or missing requirements,
even though raw BM25 cannot fully reason about that distinction.

### 2. Word-form mismatch loses supporting evidence

Examples include:

- `incident` versus `incidents`;
- `train` / `training` / `trained`;
- `deploy` / `deployment` / `deployed`;
- `monitor` / `monitoring` / `monitored`;
- `application` versus `applications`.

A transparent stemming or lemmatization layer may improve recall and ranking, but must be measured
because aggressive stemming can merge unrelated technical terms.

### 3. Abbreviations and domain concepts require semantic normalization

Examples include:

- `RAG` versus `retrieval-augmented generation`;
- `SIEM` versus Splunk and audit-log investigation;
- `MLOps` versus training, deployment, drift monitoring, rollback, and model review;
- mobile performance versus profiling, startup time, and lazy initialization.

Some can be handled with a small documented alias map. Broad concepts such as MLOps and leadership
are better candidates for embedding retrieval because a large hand-written ontology would be hard
to maintain and easy to overfit.

### 4. Negation and scope are not lexical-ranking problems

BM25 correctly ranks sentences containing the requested technology even when they say `not`,
`without`, `prototype`, `tutorial`, or `did not own`. Stopwords and stemming cannot determine
whether those sentences support or contradict a requirement. This needs a contradiction-aware
reranker or the downstream structured LLM decision.

### 5. Top-5 maximizes recall but lowers precision

The dev profiles contain 10–12 evidence units and BM25 returns 4.61 candidates on average. Many
queries have only one or two supporting units, so even a successful Top-5 naturally produces low
supporting Precision@5. Later evaluation should compare Top-3 and Top-5, but Top-k must not be tuned
until normalization changes are measured.

## Prioritized experiments

1. Add conservative query stopwords for boilerplate terms only.
2. Add lightweight word-form normalization with unit tests for technical tokens.
3. Rerun the raw and improved BM25 variants on dev without overwriting the raw baseline.
4. Accept a change only if Recall@5 remains at least 84.9% while MRR or Precision@5 improves.
5. Use validation once the dev implementation is stable; do not inspect test.
6. Add embedding retrieval for semantic concepts that aliases cannot safely cover.
7. Add contradiction labels before treating no-gold candidate rate as a strict regression gate.

## Initial hypotheses

The first improved BM25 version should test two independent changes:

- `BM25 + stopwords`, to measure noise reduction;
- `BM25 + word normalization`, to measure recall and ranking changes.

Only after measuring them separately should they be combined. This preserves attribution: if a
metric changes, the report can identify which intervention caused it.
