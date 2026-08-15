# Golden Dataset V2

Version 2 is a resume-level retrieval benchmark. It contains thirteen realistic synthetic resume
profiles and fifty-two requirement cases. Each profile has ten to twelve stable evidence units and
belongs entirely to one split, preventing the same resume from leaking across development,
validation, and test data.

## Coverage

- exact technical matches;
- abbreviations and semantic paraphrases;
- requirements supported by multiple evidence units;
- hard negatives involving adjacent technologies or insufficient scope;
- explicit negation and training-only experience;
- supported, partial, and missing classifications.

## Split policy

| Split | Profiles | Cases | Intended use |
|---|---:|---:|---|
| dev | 7 | 28 | Error analysis and retriever changes |
| validation | 3 | 12 | Model and parameter selection |
| test | 3 | 12 | Final comparison after design decisions are frozen |

Retrieval development must use the dev split. Validation can be checked after a proposed change is
stable. The test split should not be used to choose stopwords, aliases, stemming rules, embedding
models, or fusion parameters.

The raw BM25 failure analysis is documented in `bm25_error_analysis_dev.md`. It uses only the dev
split and records the baseline that future retrieval changes must compare against.

## Schema

`profiles` store resume evidence once. `cases` reference a `profile_id`, which avoids duplicating the
same resume for every requirement. `expected_evidence_ids` contain only evidence that positively
supports the requirement. Negated statements and adjacent technologies are hard negatives, not
gold evidence.
