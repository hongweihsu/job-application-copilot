# Evaluation Annotation Guide

This guide defines how to label requirement-to-resume evidence cases. Apply these rules consistently
before comparing deterministic, BM25, vector, hybrid, reranked, or LLM-based approaches.

## Status labels

### `supported`

The resume contains direct evidence covering the material parts of the requirement. The evidence
must describe real work, knowledge, or responsibility rather than merely mentioning an unrelated
keyword.

### `partial`

The resume contains relevant evidence but leaves at least one material part unsupported. Common
examples include demonstrating one of two required technologies, adjacent experience, or an activity
without the requested scope or level.

### `missing`

The resume contains no evidence that supports the requirement. Similar job titles, generic claims,
and unrelated keyword occurrences are not evidence.

## Evidence labels

- Give each resume evidence unit a stable sequential ID: `resume-s1`, `resume-s2`, and so on.
- Include every sentence required to justify the expected status in `expected_evidence_ids`.
- A `missing` case must have an empty `expected_evidence_ids` list.
- Do not label a sentence as relevant solely because it repeats a keyword.
- Prefer the smallest sufficient evidence set.

## Forbidden claims

List positive claims that the system must not generate from the supplied resume. Include skills,
certifications, seniority, scale, and outcomes that are absent from the evidence.

## Dataset splits

- `dev`: inspect freely while implementing and tuning a method.
- `validation`: use to select thresholds and compare candidate configurations.
- `test`: use only for the final frozen comparison reported in the README.

Do not change test labels or tune a method after inspecting individual test failures. When a label is
genuinely wrong, document the correction and publish a new dataset version.

## Quality control

For a larger dataset, two annotators should label cases independently. Resolve disagreements and
record inter-annotator agreement before treating the dataset as a reliable benchmark. The current
synthetic dataset is an initial portfolio benchmark, not a claim of real-world model accuracy.
