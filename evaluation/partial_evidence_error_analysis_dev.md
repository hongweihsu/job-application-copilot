# Partial Evidence Targeted Error Analysis

This analysis uses five targeted dev cases. Validation and test were not inspected.

## What improved

Adding `partial_evidence_ids` fixed three failure modes from the previous three-way schema:

- Terraform pull-request review with no module ownership;
- iOS contributions with another engineer retaining ownership;
- Kubernetes workload deployment with platform-team assistance.

All three were classified `partial`, and the expected evidence was placed in
`partial_evidence_ids` rather than being forced into `contradictory_evidence_ids`.

## Remaining failures

Cloud migration was predicted `missing`. Coordination was treated as related and the sentence that
denied architecture/implementation ownership was treated as contradictory. The expected label is
partial because the candidate performed meaningful migration delivery work, although not the
requested technical ownership.

React Native was predicted `partial`. A hackathon evaluation with no shipped application was placed
in `partial_evidence_ids`. The expected label is missing because evaluating a framework is not
application-development experience.

## Decision

The 3/5 smoke result does not qualify for a full dev rerun. The four-way relationship vocabulary is
necessary but not sufficient. The next prompt/schema iteration needs an activity-strength rule:

- meaningful execution or delivery participation with limited ownership can be partial;
- reviewing a real work product can be partial when it is part of delivery responsibility;
- merely studying, observing, or evaluating a technology without performing the requested activity
  is related or contradictory, not partial.

Do not inspect validation or weaken the safety control to improve the score.
