# Partial Evidence Targeted Error Analysis

This analysis uses five targeted dev cases. Validation and test were not inspected. The first
partial-evidence prompt scored 3/5; an activity-strength revision and a composite-requirement
revision each scored 4/5.

## What improved

Adding `partial_evidence_ids` fixed three failure modes from the previous three-way schema:

- Terraform pull-request review with no module ownership;
- iOS contributions with another engineer retaining ownership;
- Kubernetes workload deployment with platform-team assistance.

All three were classified `partial`, and the expected evidence was placed in
`partial_evidence_ids` rather than being forced into `contradictory_evidence_ids`.

## Remaining failures

Cloud migration remained `missing` in both later revisions. Coordination was treated as related and
the sentence that denied architecture/implementation ownership was treated as contradictory. The
gold label is partial because the original annotation policy treats meaningful participation with
insufficient scope as partial. However, a reasonable annotator could choose missing because none of
the specifically requested technical architecture, implementation, or ownership is demonstrated.

React Native was initially predicted `partial`. After adding the activity-strength rule, both later
runs correctly predicted missing: native Android remained related and the unshipped hackathon
evaluation did not become a citation.

## Decision

The 4/5 smoke result does not qualify for a full dev rerun under the pre-registered 5/5 gate. The
activity-strength rule fixed React Native without regressing the three partial controls. A further
composite-requirement prompt did not fix Cloud migration.

- meaningful execution or delivery participation with limited ownership can be partial;
- reviewing a real work product can be partial when it is part of delivery responsibility;
- merely studying, observing, or evaluating a technology without performing the requested activity
  is related or contradictory, not partial.

Stop prompt tuning at this point. Repeatedly adding instructions for one synthetic case would risk
overfitting. Do not inspect validation, silently change the gold label, exclude the case after seeing
the failure, or weaken the safety control. The Cloud migration case needs annotation adjudication or
additional similar cases before another model change is justified.
