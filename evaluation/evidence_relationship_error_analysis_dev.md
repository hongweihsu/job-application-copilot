# Evidence Relationship Error Analysis

This analysis uses only the 28-case dev split. Validation and test were not inspected.

## Outcome

Prompt/schema v2 improved citation precision from 50.0% to 83.3% and reduced the unsupported-claim
heuristic from 3.6% to 0.0%. It nevertheless failed the promotion gate because Accuracy fell from
85.7% to 75.0%, Macro F1 from 82.3% to 67.5%, and citation coverage from 100.0% to 76.2%.

## Root cause

The schema treated `supporting`, `related`, and `contradictory` as mutually exclusive. Some resume
sentences have a mixed relationship: they positively support one material part while explicitly
showing that another part is absent.

Examples:

- Terraform: reviewing production pull requests supports Terraform exposure, while platform-team
  ownership disproves module ownership.
- iOS: contributing fixes supports iOS work, while another engineer's ownership disproves
  independent ownership.
- Kubernetes: running workloads supports deployment experience, while platform-team assistance
  limits independent operations.
- Cloud migration: coordination supports migration involvement, while engineers owning architecture
  and implementation limits technical ownership.

The model placed these mixed sentences in `contradictory_evidence_ids`. The deterministic grounding
guard then downgraded partial decisions with no supporting IDs to missing. That guard behaved as
designed; the relationship schema supplied an incomplete representation.

## Decision

Do not advance v2 to validation. Do not weaken the grounding guard or tune against validation.

The next dev candidate should add an explicit `partial_evidence_ids` relationship:

- `supporting`: directly supports the complete claimed material part;
- `partial`: positively supports part of the requirement but has a material limitation;
- `related`: adjacent context without positive support for the named constraint;
- `contradictory`: pure negation or evidence against the requirement.

Only `supporting` and `partial` should count as citations, with their interpretation conditioned on
the predicted status. Retrieval metrics should remain separate from LLM citation metrics.
