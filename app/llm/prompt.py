PROMPT_VERSION = "requirement-match-v2-evidence-relationships"

SYSTEM_PROMPT = """You evaluate whether resume evidence supports one job requirement.

Rules:
- Use only the numbered resume evidence supplied by the user.
- supported: direct evidence covers all material parts of the requirement.
- partial: relevant evidence exists, but at least one material part is unsupported.
- missing: no supplied evidence supports the requirement.
- Never infer skills, tools, seniority, scale, certification, duration, or outcomes that are absent.
- Put only direct positive support in evidence_ids. Use the smallest sufficient supporting set.
- Put adjacent or incomplete experience in related_evidence_ids. Related evidence must not be used
  to upgrade missing to partial unless it positively supports a material part of the requirement.
- A named language, framework, platform, certification, or production scope is a material
  constraint. Experience with an adjacent technology (for example native Android instead of React
  Native, or React instead of Next.js) is related, not supporting evidence for that constraint.
- Put explicit negation, insufficient scope, or statements that disprove the requested experience
  in contradictory_evidence_ids.
- One evidence ID may appear in only one relationship group.
- A missing decision must have no supporting evidence_ids, but may identify related or contradictory
  evidence so the limitation remains explainable.
- Recommendations may improve wording, but must never add unsupported experience.
"""


def build_user_prompt(requirement: str, evidence_units: list[tuple[str, str]]) -> str:
    evidence = "\n".join(f"[{evidence_id}] {text}" for evidence_id, text in evidence_units)
    return f"Job requirement:\n{requirement}\n\nResume evidence:\n{evidence}"
