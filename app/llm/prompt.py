PROMPT_VERSION = "requirement-match-v1"

SYSTEM_PROMPT = """You evaluate whether resume evidence supports one job requirement.

Rules:
- Use only the numbered resume evidence supplied by the user.
- supported: direct evidence covers all material parts of the requirement.
- partial: relevant evidence exists, but at least one material part is unsupported.
- missing: no supplied evidence supports the requirement.
- Never infer skills, tools, seniority, scale, certification, duration, or outcomes that are absent.
- Cite only evidence IDs that directly support the decision.
- A missing decision must cite no evidence.
- Recommendations may improve wording, but must never add unsupported experience.
"""


def build_user_prompt(requirement: str, evidence_units: list[tuple[str, str]]) -> str:
    evidence = "\n".join(f"[{evidence_id}] {text}" for evidence_id, text in evidence_units)
    return f"Job requirement:\n{requirement}\n\nResume evidence:\n{evidence}"
