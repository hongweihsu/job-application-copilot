from app.analyzer import analyze, extract_requirements

RESUME = """
Software Engineer
Built Python and FastAPI REST API services used by 3 internal teams.
Created PostgreSQL reporting pipelines that reduced manual processing time by 35%.
Worked with product stakeholders in an Agile team.
"""

JOB = """
We are hiring a backend engineer.
Required: professional experience building Python REST API services.
Experience with PostgreSQL and SQL is required.
AWS and Kubernetes experience are required.
Strong stakeholder communication skills.
"""


def test_extracts_requirements():
    requirements = extract_requirements(JOB)
    assert len(requirements) >= 3
    assert any("Python" in requirement for requirement in requirements)


def test_analysis_never_cites_evidence_for_missing_requirements():
    result = analyze(RESUME, JOB)
    missing = [match for match in result.matches if match.status == "missing"]
    assert missing
    assert all(not match.evidence for match in missing)


def test_analysis_finds_supported_python_evidence():
    result = analyze(RESUME, JOB)
    python_matches = [match for match in result.matches if "Python" in match.requirement]
    assert python_matches
    assert python_matches[0].status == "supported"
    assert "python" in python_matches[0].matched_terms
    assert python_matches[0].evidence[0].evidence_id == "resume-s2"
    assert "Python and FastAPI" in python_matches[0].evidence[0].text


def test_summary_counts_all_matches():
    result = analyze(RESUME, JOB)
    total = result.summary.supported + result.summary.partial + result.summary.missing
    assert total == len(result.matches)
