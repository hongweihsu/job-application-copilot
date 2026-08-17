from fastapi.testclient import TestClient

from app.llm.models import LLMRequirementDecision
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_disables_llm_by_default(monkeypatch):
    monkeypatch.delenv("LLM_ANALYSIS_ENABLED", raising=False)
    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {"llm_analysis_enabled": False}


def test_analyze_validation():
    response = client.post(
        "/api/analyze",
        json={"resume_text": "too short", "job_description": "also too short"},
    )
    assert response.status_code == 422


def test_llm_analysis_returns_service_unavailable_when_disabled(monkeypatch):
    monkeypatch.delenv("LLM_ANALYSIS_ENABLED", raising=False)
    response = client.post(
        "/api/analyze",
        json={
            "resume_text": "Built Python REST API services for three internal teams.",
            "job_description": "Required experience building production Python API services.",
            "analysis_mode": "llm",
        },
    )
    assert response.status_code == 503


def test_llm_api_uses_configured_retriever(monkeypatch):
    monkeypatch.setenv("LLM_ANALYSIS_ENABLED", "true")
    fake_retriever = object()
    calls = {}

    class FakeProvider:
        model_name = "fake-model"

        def __init__(self, config):
            del config

        def decide(self, requirement, evidence_units):
            del requirement, evidence_units
            return LLMRequirementDecision(
                status="missing",
                confidence=0.9,
                evidence_ids=[],
                matched_terms=[],
                explanation="No evidence.",
                recommendation="Do not claim this skill.",
            )

    def fake_analyze(resume_text, job_description, provider, retriever):
        calls["retriever"] = retriever
        from app.analyzer import analyze

        return analyze(resume_text, job_description)

    monkeypatch.setattr("app.main.LLMConfig.from_environment", lambda: object())
    monkeypatch.setattr("app.main.OpenAIRequirementDecisionProvider", FakeProvider)
    monkeypatch.setattr("app.main.build_evidence_retriever", lambda: fake_retriever)
    monkeypatch.setattr("app.main.analyze_with_llm", fake_analyze)

    response = client.post(
        "/api/analyze",
        json={
            "resume_text": "Built production Python APIs for three internal teams.",
            "job_description": "Required experience building production Python API services.",
            "analysis_mode": "llm",
        },
    )

    assert response.status_code == 200
    assert calls["retriever"] is fake_retriever
