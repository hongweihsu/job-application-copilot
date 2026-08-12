from fastapi.testclient import TestClient

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
