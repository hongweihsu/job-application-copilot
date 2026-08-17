from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from app.analyzer import analyze
from app.llm.analyzer import analyze_with_llm
from app.llm.config import LLMConfig, llm_analysis_enabled
from app.llm.provider import OpenAIRequirementDecisionProvider
from app.models import AnalysisRequest, AnalysisResponse
from app.retrieval import build_evidence_retriever

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Job Application Copilot",
    description="Evidence-grounded resume and job-description matching",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def public_config() -> dict[str, bool]:
    return {"llm_analysis_enabled": llm_analysis_enabled()}


@app.post("/api/extract-resume")
async def extract_resume(file: UploadFile = File(...)) -> dict[str, str]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF resumes are supported in this MVP.")
    payload = await file.read()
    if len(payload) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The PDF must be smaller than 5 MB.")
    try:
        document = pymupdf.open(stream=payload, filetype="pdf")
        text = "\n".join(page.get_text() for page in document).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The PDF could not be parsed.") from exc
    if len(text) < 30:
        raise HTTPException(
            status_code=422,
            detail="Not enough text was found. Scanned PDFs are not supported yet.",
        )
    return {"text": text}


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_application(request: AnalysisRequest) -> AnalysisResponse:
    if request.analysis_mode == "llm":
        if not llm_analysis_enabled():
            raise HTTPException(status_code=503, detail="LLM analysis is not enabled.")
        try:
            provider = OpenAIRequirementDecisionProvider(LLMConfig.from_environment())
            retriever = build_evidence_retriever()
            return analyze_with_llm(
                request.resume_text,
                request.job_description,
                provider,
                retriever=retriever,
            )
        except OpenAIError as exc:
            raise HTTPException(status_code=502, detail="The LLM provider request failed.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    return analyze(request.resume_text, request.job_description)
